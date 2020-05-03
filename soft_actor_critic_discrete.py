import sys
import os
sys.path.append(os.path.dirname('__file__'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import argparse
from torch.distributions.categorical import Categorical
import copy
import time
import math
from tqdm import tqdm

from replay import ReplayBuffer
from networks import FFN, ConvNet
from gym_wrappers import ClipRewardEnv

class Net(nn.Module):

    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()

        layers = []
        for j in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[j], sizes[j+1]))
            if j<(len(sizes)-2):
                layers.append(activation())
            else:
                layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad=False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad=True

    def hard_update(self, source_net):
        """Updates the network parameters by copying the parameters
        of another network
        """
        for target_param, source_param in zip(self.parameters(), source_net.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source_net, tau):
        """Updates the network parameters with a soft update by polyak averaging
        """
        for target_param, source_param in zip(self.parameters(), source_net.parameters()):
            target_param.data.copy_((1-tau)*target_param.data + tau*source_param.data)
    
    def forward(self, x):
        return self.net(x)
        


def choose_action(obs, policy_net, explore=True):
    with torch.no_grad():
        logits = policy_net(obs.float())
    
    policy = Categorical(logits=logits)
    if explore:
        try:
            action = policy.sample()
            policy_law = F.softmax(logits)
            entropy = -torch.sum(torch.log(policy_law+1e-8) * policy_law, 1, keepdim=True) #we avoid numerical problems
            #print(entropy.item())
        except:
            raise ValueError('nans!')
    else:
        probs = F.softmax(logits)
        action=torch.argmax(probs)
    #action = policy.sample()
    return action.item()



def policy_update(batch, policy_net, Q_net1, Q_net2, optimizer_policy, alpha):
    """Policy improvement

    """
    policy_net.zero_grad()
    
    logits = policy_net(batch.states)
    policy_law = F.softmax(logits)
    
    q1 = Q_net1(batch.states)
    q2 = Q_net2(batch.states)
    q = torch.min(q1,q2)
    E_q = torch.sum(q*policy_law, 1, keepdim=True)
    
    entropy = -torch.sum(torch.log(policy_law+1e-8) * policy_law, 1, keepdim=True)

    loss = -torch.mean(E_q + alpha*entropy)
    
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 5)
    optimizer_policy.step()
    #print("updating policy, loss = {}, Mean(E_q)={}, Mean(Entropy)={}".format(loss.item(),E_q.mean().item(), (alpha*entropy).mean().item()))
    return loss.item()



def q_update(batch, policy_net, Q_net1, Q_net2, Q_target_net1, Q_target_net2, optimizer_q, alpha, gamma):
    """
    Policy evaluation
    """
    optimizer_q.zero_grad()

    q1_sa = Q_net1(batch.states).gather(1, batch.actions.to(torch.long))    
    q2_sa = Q_net2(batch.states).gather(1, batch.actions.to(torch.long))    
    
    batch_size = batch.states.shape[0]
    
    logits = policy_net(batch.next_states)
    policy_law_next_s = F.softmax(logits) 
    q1_next_s = Q_target_net1(batch.next_states)
    q2_next_s = Q_target_net2(batch.next_states)
    q_next_s = torch.min(q1_next_s, q2_next_s)
    E_q_next_s = torch.sum(q_next_s * policy_law_next_s, 1, keepdim=True)

    entropy = -torch.sum(torch.log(policy_law_next_s+1e-8) * policy_law_next_s, 1, keepdim=True)
    
    target = batch.rewards + gamma * (1-batch.done) * (E_q_next_s + alpha*entropy)
    
    loss_fn = nn.MSELoss()
    loss = loss_fn(q1_sa, target) + loss_fn(q2_sa, target)
    loss.backward()

    nn.utils.clip_grad_norm_(Q_net1.parameters(), 5)
    nn.utils.clip_grad_norm_(Q_net2.parameters(), 5)
    optimizer_q.step()
    #print(loss.item())
    return loss.item()



def alpha_update(batch, log_alpha, policy_net, optimizer_alpha, entropy_target):
    optimizer_alpha.zero_grad()
    with torch.no_grad():
        logits = policy_net(batch.next_states)
        policy_law = F.softmax(logits)

    entropy = -torch.sum(torch.log(policy_law+1e-8) * policy_law, 1, keepdim=True)
    loss = torch.mean(log_alpha * (entropy - entropy_target))
    loss.backward()
    optimizer_alpha.step()
    return loss


def play_episode(env, 
        policy_net, 
        Q_net1,
        Q_net2,
        Q_target_net1,
        Q_target_net2,
        optimizer_policy,
        optimizer_q,
        optimizer_alpha,
        log_alpha,
        entropy_target,
        gamma,
        replay_buffer, 
        train=True,
        train_policy=False, 
        render=False, 
        max_steps=200, 
        steps_init_training=1000,
        steps_per_learning_update=4,
        batch_size=64, 
        device="cpu"):
    """Play a single episode

    """
    obs = env.reset()
    done = False
    if render:
        env.render()
    episode_timesteps = 0
    episode_return = 0
    
    while not done:
        action = choose_action(torch.from_numpy(obs).unsqueeze(0).to(device), policy_net, explore=train)
        new_obs, reward, done, _ = env.step(action)
        replay_buffer.push(
            np.array(obs, dtype=np.float32),
            np.array([action], dtype=np.float32),
            np.array(new_obs, dtype=np.float32),
            np.array([reward], dtype=np.float32),
            np.array([done], dtype=np.float32)
        )
        
        # updates
        if train and len(replay_buffer)>=steps_init_training:
            if episode_timesteps % steps_per_learning_update==0:
                batch = replay_buffer.sample(batch_size, device)
                Q_net1.unfreeze(); Q_net2.unfreeze(); policy_net.freeze()
                loss_critic = q_update(batch, policy_net, Q_net1, Q_net2, Q_target_net1, Q_target_net2, optimizer_q, torch.exp(log_alpha), gamma)
                
                batch = replay_buffer.sample(batch_size, device)
                Q_net1.freeze(); Q_net2.freeze(); policy_net.unfreeze()
                loss_actor = policy_update(batch, policy_net, Q_net1, Q_net2, optimizer_policy, torch.exp(log_alpha))
            
                batch = replay_buffer.sample(batch_size, device)
                loss_alpha = alpha_update(batch, log_alpha, policy_net, optimizer_alpha, entropy_target)


        episode_timesteps +=1
        episode_return += reward
        
        if render:
            env.render()
        
        if max_steps == episode_timesteps:
            break

        # we are done, prepare for next action
        obs = new_obs
    #try:
    #    print("loss actor {}, loss critic {}".format(loss_actor, loss_critic))
    #except:
    #    pass
    return episode_timesteps, episode_return



def train(env, config):
    """
    Execute training of Soft Actor Critic
    """
    
    timesteps_elapsed = 0
    episodes_elapsed = 0

    STATE_SIZE = env.observation_space.shape[0]
    ACTION_SIZE = env.action_space.n

    policy_net = Net(sizes=(STATE_SIZE, *config["hidden_size"], ACTION_SIZE)).to(config["device"])

    Q_net1= Net(sizes=(STATE_SIZE, *config["hidden_size"], ACTION_SIZE)).to(config["device"])
    Q_net2= Net(sizes=(STATE_SIZE, *config["hidden_size"], ACTION_SIZE)).to(config["device"])
    Q_target_net1 = copy.deepcopy(Q_net1)
    Q_target_net2 = copy.deepcopy(Q_net2)
    Q_target_net1.freeze(); Q_target_net2.freeze()
    
    log_alpha = nn.Parameter(torch.Tensor([math.log(config["alpha"])], device=config["device"]))
    entropy_target = -math.log(1/ACTION_SIZE)*config["target_entropy_ratio"] # that is, maximum entropy times a ratio
    
    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr = config["learning_rate_policy"])
    optimizer_q = torch.optim.Adam(list(Q_net1.parameters())+list(Q_net2.parameters()),
            lr = config["learning_rate_value"])
    optimizer_alpha = torch.optim.Adam([log_alpha], lr = config["learning_rate_alpha"])

    replay_buffer = ReplayBuffer(config["buffer_capacity"])

    eval_returns_all = []
    eval_times_all = []
    
    train_policy=False

    start_time = time.time()
    with tqdm(total=config["max_timesteps"]) as pbar:
        while timesteps_elapsed < config["max_timesteps"]:
            elapsed_seconds = time.time() - start_time
            if elapsed_seconds > config["max_time"]:
                pbar.write("Training ended after {}s.".format(elapsed_seconds))
                break
            
            episode_timesteps, _ = play_episode(
                    env,
                    policy_net=policy_net,
                    Q_net1=Q_net1,
                    Q_net2=Q_net2,
                    Q_target_net1=Q_target_net1,
                    Q_target_net2=Q_target_net2,
                    optimizer_policy=optimizer_policy,
                    optimizer_q=optimizer_q,
                    optimizer_alpha=optimizer_alpha,
                    log_alpha=log_alpha,
                    entropy_target=entropy_target,
                    gamma=config["gamma"],
                    replay_buffer=replay_buffer,
                    train=True,
                    train_policy=train_policy,
                    render=config["render"],
                    max_steps=config["episode_length"],
                    steps_init_training=config["steps_init_training"],
                    steps_per_learning_update=config["steps_per_learning_update"],
                    batch_size=config["batch_size"],
                    device=config["device"]
                )
            timesteps_elapsed += episode_timesteps
            episodes_elapsed += 1
            pbar.update(episode_timesteps)

            if timesteps_elapsed % config["train_policy_freq"] < episode_timesteps:
                train_policy = True
            else:
                train_policy = False

            if timesteps_elapsed % config["target_update_freq"] < episode_timesteps:
                if timesteps_elapsed > config["steps_init_training"]:
                    Q_target_net1.soft_update(Q_net1, 0.5)
                    Q_target_net2.soft_update(Q_net2, 0.5)
            
            if timesteps_elapsed % config["eval_freq"] < episode_timesteps:
                eval_returns = 0
                for _ in range(config["eval_episodes"]):
                    _, episode_return = play_episode(
                            env,
                            policy_net,
                            Q_net1,
                            Q_net2,
                            Q_target_net1,
                            Q_target_net2,
                            optimizer_policy,
                            optimizer_q,
                            optimizer_alpha=optimizer_alpha,
                            log_alpha=log_alpha,
                            entropy_target=entropy_target,
                            gamma=config["gamma"],
                            replay_buffer=replay_buffer,
                            train=False,
                            train_policy=train_policy,
                            render=config["render"],
                            max_steps=config["episode_length"],
                            batch_size=config["batch_size"],
                            device=config["device"]
                        )
                    eval_returns += episode_return
                eval_returns = eval_returns/config["eval_episodes"]
                eval_returns_all.append(eval_returns)
                pbar.write("Evaluation at timestep {} and episode {} returned a mean returns of {}".format(timesteps_elapsed,episodes_elapsed, eval_returns))

                if eval_returns >= config["target_return"]:
                    pbar.write("Reached return {} >= target return of {}".format(eval_returns, config["target_return"]))
                    break
    
    print("Saving policy to {}".format(config["save_filename"]))
    torch.save(policy_net, config["save_filename"])
    
    return np.array(eval_returns_all)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Actor Critic algorithm')
    parser.add_argument('--env_name', '--env', default='MsPacman-ram-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--eval_episodes', default=10)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--clip_reward', action='store_true')

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "device:{}".format(args.device)
    else:
        device="cpu"

    env = gym.make(args.env_name)
    env = ClipRewardEnv(env)
    
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    CONFIG = {
            "env":args.env_name,
            "episode_length":500,
            "max_timesteps":int(2e6),
            "max_time":90*60,
            "buffer_capacity":1e6,
            "eval_freq":1000,
            "eval_episodes":args.eval_episodes,
            "learning_rate_policy":0.0003,
            "learning_rate_value":0.0003,
            "learning_rate_alpha":0.0003,
            "hidden_size":(64,32,16),
            "batch_size":64,
            "target_entropy_ratio":0.98,
            "train_policy_freq":100,
            "target_update_freq":8000,
            "target_return":180,
            "steps_init_training":20000,
            "steps_per_learning_update":4,
            "gamma":0.99,
            "alpha":0.1,
            "device":device,
            "save_filename":"sac_discrete.pth.tar",
            "render":False
            }

    train(env, CONFIG)
    

