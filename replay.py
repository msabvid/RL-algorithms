from collections import namedtuple, deque
import numpy as np
import torch


Transition = namedtuple(
    "Transition", ("states", "actions", "next_states", "rewards", "done")
)



class NstepBuffer:

    def __init__(self, max_steps):

        self.max_steps = max_steps
        self.memory = None

    def init_memory(self, transition: Transition):
        """Initialises the memory with zero-entries
        
        Parameters
        ----------
        transition: Transition
            transition(s) to take the dimensionalities from
        """
        #for t in transition:
        #    assert t.ndim == 1  # sanity check

        self.memory = Transition(
            *[deque() for t in transition]
        )
    
    def push(self, *args):
        """Adds transitions to the memory

        """
        if not self.memory:
            self.init_memory(Transition(*args))

        for i, data in enumerate(args):
            self.memory[i].append(data)

    
    def get_n_step_reward(self, gamma):
        reward = 0
        for i in range(self.max_steps):
            reward += gamma ** i * self.memory.rewards[i]
        return reward

    def pop_left(self):
        obs = self.memory.states.popleft()
        action = self.memory.actions.popleft()
        self.memory.next_states.popleft()
        self.memory.rewards.popleft()
        self.memory.done.popleft()
        return obs, action

    def __len__(self):
        return len(self.memory[0])



class ReplayBuffer:
    """Replay buffer to sample experience/ transition tuples from
    """

    def __init__(self, capacity: int):
        """Constructor for a ReplayBuffer initialising an empty buffer (without memory
        
        Parameters
        ----------
        capacity: int
            total capacity of the replay buffer

        Attributes
        ----------
        memory: (Transition):
            Each component of the transition tuple is represented by a zero-initialised np.ndarray of
            floats with dimensionality (total buffer capacity, component dimensionality)
        
        writes: int
            number of experiences/ transitions already added to the buffer
        """
        self.capacity = int(capacity)
        self.memory = None
        self.writes = 0

    def init_memory(self, transition: Transition):
        """Initialises the memory with zero-entries
        
        Parameters
        ----------
        transition: Transition
            transition(s) to take the dimensionalities from
        """
        #for t in transition:
        #    assert t.ndim == 1  # sanity check

        self.memory = Transition(
            *[np.zeros([self.capacity, *t.shape], dtype=t.dtype) for t in transition]
        )

    def push(self, *args):
        """Adds transitions to the memory

        Note:
            overwrites first transitions stored once the capacity limit is reached

        :param *args: arguments to create transition from
        """
        if not self.memory:
            self.init_memory(Transition(*args))

        position = (self.writes) % self.capacity
        for i, data in enumerate(args):
            self.memory[i][position, :] = data

        self.writes = self.writes + 1

    def sample(self, batch_size: int, device: str = "cpu") -> Transition:
        """Samples batch of experiences from the replay buffer

        :param batch_size (int): size of the batch to be sampled and returned
        :param device (str): PyTorch device to cast to (for potential GPU support)
        :return (Transition): batch of experiences of given batch size
        """
        samples = np.random.randint(0, high=len(self), size=batch_size)

        batch = Transition(
            *[
                torch.from_numpy(np.take(d, samples, axis=0)).to(device)
                for d in self.memory
            ]
        )
        return batch

    def __len__(self):
        """Gives the length of the buffer
        """
        return min(self.writes, self.capacity)

