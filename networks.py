import torch
import torch.nn as nn



class FFN(nn.Module):

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


class ConvNet(nn.Module):
    """
    Network from Nature paper of DeepMind: https://www.nature.com/articles/nature14236.pdf
    """
    def __init__(self, nChannels, nOut):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(nChannels, 32, kernel_size=8, stride=4), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(32,64,kernel_size=4,stride=2), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3, stride=1),nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(7*7*64, 512), nn.ReLU())
        self.layer5 = nn.Linear(512, nOut)

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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0],-1)
        x = self.layer4(x)
        y = self.layer5(x)
        return y






