import numpy as np
from collections import defaultdict
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('../')
from agent import Agent

class DuelingNetModel(nn.Module):
    def __init__(self, ch, n_action):
        super(DuelingNetModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=ch, out_channels=32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1_adv = nn.Linear(3136, 512)
        self.relu4_adv = nn.ReLU()
        self.fc2_adv = nn.Linear(512, n_action)

        self.fc1_val = nn.Linear(3136, 512)
        self.relu4_val = nn.ReLU()
        self.fc2_val = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        adv = self.fc1_adv(x)
        adv = self.relu4_adv(adv)
        adv = self.fc2_adv(adv)

        val = self.fc1_val(x)
        val = self.relu4_val(val)
        val = self.fc2_val(val)
        val = val.expand(-1, adv.size(1))

        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        return output

class SimpleDuelingNetModel(nn.Module):
    def __init__(self, n_state, n_action):
        super(SimpleDuelingNetModel, self).__init__()
        self.fc1 = nn.Linear(n_state, 32)
        self.relu1 = nn.ReLU()

        self.fc2_adv = nn.Linear(32, 32)
        self.relu2_adv = nn.ReLU()
        self.fc3_adv = nn.Linear(32, n_action)

        self.fc2_val = nn.Linear(32, 32)
        self.relu2_val = nn.ReLU()
        self.fc3_val = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        adv = self.fc2_adv(x)
        adv = self.relu2_adv(adv)
        adv = self.fc3_adv(adv)
        val = self.fc2_val(x)
        val = self.relu2_val(val)
        val = self.fc3_val(val)
        val = val.expand(-1, adv.size(1))
        
        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        return output

dueling_network_model_dict = {
    "Simple": SimpleDuelingNetModel,
    "Default": DuelingNetModel
}

class DuelingNetworkAgent(Agent):
    def __init__(self, model_type="Default", epsilon=0.1, epsilon_decay=1.0, epsilon_min=0.1):
        super().__init__(model_type, epsilon, epsilon_decay, epsilon_min)
    
    # エージェントを初期化する
    def initialize(self, actions, state_shape):
        self.actions = actions
        self.main_net = dueling_network_model_dict[self.model_type](state_shape[0], len(actions))
        self.main_net.to(self.device)
        self.train()

        self.target_net = copy.deepcopy(self.main_net)
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.target_net.eval()

        torch.backends.cudnn.benchmark = True
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=0.0001)