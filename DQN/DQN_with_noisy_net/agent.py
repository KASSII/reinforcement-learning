import numpy as np
import math
from collections import defaultdict
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('../')
from agent import Agent

class FactorizedNoisy(nn.Module):
    def __init__(self, in_features, out_features):
        super(FactorizedNoisy, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 学習パラメータを生成
        self.u_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_w  = nn.Parameter(torch.Tensor(out_features, in_features))
        self.u_b = nn.Parameter(torch.Tensor(out_features))
        self.sigma_b = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # 初期値設定
        stdv = 1. / math.sqrt(self.u_w.size(1))
        self.u_w.data.uniform_(-stdv, stdv)
        self.u_b.data.uniform_(-stdv, stdv)

        initial_sigma = 0.5 * stdv
        self.sigma_w.data.fill_(initial_sigma)
        self.sigma_b.data.fill_(initial_sigma)
    
    def reset_noise(self):
        rand_in = self._f(torch.randn(1, self.in_features, device=self.u_w.device))
        rand_out = self._f(torch.randn(self.out_features, 1, device=self.u_w.device))
        self.epsilon_w = torch.matmul(rand_out, rand_in)
        self.epsilon_b = rand_out.squeeze()

    def forward(self, x):
        w = self.u_w + self.sigma_w * self.epsilon_w
        b = self.u_b + self.sigma_b * self.epsilon_b
        return F.linear(x, w, b)

    def _f(self, x):
       return torch.sign(x) * torch.sqrt(torch.abs(x))

class WithNoisyNetDQNModel(nn.Module):
    def __init__(self, ch, n_action):
        super(WithNoisyNetDQNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=ch, out_channels=32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = FactorizedNoisy(3136, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = FactorizedNoisy(512, n_action)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x
    
    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()

class SimpleWithNoisyNetDQNModel(nn.Module):
    def __init__(self, n_state, n_action):
        super(SimpleWithNoisyNetDQNModel, self).__init__()
        self.fc1 = nn.Linear(n_state, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = FactorizedNoisy(32, n_action)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
    def reset_noise(self):
        self.fc3.reset_noise()

with_noisy_model_dict = {
    "Simple": SimpleWithNoisyNetDQNModel,
    "Default": WithNoisyNetDQNModel
}

class WithNoisyNetAgent(Agent):
    def __init__(self, model_type="Default", epsilon=0.1, epsilon_decay=1.0, epsilon_min=0.1):
        super().__init__(model_type, epsilon, epsilon_decay, epsilon_min)
    
    # エージェントを初期化する
    def initialize(self, actions, state_shape):
        self.actions = actions
        self.main_net = with_noisy_model_dict[self.model_type](state_shape[0], len(actions))
        self.main_net.to(self.device)
        self.train()

        self.target_net = copy.deepcopy(self.main_net)
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.target_net.eval()

        torch.backends.cudnn.benchmark = True
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=0.0001)

    # 状態sに対応する行動を返す
    def policy(self, state):
        # 推論モードの時は探索を行わないようにする
        if self.mode == "train":
            # εを減衰させる
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
            epsilon = self.epsilon
        else:
            epsilon = 0.0

        # εの確率で探索、(1-ε)の確率で活用を行う
        # ネットワークの出力が偏るので、未学習の状態の時はランダムに行動する
        if np.random.random() < epsilon or not self.updated:
            action = torch.LongTensor([[np.random.randint(len(self.actions))]]).long()
        else:
            self.main_net.eval()  # ネットワークを推論モードに切り替える
            with torch.no_grad():
                self.main_net.reset_noise()
                action = self.main_net(state.to(self.device)).max(1)[1].view(1, 1)
                action = action.to("cpu")
        return action
    
    def calc_td_error(self, batch, gamma, reduction='mean'):
        # バッチを分解する
        batch_size = len(batch)
        state_batch = torch.cat([b.s for b in batch]).to(self.device)
        action_batch = torch.cat([b.a for b in batch]).to(self.device)
        next_state_batch = torch.cat([b.n_s for b in batch]).to(self.device)
        reward_batch = torch.from_numpy(np.array([b.r for b in batch])).float().to(self.device)

        # 現在の状態s、選択された行動aに対する行動価値Q(s, a)を求める
        self.main_net.eval()
        self.main_net.reset_noise()
        Q = self.main_net(state_batch).gather(1, action_batch)

        # 次の状態s'における最大行動価値max_a'{Q(s', a')}を求める
        self.target_net.reset_noise()
        next_Q = torch.zeros(batch_size).to(self.device)
        final_state_mask = torch.from_numpy(np.array([not b.d for b in batch]).astype(np.bool))    # 次の状態が存在するインデックスのみTrueとなるようなマスクを生成
        if final_state_mask.any():      # 全ての要素がFalseだエラーになるので条件分岐
            next_Q[final_state_mask] = self.target_net(next_state_batch[final_state_mask]).max(1)[0].detach()

        # r+γQ(s', a')を計算
        expected = reward_batch + gamma * next_Q

        # TD誤差（損失関数）を計算
        self.main_net.train()
        # reductionモードがnoneの時は、差分をそのまま返す
        if reduction == 'none':
            loss = expected.unsqueeze(1) - Q
        # none以外の時は、Huber損失まで計算する
        else:
            loss = F.smooth_l1_loss(Q, expected.unsqueeze(1), reduction=reduction)
        return loss