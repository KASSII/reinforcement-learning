import numpy as np
from collections import defaultdict
import dill
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDQNModel(nn.Module):
    def __init__(self, n_state, n_action):
        super(SimpleDQNModel, self).__init__()
        self.fc1 = nn.Linear(n_state, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, n_action)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

class Agent():
    def __init__(self, epsilon=0.1):
        self.Q = None
        self.actions = None
        self.epsilon = epsilon
        self.mode = "train"
        self.main_net = None
        self.updated = False
    
    # エージェントを初期化する
    def initialize(self, actions, state_shape):
        self.actions = actions
        self.main_net = SimpleDQNModel(state_shape[0], len(actions))
        self.train()

    # 状態sに対応する行動を返す
    def policy(self, state):
        # 推論モードの時は探索を行わないようにする
        if self.mode == "train":
            epsilon = self.epsilon
        else:
            epsilon = 0.0

        # εの確率で探索、(1-ε)の確率で活用を行う
        # ネットワークの出力が偏るので、未学習の状態の時はランダムに行動する
        if np.random.random() < epsilon or not self.updated:
            action = torch.LongTensor([[np.random.randint(len(self.actions))]]).long()
        else:
            network_mode = self.main_net.training
            self.main_net.eval()  # ネットワークを推論モードに切り替える
            with torch.no_grad():
                action = self.main_net(state).max(1)[1].view(1, 1)
            if network_mode:
                self.main_net.train()
        return action
    
    def update(self, batch):
        # バッチを分解する
        state_batch = torch.cat([b.s for b in batch])
        action_batch = torch.cat([b.a for b in batch])
        reward_batch = torch.from_numpy(np.array([b.r for b in batch])).float()

        # 現在の状態s、選択された行動aに対する行動価値Q(s, a)を求める
        self.main_net.train()
        Q_s_a = self.main_net(state_batch).gather(1, action_batch)
        import pdb;pdb.set_trace()
        self.updated = True

    # 学習モードにする
    def train(self):
        self.mode = "train"
        self.main_net.train()

    # 推論モードにする
    def eval(self):
        self.mode = "eval"
        self.main_net.eval()
    
    # 学習結果を保存
    def save(self, path):
        with open(path, 'wb') as f:
            dill.dump(self.Q , f)
    
    # 学習結果を読み込み
    def load(self, path):
        with open(path, 'rb') as f:
            self.Q = dill.load(f)