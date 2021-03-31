import numpy as np
from collections import defaultdict
import copy
import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimpleDQNModel(nn.Module):
    def __init__(self, n_state, n_action):
        super(SimpleDQNModel, self).__init__()
        self.fc1 = nn.Linear(n_state, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, n_action)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class Agent():
    def __init__(self, epsilon=0.1):
        self.Q = None
        self.actions = None
        self.epsilon = epsilon
        self.mode = "train"
        self.main_net = None
        self.target_net = None
        self.updated = False
    
    # エージェントを初期化する
    def initialize(self, actions, state_shape):
        self.actions = actions
        self.main_net = SimpleDQNModel(state_shape[0], len(actions))
        self.target_net = SimpleDQNModel(state_shape[0], len(actions))
        self.target_net.eval()
        self.train()
        torch.backends.cudnn.benchmark = True
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=0.0001)

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
            self.main_net.eval()  # ネットワークを推論モードに切り替える
            with torch.no_grad():
                action = self.main_net(state).max(1)[1].view(1, 1)
        return action
    
    def update(self, batch, gamma):
        # バッチを分解する
        batch_size = len(batch)
        state_batch = torch.cat([b.s for b in batch])
        action_batch = torch.cat([b.a for b in batch])
        next_state_batch = torch.cat([b.n_s for b in batch])
        reward_batch = torch.from_numpy(np.array([b.r for b in batch])).float()

        # 現在の状態s、選択された行動aに対する行動価値Q(s, a)を求める
        self.main_net.eval()
        Q = self.main_net(state_batch).gather(1, action_batch)

        # 次の状態s'における最大行動価値max_a'{Q(s', a')}を求める
        next_Q = torch.zeros(batch_size)
        final_state_mask = torch.from_numpy(np.array([not b.d for b in batch]).astype(np.uint8))    # 次の状態が存在するインデックスのみ1となるようなマスクを生成
        next_Q[final_state_mask] = self.target_net(next_state_batch[final_state_mask]).max(1)[0].detach()

        # r+γQ(s', a')を計算
        expected = reward_batch + gamma * next_Q

        # TD誤差（損失関数）を計算
        self.main_net.train()
        loss = F.smooth_l1_loss(Q, expected.unsqueeze(1))

        # パラメータ更新
        self.optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_value_(self.main_net.parameters(), clip_value=1.0)
        self.optimizer.step()

        # 学習済みフラグを設定
        self.updated = True

    def update_target_model(self):
        #for name, param in self.main_net.named_parameters():
        #    self.target_net.state_dict()[name] = self.main_net.state_dict()[name].clone()
        self.target_net = copy.deepcopy(self.main_net)

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