import numpy as np
from collections import defaultdict
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

class Agent():
    def __init__(self, epsilon=0.1):
        self.actions = None
        self.epsilon = epsilon
        self.mode = "train"
        self.main_net = None
        self.target_net = None
        self.updated = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # エージェントを初期化する
    def initialize(self, actions, state_shape):
        self.actions = actions
        self.main_net = DuelingNetModel(state_shape[0], len(actions))
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
                action = self.main_net(state.to(self.device)).max(1)[1].view(1, 1)
                action = action.to("cpu")
        return action
    
    def update(self, batch, gamma):
        # バッチを分解する
        batch_size = len(batch)
        state_batch = torch.cat([b.s for b in batch]).to(self.device)
        action_batch = torch.cat([b.a for b in batch]).to(self.device)
        next_state_batch = torch.cat([b.n_s for b in batch]).to(self.device)
        reward_batch = torch.from_numpy(np.array([b.r for b in batch])).float().to(self.device)

        # 現在の状態s、選択された行動aに対する行動価値Q(s, a)を求める
        self.main_net.eval()
        Q = self.main_net(state_batch).gather(1, action_batch)

        # 次の状態s'における最大行動価値max_a'{Q(s', a')}を求める
        next_Q = torch.zeros(batch_size).to(self.device)
        final_state_mask = torch.from_numpy(np.array([not b.d for b in batch]).astype(np.bool))    # 次の状態が存在するインデックスのみ1となるようなマスクを生成
        next_Q[final_state_mask] = self.target_net(next_state_batch[final_state_mask]).max(1)[0].detach()

        # r+γQ(s', a')を計算
        expected = reward_batch + gamma * next_Q

        # TD誤差（損失関数）を計算
        self.main_net.train()
        loss = F.smooth_l1_loss(Q, expected.unsqueeze(1))

        # パラメータ更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 学習済みフラグを設定
        self.updated = True

    def update_target_model(self):
        self.target_net.load_state_dict(self.main_net.state_dict())

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
        torch.save(self.main_net.state_dict(), path)
    
    # 学習結果を読み込み
    def load(self, path):
        self.main_net.load_state_dict(torch.load(path))
        self.updated = True
    
class SimpleAgent(Agent):
    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)
    
    def initialize(self, actions, state_shape):
        self.actions = actions
        self.main_net = SimpleDuelingNetModel(state_shape[0], len(actions))
        self.main_net.to(self.device)
        self.train()

        self.target_net = copy.deepcopy(self.main_net)
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.target_net.eval()

        torch.backends.cudnn.benchmark = True
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=0.0001)