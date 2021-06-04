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

class DDQNAgent(Agent):
    def __init__(self, model_type="Default", epsilon=0.1, epsilon_decay=1.0, epsilon_min=0.1):
        super().__init__(model_type, epsilon, epsilon_decay, epsilon_min)
    
    def calc_td_error(self, batch, gamma, reduction='mean'):
        # バッチを分解する
        batch_size = len(batch)
        state_batch = torch.cat([b.s for b in batch]).to(self.device)
        action_batch = torch.cat([b.a for b in batch]).to(self.device)
        next_state_batch = torch.cat([b.n_s for b in batch]).to(self.device)
        reward_batch = torch.from_numpy(np.array([b.r for b in batch])).float().to(self.device)

        # 現在の状態s、選択された行動aに対する行動価値Q(s, a)を求める
        self.main_net.eval()
        Q = self.main_net(state_batch).gather(1, action_batch)

        # 次の状態s'における行動価値Q(s', a')を求める
        # 行動a'はmain_networkから求める
        next_a = torch.argmax(self.main_net(next_state_batch), axis=1)
        # 行動価値Q(s', a')はtaret_networkから求める
        next_Q = torch.zeros(batch_size).to(self.device)
        final_state_mask = torch.from_numpy(np.array([not b.d for b in batch]).astype(np.bool))    # 次の状態が存在するインデックスのみ1となるようなマスクを生成
        """
        [メモ]
        self.target_net(next_state_batch): target_networkによる各行動に対する行動価値 [32, 2]
        self.target_net(next_state_batch)[np.arange(0, batch_size), next_a]: target_networkによる行動a'に対する行動価値 [32]
        self.target_net(next_state_batch)[np.arange(0, batch_size), next_a][final_state_mask]: 次の状態が存在する状態s'、行動a'に対する行動価値
        """
        next_Q[final_state_mask] = self.target_net(next_state_batch)[np.arange(0, batch_size), next_a][final_state_mask].detach()

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