import numpy as np
from collections import defaultdict
import dill

class Agent():
    def __init__(self, epsilon=0.1):
        self.Q = None
        self.actions = None
        self.epsilon = epsilon
        self.mode = "train"
    
    # エージェントを初期化する
    def initialize(self, actions):
        self.actions = actions
        self.Q = defaultdict(lambda: [0] * len(actions))
        self.train()

    # 状態sに対応する行動を返す
    def policy(self, state):
        # 推論モードの時は探索を行わないようにする
        if self.mode == "train":
            epsilon = self.epsilon
        else:
            epsilon = 0.0

        # εの確率で探索、(1-ε)の確率で活用を行う
        if np.random.random() < epsilon:
            return np.random.randint(len(self.actions))
        else:
            if state in self.Q and sum(self.Q[state]) != 0:
                return np.argmax(self.Q[state])
            # Q値が確定していない時にQ値に従うと行動が偏って学習できなくなるので、ランダムに行動選択する
            else:
                return np.random.randint(len(self.actions))

    # 学習モードにする
    def train(self):
        self.mode = "train"

    # 推論モードにする
    def eval(self):
        self.mode = "eval"
    
    # 学習結果を保存
    def save(self, path):
        with open(path, 'wb') as f:
            dill.dump(self.Q , f)
    
    # 学習結果を読み込み
    def load(self, path):
        with open(path, 'rb') as f:
            self.Q = dill.load(f)