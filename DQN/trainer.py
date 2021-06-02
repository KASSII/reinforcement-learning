import random
import torch
import torch.nn as nn
from collections import namedtuple
from collections import deque

Experience = namedtuple("Experience",
                        ["s", "a", "r", "n_s", "d"])

class Trainer():
    def __init__(self, agent, env, logger=None):
        self.agent = agent
        self.env = env
        self.logger = logger

        # エージェントの初期化
        actions = list(range(self.env.action_space.n))
        state_shape = self.env.observation_space.shape
        self.agent.initialize(actions, state_shape)
    
    # 学習
    def train(self, episode_count=500, buffer_size=50000, batch_size=32, learning_start_buffer_size=4096, gamma=0.9, render=False, disp_freq=10):
        # ロガーの初期化
        config_data = {
            "episode_count": episode_count,
            "buffer_size":buffer_size,
            "batch_size":batch_size,
            "learning_start_buffer_size":learning_start_buffer_size,
            "gamma": gamma,
        }
        self.logger.initialize(str(self.env.spec), config_data)

        self.experiences = deque(maxlen=buffer_size)

        # 指定した回数分エピソードを実行
        for e in range(episode_count):
            s = self.env.reset()
            s = torch.from_numpy(s).float().unsqueeze(0)
            episode_reward = 0.0
            done = False
            # エピソード実行
            while not done:
                if render:
                    self.env.render()
                
                # エージェントの方策にしたがって環境を更新
                a = self.agent.policy(s)
                n_state, reward, done, info = self.env.step(a[0][0].item())
                n_state = torch.from_numpy(n_state).float().unsqueeze(0)
                experience = Experience(s, a, reward, n_state, done)
                self.experiences.append(experience)

                if len(self.experiences) > learning_start_buffer_size:
                    batch = random.sample(self.experiences, batch_size)
                    self.agent.update(batch, gamma)

                # 状態を更新
                s = n_state
                episode_reward += reward
            
            self.agent.update_target_model()

            self.logger.add({"episode":e, "reward":episode_reward, "epsilon":self.agent.epsilon})
            # 一定間隔ごとにlog表示
            if e % disp_freq == 0:
                print("At episode {}, reward={}, epsilon={}".format(e, self.logger.get("reward", disp_freq), self.agent.epsilon))

    # 学習結果を使って1エピソード実行
    def play(self):
        self.agent.eval()
        s = self.env.reset()
        s = torch.from_numpy(s).float().unsqueeze(0)
        episode_reward = 0.0
        done = False
        while not done:
            self.env.render()
            a = self.agent.policy(s)
            n_state, reward, done, info = self.env.step(a[0][0].item())
            n_state = torch.from_numpy(n_state).float().unsqueeze(0)
            s = n_state
            episode_reward += reward
        self.env.render()
        print(episode_reward)
        self.env.close()
    
    # 学習モデルを読み込む
    def load_model(self, model_path):
        self.agent.load(model_path)