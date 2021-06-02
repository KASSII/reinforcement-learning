import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from collections import deque
import sys
sys.path.append('../')
from trainer import Trainer, Experience

class WithPrioritizedExperienceTrainer(Trainer):
    def __init__(self, agent, env, logger=None, prioritized_experience_replay_valid_ep_num=30):
        super().__init__(agent, env, logger)
        self.prioritized_experience_replay_valid_ep_num = prioritized_experience_replay_valid_ep_num
    
    # TDエラーに合わせた確率でバッチを作成する
    def make_batch(self, batch_size):
        td_errors = np.asarray(self.experience_td_errors) + 1e-9
        probs = np.exp(td_errors)/np.sum(np.exp(td_errors))
        index = np.random.choice(np.arange(len(td_errors)), batch_size, p=probs, replace=False)
        batch = []
        for i in index:
            batch.append(self.experiences[i])
        return batch
    
    # キューに保存されている全経験データのTD誤差を更新
    def update_td_error(self, agent, batch_size, gamma):      
        experiences = list(self.experiences)
        experience_td_errors = np.zeros(len(experiences))
        for i in range(0, len(experiences), batch_size):
            # バッチを作成
            batch = experiences[i:i+batch_size]
            # バッチに対するTD誤差を計算
            losses = agent.calc_td_error(batch, gamma, 'none')
            losses = losses.to('cpu').detach().numpy()
            losses = losses.reshape((losses.shape[0]))
            # 計算結果を格納
            experience_td_errors[i:i+batch_size] = losses
        # 新しいTD誤差で更新
        self.experience_td_errors.clear()
        self.experience_td_errors.extend(experience_td_errors)
    
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
        self.experience_td_errors = deque(maxlen=buffer_size)
        update_episode_num = 0      # updateを行なったエピソード数

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

                # 得られた経験のTD誤差を計算
                losses = self.agent.calc_td_error([experience], gamma, 'none')
                losses = losses.to('cpu').detach().numpy().copy()
                self.experience_td_errors.append(losses[0][0])

                if len(self.experiences) > learning_start_buffer_size:
                    # 指定回数以上、更新を行なったらprioritized_experience_replayを有効にする
                    # （ある程度学習を進めないと、選択が不正に偏ることがあるので初期はランダムサンプリングする）
                    if update_episode_num > self.prioritized_experience_replay_valid_ep_num:
                        batch = self.make_batch(batch_size)
                    else:
                        batch = random.sample(self.experiences, batch_size)
                    self.agent.update(batch, gamma)

                # 状態を更新
                s = n_state
                episode_reward += reward
            
            # 更新エピソード数を更新
            if self.agent.updated:
                update_episode_num += 1

            # ExperienceのTD誤差を更新
            if update_episode_num > self.prioritized_experience_replay_valid_ep_num:
                self.update_td_error(self.agent, 4096, gamma)

            # エージェントのtarget_modelを更新
            self.agent.update_target_model()

            self.logger.add({"episode":e, "reward":episode_reward, "epsilon":self.agent.epsilon})
            # 一定間隔ごとにlog表示
            if e % disp_freq == 0:
                print("At episode {}, reward={}, epsilon={}".format(e, self.logger.get("reward", disp_freq), self.agent.epsilon))