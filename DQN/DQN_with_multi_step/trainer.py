import random
import math
import torch
import torch.nn as nn
from collections import namedtuple
from collections import deque
import sys
sys.path.append('../')
from trainer import Trainer, Experience

class WithMultiStepTrainer(Trainer):
    def __init__(self, agent, env, logger=None, multi_step_num=3):
        super().__init__(agent, env, logger)
        self.multi_step_num = multi_step_num
    
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

        # 過去の状態を保存しておくためのキュー
        # 現在のステップを含めてmulti_step_num回分の情報を使って更新をしていくため、multi_step_num-1回分のデータを保持できるようにしておく
        past_states = deque(maxlen=self.multi_step_num-1)
        past_actions = deque(maxlen=self.multi_step_num-1)
        past_rewards = deque(maxlen=self.multi_step_num-1)

        # 指定した回数分エピソードを実行
        for e in range(episode_count):
            past_states.clear()
            past_actions.clear()
            past_rewards.clear()

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

                if len(past_states) == self.multi_step_num-1:
                    # multi_step_num分の報酬(r_t1 + gamma*r_t2 + ... + gamma^n*r_tn-1)を計算
                    sum_reward = 0.0
                    for i in range(self.multi_step_num-1):
                        sum_reward += past_rewards[i] * math.pow(gamma, i)
                    sum_reward += reward * math.pow(gamma, self.multi_step_num-1)

                    # multi_step_num前の状態のexperienceを記憶
                    experience = Experience(past_states[0], past_actions[0], sum_reward, n_state, done)
                    self.experiences.append(experience)

                if len(self.experiences) > learning_start_buffer_size:
                    batch = random.sample(self.experiences, batch_size)
                    self.agent.update(batch, math.pow(gamma, self.multi_step_num))        # Q(s', a')にかかるγはmulti_step_num乗される

                # 過去の情報を更新
                past_states.append(s)
                past_actions.append(a)
                past_rewards.append(reward)

                # 状態を更新
                s = n_state
                episode_reward += reward
            
            self.agent.update_target_model()

            self.logger.add({"episode":e, "reward":episode_reward, "epsilon":self.agent.epsilon})
            # 一定間隔ごとにlog表示
            if e % disp_freq == 0:
                print("At episode {}, reward={}, epsilon={}".format(e, self.logger.get("reward", disp_freq), self.agent.epsilon))
