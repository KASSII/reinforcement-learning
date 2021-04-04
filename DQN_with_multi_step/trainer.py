import random
import math
import torch
import torch.nn as nn
from collections import namedtuple
from collections import deque

Experience = namedtuple("Experience",
                        ["s", "a", "r", "n_s", "d"])

class Trainer():
    def __init__(self):
        pass
    
    # 学習
    def train(self, agent, env, logger, episode_count=500, buffer_size=50000, batch_size=32, learning_start_buffer_size=4096, gamma=0.9, multi_step_num=3, render=False, disp_freq=10):
        # エージェントの初期化
        actions = list(range(env.action_space.n))
        state_shape = env.observation_space.shape
        agent.initialize(actions, state_shape)

        # ロガーの初期化
        config_data = {
            "episode_count": episode_count,
            "gamma": gamma,
        }
        logger.initialize(str(env.spec), config_data)

        self.experiences = deque(maxlen=buffer_size)

        # 過去の状態を保存しておくためのキュー
        # 現在のステップを含めてmulti_step_num回分の情報を使って更新をしていくため、multi_step_num-1回分のデータを保持できるようにしておく
        past_states = deque(maxlen=multi_step_num-1)
        past_actions = deque(maxlen=multi_step_num-1)
        past_rewards = deque(maxlen=multi_step_num-1)

        # 指定した回数分エピソードを実行
        for e in range(episode_count):
            past_states.clear()
            past_actions.clear()
            past_rewards.clear()

            s = env.reset()
            s = torch.from_numpy(s).float().unsqueeze(0)
            episode_reward = 0.0
            done = False
            # エピソード実行
            while not done:
                if render:
                    env.render()
                
                # エージェントの方策にしたがって環境を更新
                a = agent.policy(s)
                n_state, reward, done, info = env.step(a[0][0].item())
                n_state = torch.from_numpy(n_state).float().unsqueeze(0)

                if len(past_states) == multi_step_num-1:
                    # multi_step_num分の報酬(r_t1 + gamma*r_t2 + ... + gamma^n*r_tn-1)を計算
                    sum_reward = 0.0
                    for i in range(multi_step_num-1):
                        sum_reward += past_rewards[i] * math.pow(gamma, i)
                    sum_reward += reward * math.pow(gamma, multi_step_num-1)

                    # multi_step_num前の状態のexperienceを記憶
                    experience = Experience(past_states[0], past_actions[0], sum_reward, n_state, done)
                    self.experiences.append(experience)

                if len(self.experiences) > learning_start_buffer_size:
                    batch = random.sample(self.experiences, batch_size)
                    agent.update(batch, math.pow(gamma, multi_step_num))        # Q(s', a')にかかるγはmulti_step_num乗される

                # 過去の情報を更新
                past_states.append(s)
                past_actions.append(a)
                past_rewards.append(reward)

                # 状態を更新
                s = n_state
                episode_reward += reward
            
            agent.update_target_model()

            logger.add({"episode":e, "reward":episode_reward})
            # 一定間隔ごとにlog表示
            if e % disp_freq == 0:
                print("At episode {}, reward={}".format(e, logger.get("reward", disp_freq)))

    
    # 学習結果を使って1エピソード実行
    def play(self, agent, env):
        agent.eval()
        s = env.reset()
        s = torch.from_numpy(s).float().unsqueeze(0)
        episode_reward = 0.0
        done = False
        while not done:
            env.render()
            a = agent.policy(s)
            n_state, reward, done, info = env.step(a[0][0].item())
            n_state = torch.from_numpy(n_state).float().unsqueeze(0)
            s = n_state
            episode_reward += reward
        env.render()
        print(episode_reward)