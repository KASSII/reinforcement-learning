import random
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
    def train(self, agent, env, logger, episode_count=200, buffer_size=50000, batch_size=32, learning_start_buffer_size=4096, gamma=0.9, render=False, disp_freq=10):
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

        # 指定した回数分エピソードを実行
        for e in range(episode_count):
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
                experience = Experience(s, a, reward, n_state, done)
                self.experiences.append(experience)

                if len(self.experiences) > learning_start_buffer_size:
                    batch = random.sample(self.experiences, batch_size)
                    agent.update(batch, gamma)

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