import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from collections import deque

Experience = namedtuple("Experience",
                        ["s", "a", "r", "n_s", "d"])

class Trainer():
    def __init__(self):
        pass
    
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
    def train(self, agent, env, logger, episode_count=200, buffer_size=50000, batch_size=32, learning_start_buffer_size=4096, gamma=0.9, prioritized_experience_replay_valid_ep_num=30, render=False, disp_freq=10):
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
        self.experience_td_errors = deque(maxlen=buffer_size)
        update_episode_num = 0      # updateを行なったエピソード数

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

                # 得られた経験のTD誤差を計算
                losses = agent.calc_td_error([experience], gamma, 'none')
                losses = losses.to('cpu').detach().numpy().copy()
                self.experience_td_errors.append(losses[0][0])

                if len(self.experiences) > learning_start_buffer_size:
                    # 指定回数以上、更新を行なったらprioritized_experience_replayを有効にする
                    # （ある程度学習を進めないと、選択が不正に偏ることがあるので初期はランダムサンプリングする）
                    if update_episode_num > prioritized_experience_replay_valid_ep_num:
                        batch = self.make_batch(batch_size)
                    else:
                        batch = random.sample(self.experiences, batch_size)
                    agent.update(batch, gamma)

                # 状態を更新
                s = n_state
                episode_reward += reward
            
            # 更新エピソード数を更新
            if agent.updated:
                update_episode_num += 1

            # ExperienceのTD誤差を更新
            if update_episode_num > prioritized_experience_replay_valid_ep_num:
                self.update_td_error(agent, 4096, gamma)

            # エージェントのtarget_modelを更新
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