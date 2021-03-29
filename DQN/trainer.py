import torch
import torch.nn as nn

class Trainer():
    def __init__(self):
        pass
    
    # 学習
    def train(self, agent, env, logger, episode_count=1200, gamma=0.9, learning_rate=0.1, render=False, disp_freq=100):
        # エージェントの初期化
        actions = list(range(env.action_space.n))
        state_shape = env.observation_space.shape
        agent.initialize(actions, state_shape)

        # ロガーの初期化
        config_data = {
            "episode_count": episode_count,
            "gamma": gamma,
            "learning_rate": learning_rate
        }
        logger.initialize(str(env.spec), config_data)

        # 指定した回数分エピソードを実行
        for e in range(episode_count):            
            s = env.reset()
            episode_reward = 0.0
            done = False
            # エピソード実行
            while not done:
                if render:
                    env.render()
                
                # エージェントの方策にしたがって環境を更新
                s = torch.from_numpy(s).float().unsqueeze(0)
                a = agent.policy(s)
                n_state, reward, done, info = env.step(a[0][0].item())
                import pdb;pdb.set_trace()










                # 現在の見積もり値（estimated）と実測した値（gain）からTD誤差を計算
                gain = reward + gamma * max(agent.Q[n_state])
                estimated = agent.Q[s][a]
                agent.Q[s][a] += learning_rate * (gain - estimated)     # Q値を更新

                # 状態を更新
                s = n_state
                episode_reward += reward
        
            logger.add({"episode":e, "reward":episode_reward})
            # 一定間隔ごとにlog表示
            if e % disp_freq == 0:
                print("At episode {}, reward={}".format(e, logger.get("reward", disp_freq)))

    
    # 学習結果を使って1エピソード実行
    def play(self, agent, env):
        agent.eval()
        s = env.reset()
        done = False
        while not done:
            env.render()
            a = agent.policy(s)
            n_state, reward, done, info = env.step(a)
            s = n_state
        env.render()