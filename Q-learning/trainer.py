from collections import defaultdict

class Trainer():
    def __init__(self):
        pass
    
    def train(self, agent, env, logger, episode_count=10000, gamma=0.9, learning_rate=0.1, render=False):
        # エージェントの初期化
        actions = list(range(env.action_space.n))
        agent.actions = actions
        agent.Q = defaultdict(lambda: [0] * len(actions))
        agent.train()

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
                a = agent.policy(s)
                n_state, reward, done, info = env.step(a)

                # 現在の見積もり値（estimated）と実測した値（gain）からTD誤差を計算
                gain = reward + gamma * max(agent.Q[n_state])
                estimated = agent.Q[s][a]
                agent.Q[s][a] += learning_rate * (gain - estimated)     # Q値を更新

                # 状態を更新
                s = n_state
                episode_reward += reward
        
            logger.add({"reward":episode_reward})
    
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