import math
from collections import defaultdict

class Trainer():
    def __init__(self):
        pass
    
    # 学習
    def train(self, agent, env, logger, episode_count=10000, gamma=0.9, render=False, disp_freq=100):
        # エージェントの初期化
        actions = list(range(env.action_space.n))
        agent.initialize(actions)

        # ロガーの初期化
        config_data = {
            "episode_count": episode_count,
            "gamma": gamma,
        }
        logger.initialize(str(env.spec), config_data)

        # 状態の出現回数を保持する辞書を初期化
        N = defaultdict(lambda: [0] * len(actions))

        # 指定した回数分エピソードを実行
        for e in range(episode_count):            
            s = env.reset()
            episode_reward = 0.0
            done = False
            experiences = []
            # エピソード実行
            while not done:
                if render:
                    env.render()
                
                # エージェントの方策にしたがって環境を更新
                a = agent.policy(s)
                n_state, reward, done, info = env.step(a)

                # 経験を保存
                experiences.append({"state":s, "action": a, "reward": reward})

                # 状態を更新
                s = n_state
                episode_reward += reward
        
            # 得られた経験からQ値を更新
            for i, experience in enumerate(experiences):
                s = experience["state"]
                a = experience["action"]
                G = 0.0
                t = 0
                for j in range(i, len(experiences)):
                    G += math.pow(gamma, t) * experiences[j]["reward"]
                    t += 1
                
                N[s][a] += 1
                agent.Q[s][a] += (1 / N[s][a]) * (G - agent.Q[s][a])

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