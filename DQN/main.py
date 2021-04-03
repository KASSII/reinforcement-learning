import argparse
import os
import numpy as np
import gym
import gym_ple
from gym.wrappers import FrameStack
import datetime
from agent import Agent, SimpleAgent
from observer import CartPoleObserver, ImageObserver
from logger import Logger
from trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description='Deep Q-Network')
    parser.add_argument('--env', '-e', choices=('CartPole', 'Catcher'), default='CartPole', help='Environment Name')
    parser.add_argument('--play', action='store_true', help='Play Mode')
    parser.add_argument('--model_path', '-p', help='Trained Model Path (Valid only in play mode)')
    args = parser.parse_args()

    # 環境、エージェント、トレーナーの初期化
    if args.env == "CartPole":
        env = gym.make("Catcher-v0")
        obs = CartPoleObserver(gym.make("CartPole-v0"))
    elif args.env == "Catcher":
        env = gym.make("Catcher-v0")
        obs = ImageObserver(env, 4, (84, 84), 4)
    
    agent = Agent()
    trainer = Trainer()

    if args.play:
        # エージェントの初期化
        actions = list(range(obs.action_space.n))
        state_shape = obs.observation_space.shape
        agent.initialize(actions, state_shape)
        # 学習済みモデルの読み込み
        agent.load(args.model_path)
        # 1エピソード実行
        trainer.play(agent, obs)
    else:
        # ログ出力先の設定
        now = datetime.datetime.now()
        dst_path = os.path.join("log/", now.strftime('%Y%m%d_%H%M%S'))
        logger = Logger(os.path.join(dst_path, "log.txt"))

        # 学習実行
        trainer.train(agent, obs, logger)
        # ログ出力
        logger.plot(key="reward", freq=50, save_path=os.path.join(dst_path, "reward.png"))
        agent.save(os.path.join(dst_path, "dqn_model.pt"))

        # 学習したエージェントで1エピソード実行
        trainer.play(agent, obs)

if __name__ == '__main__':
    main()