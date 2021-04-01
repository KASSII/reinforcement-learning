import os
import numpy as np
import gym
import gym_ple
from gym.wrappers import FrameStack
import datetime
from agent import Agent, SimpleAgent
from observer import CartPoleObserver, SkipFrame, GrayScaleObservation, ResizeObservation
from logger import Logger
from trainer import Trainer

def main():
    # ログ出力先の設定
    now = datetime.datetime.now()
    dst_path = os.path.join("log/", now.strftime('%Y%m%d_%H%M%S'))

    # 環境、エージェント、ロガー、トレーナーの初期化
    #obs = CartPoleObserver(gym.make("CartPole-v0"))
    #agent = SimpleAgent()

    obs = gym.make("Catcher-v0")
    obs = SkipFrame(obs, skip=4)
    obs = GrayScaleObservation(obs)
    #obs = ResizeObservation(obs, shape=84)
    #obs = FrameStack(obs, num_stack=4)
    agent = Agent()

    logger = Logger(os.path.join(dst_path, "log.txt"))
    trainer = Trainer()

    # 学習実行
    trainer.train(agent, obs, logger)

    # ログ出力
    logger.plot(key="reward", freq=50, save_path=os.path.join(dst_path, "reward.png"))
    agent.save(os.path.join(dst_path, "dqn_model.pt"))

    # 学習したエージェントで1エピソード実行
    trainer.play(agent, obs)


if __name__ == '__main__':
    main()