import gym
from frozen_lake_util import show_q_value
from agent import Agent
from logger import Logger
from trainer import Trainer

def main():
    # 環境、エージェント、ロガー、トレーナーの初期化
    obs = gym.make("FrozenLakeEasy-v0")
    agent = Agent()
    logger = Logger()
    trainer = Trainer()

    # 学習実行
    trainer.train(agent, obs, logger)

    # ログ出力
    logger.show_graph(key="reward", freq=50)

    # 学習したエージェントで1エピソード実行
    trainer.play(agent, obs)


if __name__ == '__main__':
    main()