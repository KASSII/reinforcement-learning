import os
import gym
import datetime
from frozen_lake_util import show_q_value
from agent import Agent
from logger import Logger
from trainer import Trainer

def main():
    # ログ出力先の設定
    now = datetime.datetime.now()
    dst_path = os.path.join("log/", now.strftime('%Y%m%d_%H%M%S'))

    # 環境、エージェント、ロガー、トレーナーの初期化
    obs = gym.make("FrozenLakeEasy-v0")
    agent = Agent()
    logger = Logger(os.path.join(dst_path, "log.txt"))
    trainer = Trainer()

    # 学習実行
    trainer.train(agent, obs, logger)

    # ログ出力
    logger.plot(key="reward", freq=50, save_path=os.path.join(dst_path, "reward.png"))
    agent.save(os.path.join(dst_path, "result.pkl"))

    # 学習したエージェントで1エピソード実行
    trainer.play(agent, obs)


if __name__ == '__main__':
    main()