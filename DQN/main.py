import argparse
import os
import numpy as np
import gym
import gym_ple
from gym.wrappers import FrameStack
import datetime
from agent import Agent
from observer import CartPoleObserver, ImageObserver
from logger import Logger
from trainer import Trainer

from DDQN.agent import DDQNAgent
from DQN_with_multi_step.trainer import WithMultiStepTrainer
from DQN_with_noisy_net.agent import WithNoisyNetAgent
from DQN_with_prioritized_experience_replay.trainer import WithPrioritizedExperienceTrainer

def main():
    parser = argparse.ArgumentParser(description='Deep Q-Network')
    parser.add_argument('--algorithm', '-a', choices=('DQN', 'DDQN', 'DQN_with_multi_step', 'DQN_with_noisy_net', 'DQN_with_prioritized_experience_replay'), default='DQN', help='Algorithm to be used')
    parser.add_argument('--env', '-e', choices=('CartPole', 'Catcher'), default='CartPole', help='Environment Name')
    parser.add_argument('--play', action='store_true', help='Play Mode')
    parser.add_argument('--model_path', '-p', help='Trained Model Path (Valid only in play mode)')
    args = parser.parse_args()

    # 環境の初期化
    if args.env == "CartPole":
        # 環境の設定
        env = gym.make("Catcher-v0")
        obs = CartPoleObserver(gym.make("CartPole-v0"))

        # パラメータの設定
        model_type = "Simple"
        epsilon = 0.1
        epsilon_decay = 1.0
        epsilon_min = 0.1
        episode_count = 500

    elif args.env == "Catcher":
        # 環境の設定
        env = gym.make("Catcher-v0")
        obs = ImageObserver(env, 4, (84, 84), 4)

        # パラメータの設定
        model_type = "Default"
        epsilon = 0.1
        epsilon_decay = 1.0
        epsilon_min = 0.1
        episode_count = 200
    
    # ログ出力先の設定
    now = datetime.datetime.now()
    dst_path = os.path.join("log/", now.strftime('%Y%m%d_%H%M%S'))
    logger = Logger(args.algorithm, os.path.join(dst_path, "log.txt"))
    
    # エージェント、トレーナーの初期化
    if args.algorithm == "DQN":
        agent = Agent(model_type, epsilon, epsilon_decay, epsilon_min)
        trainer = Trainer(agent, obs, logger)
    elif args.algorithm == "DDQN":
        agent = DDQNAgent(model_type, epsilon, epsilon_decay, epsilon_min)
        trainer = Trainer(agent, obs, logger)
    elif args.algorithm == "DQN_with_multi_step":
        multi_step_num = 3
        agent = Agent(model_type, epsilon, epsilon_decay, epsilon_min)
        trainer = WithMultiStepTrainer(agent, obs, logger, multi_step_num)
    elif args.algorithm == "DQN_with_noisy_net":
        agent = WithNoisyNetAgent(model_type, epsilon, epsilon_decay, epsilon_min)
        trainer = Trainer(agent, obs, logger)
    elif args.algorithm == "DQN_with_prioritized_experience_replay":
        prioritized_experience_replay_valid_ep_num = 30
        agent = Agent(model_type, epsilon, epsilon_decay, epsilon_min)
        trainer = WithPrioritizedExperienceTrainer(agent, obs, logger, prioritized_experience_replay_valid_ep_num)

    if args.play:
        # 学習済みモデルの読み込み
        trainer.load_model(args.model_path)
        # 1エピソード実行
        trainer.play()
    else:
        # 学習実行
        trainer.train(episode_count=episode_count)
        # ログ出力
        logger.plot(key="reward", freq=50, save_path=os.path.join(dst_path, "reward.png"))
        agent.save(os.path.join(dst_path, "dqn_model.pt"))

        # 学習したエージェントで1エピソード実行
        trainer.play()

if __name__ == '__main__':
    main()