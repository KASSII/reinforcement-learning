import numpy as np
import gym
from gym.spaces import Box
from PIL import Image
from collections import deque
import torch
from torchvision import transforms as T

class CartPoleObserver(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.env = env
    
    def reset(self):
        self.step_count = 0
        return self.env.reset()
    
    def render(self):
        self.env.render()
    
    def step(self, action):
        self.step_count += 1
        n_state, reward, done, info = self.env.step(action)

        if done:
            if self.step_count < 195:
                reward = -1.0
            else:
                reward = 1.0
        else:
            reward = 0.0
        
        return n_state, reward, done, info

class ImageObserver(gym.ObservationWrapper):
    def __init__(self, env, skip, resize_shape, num_stack):
        super().__init__(env)
        self.skip = skip
        self.resize_shape = resize_shape   # (width, height)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        self.env = env
        self.observation_space = Box(low=0, high=1, shape=(num_stack, resize_shape[0], resize_shape[1]), dtype=np.float64)
    
    def _transform(self, n_state):
        # グレースケール変換
        grayed_n_state = Image.fromarray(n_state).convert("L")

        # リサイズ
        resized_n_state = grayed_n_state.resize(self.resize_shape)
        resized_n_state = np.array(resized_n_state).astype("float")

        # 正規化
        normalized_n_state = resized_n_state / 255.0

        return normalized_n_state

    def reset(self):
        self.frames.clear()
        state = self.env.reset()

        # 画像処理変換
        transformed_state = self._transform(state)

        # 指定フレーム分スタックする
        for i in range(self.num_stack):
            self.frames.append(transformed_state)
        state = np.array(self.frames)
        return state
    
    def step(self, action):
        # 指定フレームスキップする（連続フレームは動きが少ないので間引く）
        total_reward = 0.0
        done = False
        for i in range(self.skip):
            n_state, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        
        # 画像処理変換
        transformed_n_state = self._transform(n_state)

        # 指定フレーム分スタックする
        if len(self.frames) == 0:
            for i in range(self.num_stack):
                self.frames.append(transformed_n_state)
        else:
            self.frames.append(transformed_n_state)
        
        n_state = np.array(self.frames)
        return n_state, total_reward, done, info

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        import pdb;pdb.set_trace()
        return observation