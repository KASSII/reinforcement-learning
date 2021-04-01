import gym

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
