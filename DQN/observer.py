class CartPoleObserver():
    def __init__(self, env):
        self.step_count = 0
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.spec = env.spec
    
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
    
