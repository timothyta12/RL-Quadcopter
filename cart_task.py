import gym

class Task:
    def __init__(self):
        self.env = gym.make('Pendulum-v0')
        self.env.reset()
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.action_size = self.env.action_space.shape[0]
        
    def get_reward(self):
        return self.reward
    
    def step(self, action):
        observation, self.reward, done, info = self.env.step(action)
        
        return observation, self.reward, done
        
    def reset(self):
        return self.env.reset()