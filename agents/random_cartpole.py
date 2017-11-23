class RandomAgent():
    # Random agent

    def __init__(self, exp):
        self.steps = 0
        self.exp = exp

    def act(self, state):
        # Select a random action
        action = self.exp.env.action_space.sample()
        return action

    def observe(self, sample):  # (state_0, action, reward, state_1)
        # Do nothing
        pass

    def replay(self):
        # Do nothing
        return None