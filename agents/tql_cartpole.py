# Import libraries
import math
import numpy as np


class TQLAgent():
    # Table Q-learning agent

    def __init__(self, exp):
        # Bounds for each discrete state
        self.state_bounds = list(zip(exp.env.observation_space.low, exp.env.observation_space.high))
        # Limit the velocity bounds
        self.state_bounds[1] = [-0.5, 0.5]
        self.state_bounds[3] = [-math.radians(50), math.radians(50)]
        # Number of discrete actions
        self.n_actions = exp.env.action_space.n  # (left, right)
        # Number of discrete states (bucket) per state dimension
        self.n_bucket = (1, 1, 6, 3)  # (x, dot x, theta, dot theta)
        # Q-table
        self.Q_table = np.zeros(self.n_bucket+(self.n_actions,))
        self.steps = 0
        self.env = exp.env

        # Hyperparameters
        self.alpha_max = 0.2  # Max learning rate
        self.alpha = self.alpha_max  # Max learning rate
        self.alpha_min = 0.1  # Min learning rate
        self.epsilon_max = 1  # Max exploration rate
        self.epsilon_min = 0.01  # Min exploration rate
        self.epsilon = self.epsilon_max
        self.tao = 0.0002
        self.gamma = 0.95  # Discount rate

    def _state_to_bucket(self, state):

        bucket_indices = []
        for i in range(len(state)):
            if state[i] <= self.state_bounds[i][0]:
                bucket_index = 0
            elif state[i] >= self.state_bounds[i][1]:
                bucket_index = self.n_bucket[i] - 1
            else:
                # Mapping the state bounds to the bucket array
                bound_width = self.state_bounds[i][1] - self.state_bounds[i][0]
                position = (self.n_bucket[i] - 1) * (state[i] - self.state_bounds[i][0]) / bound_width
                bucket_index = int(round(position))

            bucket_indices.append(bucket_index)

        return tuple(bucket_indices)


    def act(self, state):
        # Select an action

        #print(self.epsilon)
        if np.random.rand() < self.epsilon:  # Select a random action
            action = self.env.action_space.sample()
        else:  # Select the action with the highest Q
            state = self._state_to_bucket(state)
            action = np.argmax(self.Q_table[state])

        return action

    def observe(self, sample, done):  # (state_0, action, reward, state_1)
        # Observe the result

        (state_0, action, reward, state_1) = sample

        # Discretize
        state_0 = self._state_to_bucket(state_0)
        state_1 = self._state_to_bucket(state_1)

        # Q-learning
        Q_best = np.amax(self.Q_table[state_1])
        self.Q_table[state_0 + (action,)] = (1 - self.alpha) * self.Q_table[state_0 + (action,)] + self.alpha * (reward + self.gamma * Q_best)

        # Update epsilon and alpha
        self.steps = self.steps+1
        self.epsilon = self.epsilon_min+(self.epsilon_max-self.epsilon_min)*np.exp(-self.tao*self.steps)
        self.alpha = self.alpha_min+(self.alpha_max-self.alpha_min)*np.exp(-self.tao*self.steps)

    def replay(self):
        # Do nothing
        return None