# Import libraries
import os
import gym
import numpy as np
import random
import math
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib import animation

EPISODES = 1000
class Experiment():

    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.n_episodes = 300
        self.n_animation = 0
        self.n_t = 500
        self.t_solved = 199
        self.streak_end = 120


class DQLAgent():

    def __init__(self, exp):

        # Bounds for each discrete state
        self.state_bounds = list(zip(exp.env.observation_space.low, exp.env.observation_space.high))
        # Limit the velocity bounds
        self.state_bounds[1] = [-0.5, 0.5]
        self.state_bounds[3] = [-math.radians(50), math.radians(50)]
        # Number of state variables
        self.n_states = exp.env.observation_space.shape[0]
        # Number of discrete actions
        self.n_actions = exp.env.action_space.n  # (left, right)
        # Convergence
        self.converge = np.array([])

        # Hyperparameters
        self.alpha = 0.02                 # Max learning rate
        self.alpha_min = 0.1              # Min learning rate
        self.epsilon_type = 'log'         # Exploration rate type
        self.epsilon_max = 1.0            # Max exploration rate
        self.epsilon_min = 0.01           # Min exploration rate
        self.gamma = 0.95                 # Discount rate
        self.memory = deque(maxlen=2000)  # Size of memory
        self.batch_size = 32              # Batch size

        # Build model
        self.model = self._build_model()

    def _build_model(self):

        # Model for Deep Q-learning
        model = Sequential()
        model.add(Dense(24, input_dim=self.n_states, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.n_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

        return model

    def remember(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, epsilon):

        if np.random.rand() <= epsilon:
            action = exp.env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            action = np.argmax(act_values[0])

        return action  # returns action

    def get_explore_rate(self, ep):

        if self.epsilon_type == 'const':
            epsilon = self.epsilon_max
        elif self.epsilon_type == 'log':
            epsilon = max(self.epsilon_min, min(self.epsilon_max, 1-math.log10((ep+1)/25)))
        else:
            raise ValueError('Learning rate does not exist.')

        return epsilon

    def train_model(self):

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma*np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)


def simulate(exp, agent):

    # Hyperparameters
    epsilon = agent.get_explore_rate(0)

    streak = 0
    exp.frames, exp.episodes, exp.times, exp.streaks = [], [], [], []
    for ep in range(exp.n_episodes+exp.n_animation):

        # Reset the environment
        state_0 = exp.env.reset()
        state_0 = np.reshape(state_0, [1, agent.n_states])

        for t in range(exp.n_t):

            if ep > exp.n_episodes-1:
                if ep == exp.n_episodes:
                    streak = 0
                exp.frames.append(exp.env.render(mode='rgb_array'))
                exp.episodes.append(ep-exp.n_episodes+1)
                exp.times.append(t)
                exp.streaks.append(streak)

            # Select an action
            action = agent.select_action(state_0, epsilon)

            # Execute the action
            state_1, reward, done, _ = exp.env.step(action)
            state_1 = np.reshape(state_1, [1, agent.n_states])

            agent.remember(state_0, action, reward, state_1, done)

            state_0 = state_1

            if done:
                print("Episode %d finished after %d time steps" % (ep, t))

                if (t >= exp.t_solved):
                    streak = streak+1
                else:
                    streak = 0
                break

        if len(agent.memory) > agent.batch_size:
            agent.train_model()

        # Update learning parameters
        epsilon = agent.get_explore_rate(ep)

        # TODO
        # Check how you can determine the convergence of the agent

        # # Update Q-table
        # if ep >= 0:
        #     test = agent.q_table_0-agent.q_table_1
        #     test = np.max(np.abs(test.flatten()))
        #     print(test)
        #     q_table_0_norm = np.sqrt(np.dot(agent.q_table_0.flatten(),agent.q_table_0.flatten()))
        #     q_table_1_norm = np.sqrt(np.dot(agent.q_table_1.flatten(),agent.q_table_1.flatten()))
        #     dot_product = np.dot(agent.q_table_0.flatten(), agent.q_table_1.flatten())
        #     distance = dot_product/(q_table_0_norm*q_table_1_norm)
        #     agent.converge = np.append(agent.converge, test)

if __name__ == "__main__":
    exp = Experiment()
    agent = DQLAgent(exp)

    # Simulate
    simulate(exp, agent)
