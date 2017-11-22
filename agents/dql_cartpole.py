# Import libraries
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQLAgent():
    # Deep Q-learning agent

    def __init__(self, exp):
        # Hyperparameters
        self.epsilon_max = 1.0            # Max exploration rate
        self.epsilon_min = 0.01           # Min exploration rate
        self.epsilon = self.epsilon_max
        self.tao = 0.001
        self.gamma = 0.99                 # Discount rate
        self.capacity = 100               # Size of memory
        self.batch_size = 64              # Batch size
        self.steps = 0
        self.exp = exp

        # Initiate brain and memory
        self.brain = Brain(exp.n_states, exp.n_actions, self.batch_size)
        self.memory = Memory(self.capacity)

    def act(self, state):
        # Select an action
        if np.random.rand() < self.epsilon:
            action = self.exp.env.action_space.sample()
        else:
            state = state.reshape(1, self.exp.n_states)  # (samples, state)
            Q = self.brain.predict(state)
            action = np.argmax(Q)
        return action

    def observe(self, sample, done):  # (state_0, action, reward, state_1)
        # Add experience to memory and update epsilon
        if done:  # Terminal state
            sample[3] = None
            # Why is this important?

        self.memory.add(sample)
        self.steps = self.steps+1
        self.epsilon = self.epsilon_min+(self.epsilon_max-self.epsilon_min)*np.exp(-self.tao*self.steps)

    def replay(self):
        # Sample batch from memory
        batch = self.memory.sample(self.batch_size)

        no_state = np.zeros(self.exp.n_states)

        states_0 = np.array([o[0] for o in batch])
        states_1 = np.array([(no_state if o[3] is None else o[3]) for o in batch])

        p_0 = self.brain.predict(states_0)
        p_1 = self.brain.predict(states_1)

        x = np.zeros((self.batch_size, self.exp.n_states))
        y = np.zeros((self.batch_size, self.exp.n_actions))

        for i, sample in enumerate(batch):
            (state_0, action, reward, state_1) = sample

            target = p_0[i]
            if state_1 is None:
                target[action] = reward
            else:
                target[action] = reward+self.gamma*np.amax(p_1[i])

            x[i] = state_0
            y[i] = target

        loss = self.brain.train(x, y)

        return loss

class Brain():
    def __init__(self, n_states, n_actions, batch_size):
        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.units = 64
        self.lr = 0.00025
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(units=self.units, input_dim=self.n_states, activation='relu'))
        model.add(Dense(units=self.n_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def train(self, x, y, epochs=1, verbose=0):
        loss = self.model.fit(x, y, batch_size=self.batch_size, epochs=epochs, verbose=verbose)
        return loss

    def predict(self, s):
        return self.model.predict(s)

class Memory():
    samples = []  # (state_0, action, reward, state_1)

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)