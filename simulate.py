# Import libraries
import gym
import numpy as np

# Import agents
from tql_cartpole import TQLAgent
from dql_cartpole import DQLAgent


class Experiment():
    def __init__(self):
        # Environment
        self.env = gym.make('CartPole-v1')
        self.n_states = self.env.observation_space.shape[0]  # (x, x', theta, theta')
        self.n_actions = self.env.action_space.n  # (left, right)
        # Experiment
        self.n_episodes = 1300


class RandomAgent():
    # Random agent

    def __init__(self, exp):
        self.steps = 0

    def act(self, state):
        # Select a random action
        action = exp.env.action_space.sample()
        return action

    def observe(self, sample):  # (state_0, action, reward, state_1)
        # Do nothing
        pass

    def replay(self):
        # Do nothing
        return None


def simulate(exp, agent):

    history = {'episodes': np.array([]), 'reward_cum': np.array([])}
    for ep in range(exp.n_episodes):

        # Reset the environment
        state_0 = exp.env.reset()
        reward_cum = 0

        t = 0
        while True:

            # Render
            #exp.env.render()

            # Take action
            action = agent.act(state_0)
            state_1, reward, done, _ = exp.env.step(action)
            sample = [state_0, action, reward, state_1]

            # Observe
            agent.observe(sample, done)

            # Replay
            loss = agent.replay()

            # Update state and cumulative reward
            state_0 = state_1
            reward_cum = reward_cum+reward

            if done:
                print("Episode %d finished after %d time steps" % (ep, t))
                break

            t = t+1

        history['episodes'] = np.append(history['episodes'], ep)
        history['reward_cum'] = np.append(history['reward_cum'], reward_cum)

    return history


if __name__ == "__main__":
    # Create experiment and agent
    exp = Experiment()
    agent = DQLAgent(exp) # Change here to use table q-learning

    # Simulate
    history = simulate(exp, agent)

    # Plot
    # plt.plot(history['episodes'], history['reward_cum'])
    # plt.xlabel('Episodes')
    # plt.ylabel('Reward')
    # plt.grid()
    # plt.show()
    # plt.close()