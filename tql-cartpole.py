# Import libraries
import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation


class Experiment():

    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.n_episodes = 300
        self.n_animation = 0
        self.n_t = 500
        self.t_solved = 199
        self.streak_end = 120


class TQLAgent():

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
        # Q-Tables
        self.q_table_0 = np.zeros(self.n_bucket+(self.n_actions,))
        self.q_table_1 = np.zeros(self.n_bucket+(self.n_actions,))
        # Convergence
        self.converge = np.array([])

        # Hyperparameters
        self.alpha_type = 'const'   # Learning rate type
        self.alpha_max = 0.2        # Max learning rate
        self.alpha_min = 0.1        # Min learning rate
        self.epsilon_type = 'log' # Exploration rate type
        self.epsilon_max = 1.0      # Max exploration rate
        self.epsilon_min = 0.01     # Min exploration rate
        self.gamma = 0.95           # Discount rate

    def select_action(self, state, epsilon):

        if np.random.rand() < epsilon:  # Select a random action
            action = exp.env.action_space.sample()
        else:  # Select the action with the highest q
            action = np.argmax(self.q_table_1[state])

        return action

    def get_explore_rate(self, ep):

        if self.epsilon_type == 'const':
            epsilon = self.epsilon_max
        elif self.epsilon_type == 'log':
            epsilon = max(self.epsilon_min, min(self.epsilon_max, 1-math.log10((ep+1)/25)))
        else:
            raise ValueError('Learning rate does not exist.')

        return epsilon

    def get_learning_rate(self, ep):

        if self.alpha_type == 'const':
            alpha = self.alpha_max
        elif self.alpha_type == 'log':
            alpha = max(self.alpha_min, min(self.alpha_max, 1-math.log10((ep+1)/25)))
        else:
            raise ValueError('Learning rate does not exist.')

        return alpha

    def state_to_bucket(self, state):

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


def simulate(exp, agent):

    # Hyperparameters
    alpha = agent.get_learning_rate(0)
    epsilon = agent.get_explore_rate(0)
    gamma = agent.gamma

    streak = 0
    exp.frames, exp.episodes, exp.times, exp.streaks = [], [], [], []
    for ep in range(exp.n_episodes+exp.n_animation):

        # Reset the environment
        obv = exp.env.reset()

        # The initial state
        state_0 = agent.state_to_bucket(obv)

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
            obv, reward, done, _ = exp.env.step(action)

            # Observe the result
            state_1 = agent.state_to_bucket(obv)

            # Q-learning
            q_best = np.amax(agent.q_table_1[state_1])
            agent.q_table_1[state_0+(action,)] = (1-alpha)*agent.q_table_1[state_0+(action,)]+alpha*(reward+gamma*q_best)

            # Setting up for the next iteration
            state_0 = state_1

            if done:
                print("Episode %d finished after %d time steps" % (ep, t))

                if (t >= exp.t_solved):
                    streak = streak+1
                else:
                    streak = 0
                break

        # Update learning parameters
        epsilon = agent.get_explore_rate(ep)
        alpha = agent.get_learning_rate(ep)

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

        agent.q_table_0 = agent.q_table_1


def save_animation(exp):

    fig = plt.figure()  # make figure
    ax = fig.add_subplot(111)

    # Initial
    im = ax.imshow(exp.frames[0])
    ax.text(0, 0, 'Trained on '+str(exp.n_episodes)+' episode', fontsize=15, color='black')
    txt1 = ax.text(0, 80, 'Episode: ', fontsize=15, color='black')
    txt2 = ax.text(0, 160, 'Time: ', fontsize=15, color='black')
    txt3 = ax.text(0, 240, 'Time: ', fontsize=15, color='black')
    plt.axis('off')

    def animate(i):
        im.set_data(exp.frames[i])
        txt1.set_text('Episode: ' + str(exp.episodes[i]))
        txt2.set_text('Time: ' + str(exp.times[i]))
        txt3.set_text('Streaks: ' + str(exp.streaks[i]))

        return im, [txt1, txt2]

    anim = animation.FuncAnimation(fig, animate, frames=len(exp.frames), interval=50)

    # Save animation
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    anim.save('cartpole-episodes'+str(exp.n_episodes)+'.mp4', writer=writer)


if __name__ == "__main__":
    exp = Experiment()
    agent = TQLAgent(exp)

    # Simulate
    simulate(exp, agent)

    # Animation
    if exp.n_animation > 0:
        save_animation(exp)

