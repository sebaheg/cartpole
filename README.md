
In this repository we learn cartpole...

# Notebooks

* **nb1-gym-cartpolen**

This notebooks contains...

* **nb2-tql-cartpolen**
* **nb3-dql-cartpolen**

# Code problems
The current problems with the code.

1. Convergence of table Q-learning
The convergence of table Q-learning seem to be very sensitive to the learning parameters. Changing ```tao``` from 0.0002 to 0.0004 might cause the algorithm to not converge anymore. Is it possible to make the algorithm more robust? What is the cause of not converging?

1. Must set terminal state to zero for deep Q-learning
The deep Q-learning agent has needs to set the terminal state to zero otherwise it will not converge, why is this?

This is done with the code:

```
if done:  # Terminal state
    sample[3] = None
```

and then

```
no_state = np.zeros(self.exp.n_states)

states_0 = np.array([o[0] for o in batch])
states_1 = np.array([(no_state if o[3] is None else o[3]) for o in batch])
```

# To do
* In order to better understand the cartpole problem. We should plot the learned function (by the function approximition) in a contour plot. To do this we can discard two of the state variables in the problem. and only use the position based variables. Is this correct? Not Markov anymore but might still work?


# Theory discussion topics
* Difference between value iteration and policy iteration. In practice, value iteration is much faster per iteration but policy iteration takes fewer iterations.
* "In discrete environments, there is a guarantee that any operation that updates the value function (according to the Bellman Equations) can only reduce the error between the current value function and the optimal value function. This guarantee no longer holds when generalization is used."
* How does it work with trajectories. Refer to the Bellman optimality in the study group slides.
