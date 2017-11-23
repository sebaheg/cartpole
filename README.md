
In this repository we learn cartpole...

# Notebooks

* **nb1-gym-cartpolen**

This notebooks contains...

* **nb2-tql-cartpolen**
* **nb3-dql-cartpolen**

# Problems
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

1. Next problem
