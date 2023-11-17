# Experience Replay

Why do we create a replay memory? Experience Replay in Deep Q-Learning has two functions:

## **Make more efficient use of the experiences during the training**.

Usually, in online reinforcement learning, the agent interacts with the environment, gets experiences (state, action, reward, and next state), learns from them (updates the neural network), and discards them. This is not efficient.

Experience replay helps by **using the experiences of the training more efficiently**. We use a replay buffer that saves experience samples **that we can reuse during the training.**

This allows the agent to **learn from the same experiences multiple times**.

## **Avoid forgetting previous experiences and reduce the correlation between experiences**.

The problem we get if we give sequential samples of experiences to our neural network is that it tends to forget **the previous experiences as it gets new experiences.** For instance, if the agent is in the first level and then in the second, which is different, it can forget how to behave and play in the first level.



> The solution is to create a Replay Buffer that stores experience tuples while interacting with the environment and then sample a small batch of tuples. This prevents **the network from only learning about what it has done immediately before.**

Experience replay also has other benefits. By randomly sampling the experiences, we remove correlation in the observation sequences and avoid **action values from oscillating or diverging catastrophically.**

In the Deep Q-Learning pseudocode, we **initialize a replay memory buffer D with capacity N** (N is a hyperparameter that you can define). We then store experiences in the memory and sample a batch of experiences to feed the Deep Q-Network during the training phase.
