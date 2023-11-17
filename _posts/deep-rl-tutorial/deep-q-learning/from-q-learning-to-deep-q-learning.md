# From Q-Learning to Deep Q-Learning

We learned that **Q-Learning is an algorithm we use to train our Q-Function**, an **action-value function** that determines the value of being at a particular state and taking a specific action at that state.

Internally, our Q-function is encoded by **a Q-table, a table where each cell corresponds to a state-action pair value.** Think of this Q-table as **the memory or cheat sheet of our Q-function.**

The problem is that Q-Learning is a _tabular method_. This becomes a problem if the states and actions spaces **are not small enough to be represented efficiently by arrays and tables**. In other words: it is **not scalable**.

* **Atari environments** have an observation space with a shape of (210, 160, 3)\*, containing values ranging from 0 to 255 so that gives us 256^(210×160×3)=256^(100800) (for comparison, we have approximately 10801080 atoms in the observable universe).

Therefore, the state space is gigantic; due to this, creating and updating a Q-table for that environment would not be efficient. In this case, the best idea is to approximate the Q-values using a **parametrized Q-function** 𝑄𝜃⟮𝑠**,**𝑎⟯**.**

possible observations (for comparison, we have approximately

This neural network will approximate, given a state, the different Q-values for each possible action at that state. And that’s exactly what Deep Q-Learning does.

<figure><img src="../assets/deep.jpg" alt=""><figcaption></figcaption></figure>

