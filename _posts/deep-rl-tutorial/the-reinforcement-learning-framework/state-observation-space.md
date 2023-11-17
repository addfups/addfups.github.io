# State/Observation Space

Observations/States are the information our agent gets from the environment. In the case of a video game, it can be a frame (a screenshot). In the case of the trading agent, it can be the value of a certain stock, etc.

There is a differentiation to make between _observation_ and _state_, however:

### _**State s:**_

> is **a complete description of the state of the world** (there is no hidden information). In a fully observed environment.

<figure><img src="../assets/2 (1).jpeg" alt=""><figcaption></figcaption></figure>

In a chess game, we have access to the whole board information, so we receive a state from the environment. In other words, the environment is fully observed.

### _**Observation o:**_

> is a **partial description of the state**. In a partially observed environment

<figure><img src="../assets/3 (1).jpeg" alt=""><figcaption></figcaption></figure>

In Super Mario Bros, we only see the part of the level close to the player, so we receive an observation. In Super Mario Bros, we are in a partially observed environment. We receive an observation since we only see a part of the level.
