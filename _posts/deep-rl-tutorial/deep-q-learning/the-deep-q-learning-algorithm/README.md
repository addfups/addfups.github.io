# The Deep Q-Learning Algorithm

We learned that Deep Q-Learning **uses a deep neural network to approximate the different Q-values for each possible action at a state** (value-function estimation).

The difference is that, during the training phase, instead of updating the Q-value of a state-action pair directly as we have done with Q-Learning:

<figure><img src="../../.gitbook/assets/Q-learning-8.png" alt=""><figcaption></figcaption></figure>

{% hint style="info" %}
In Deep Q-Learning, we create a **loss function that compares our prediction and the gradient descent to update the weights of our Deep Q-Network to approximate our Q-values better**.
{% endhint %}

## The Deep Q-Learning training algorithm has _two phases_:

1. **Sampling**: we perform actions and **store the observed experience tuples in a replay memory**.
2. **Training**: Select a **small batch of tuples randomly and learn from this batch using a gradient descent update step**.

<figure><img src="../../.gitbook/assets/image (19).png" alt=""><figcaption></figcaption></figure>

This is not the only difference compared with Q-Learning.

Deep Q-Learning training **might suffer from instability**, mainly because of combining a **non-linear Q-value function** (Neural Network) and **bootstrapping** (when we update targets with existing estimates and not an actual complete return).

To help us stabilize the training, we implement three different solutions:

<table data-header-hidden><thead><tr><th width="68"></th><th></th></tr></thead><tbody><tr><td><em>1.</em></td><td><em>Experience Replay</em> to make more <strong>efficient use of experiences</strong>.</td></tr><tr><td><em>2.</em></td><td><em>Fixed Q-Target</em> <strong>to stabilize the training</strong>.</td></tr><tr><td>2.</td><td><em>Double Deep Q-Learning</em>, to <strong>handle the problem of the overestimation of Q-values</strong>.</td></tr></tbody></table>

