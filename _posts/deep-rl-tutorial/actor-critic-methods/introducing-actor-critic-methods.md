# Introducing Actor-Critic Methods

In Policy-Based methods, **we aim to optimize the policy directly without using a value function**. More precisely, Reinforce is part of a subclass of _Policy-Based Methods_ called _Policy-Gradient methods_. This subclass optimizes the policy directly by **estimating the weights of the optimal policy using Gradient Ascent**.

We saw that Reinforce worked well. However, because we use Monte-Carlo sampling to estimate return (we use an entire episode to calculate the return), **we have significant variance in policy gradient estimation**.

Remember that the policy gradient estimation is **the direction of the steepest increase in return**. In other words, how to update our policy weights so that actions that lead to good returns have a higher probability of being taken. The Monte Carlo variance, which we will further study in this unit, **leads to slower training since we need a lot of samples to mitigate it**.

So now we’ll study **Actor-Critic methods**, a hybrid architecture combining value-based and Policy-Based methods that helps to stabilize the training by reducing the variance using:

* _An Actor_ that controls **how our agent behaves** (Policy-Based method)
* _A Critic_ that measures **how good the taken action is** (Value-Based method)



***

### Reducing variance with Actor-Critic methods

The solution to reducing the variance of the Reinforce algorithm and training our agent faster and better is to use a combination of Policy-Based and Value-Based methods: _the Actor-Critic method_.

To understand the Actor-Critic, imagine you’re playing a video game. You can play with a friend that will provide you with some feedback. You’re the Actor and your friend is the Critic.

<figure><img src="../assets/ac.png" alt=""><figcaption></figcaption></figure>

You don’t know how to play at the beginning, **so you try some actions randomly**. The Critic observes your action and **provides feedback**.

Learning from this feedback, **you’ll update your policy and be better at playing that game.**

On the other hand, your friend (Critic) will also update their way to provide feedback so it can be better next time.

This is the idea behind Actor-Critic. We learn two function approximations:

* _A policy_ that **controls how our agent acts**: $$\pi_\theta (s)$$
* _A value function_ to assist the policy update by measuring how good the action taken is: $$\hat{q}_w(s,a)$$
