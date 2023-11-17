# Policy-Based Methods

{% hint style="info" %}
Now that we learned the RL framework, how do we solve the RL problem? In other words, how do we build an RL agent that can **select the actions that maximize its expected cumulative reward?**&#x20;
{% endhint %}

## **The policy** π: the agent’s brain

The Policy π is the brain of our Agent, it’s the function that tells us what action to take given the state we are in. So it defines the agent’s behavior at a given time.

<figure><img src="../assets/13 (1).jpeg" alt=""><figcaption><p>Think of policy as the brain of our agent, the function that will tell us the action to take given a state</p></figcaption></figure>

This Policy is **the function we want to learn**, our goal is to find the optimal policy π\*, the policy that **maximizes expected return** when the agent acts according to it. We find this π\* **through training.**

There are two approaches to train our agent to find this optimal policy π\*:

* **Directly**, by teaching the agent to learn which **action to take**, given the current state: **Policy-Based Methods**.
* Indirectly, **teach the agent to learn w**hich state is more valuable and then take the action that **leads to the more valuable states**: Value-Based Methods.

***

### In Policy Based methods, **we learn a policy function directly**.

<figure><img src="../assets/14 (1).jpeg" alt=""><figcaption><p><em>As we can see here, the policy (deterministic) <strong>directly indicates the action to take for each step</strong></em>.</p></figcaption></figure>

This function will define a mapping from each state to the best corresponding action. Alternatively, it could define a **probability distribution over the set of possible actions at that state** . We have two types of policies:

### _Deterministic_:

> a policy at  a given state will **always return the same action**

<figure><img src="../assets/16.jpeg" alt=""><figcaption><p>action = policy(state)</p></figcaption></figure>

<figure><img src="../assets/15 (1).jpeg" alt=""><figcaption></figcaption></figure>

### _Stochastic_:

> outputs **a probability distribution over actions.**
>
>

<figure><img src="../assets/17.jpeg" alt=""><figcaption><p>policy(actions | state) = probability distribution over the set of actions given the current state</p></figcaption></figure>

_Given an initial state, our stochastic policy will output probability distributions over the possible actions at that state._

<figure><img src="../assets/image (21).png" alt=""><figcaption></figcaption></figure>
