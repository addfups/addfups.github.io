# Action Value Function

#### The value of taking action a in state s under a policy π is:

<figure><img src="../assets/image (5).png" alt=""><figcaption></figcaption></figure>

In the action-value function, for each state and action pair, the action-value function **outputs the expected return** if the agent starts in that state, takes that action, and then follows the policy forever after.

#### We see that the difference is:

* For the state-value function, we calculate **the value of a state** 𝑆𝑡.
* For the action-value function, we calculate **the value of the state-action pair (** 𝑆𝑡, 𝐴𝑡 **) hence the value of taking that action at that state.**
* In either case, whichever value function we choose (state-value or action-value function), **the returned value is the expected return.**
* However, the problem is that **to calculate EACH value of a state or a state-action pair, we need to sum all the rewards an agent can get if it starts at that state.**
* This can be a computationally expensive process, and that’s **where the Bellman equation comes in to help us.**
