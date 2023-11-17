# Adding Advantage in Actor-Critic (A2C)

We can stabilize learning further by using the Advantage function as Critic instead of the Action value function.

The idea is that the Advantage function calculates the relative advantage of an action compared to the others possible at a state: **how taking that action at a state is better compared to the average value of the state**. It’s subtracting the mean value of the state from the state action pair:

&#x20;

<figure><img src="../assets/image (11).png" alt=""><figcaption></figcaption></figure>

In other words, this function calculates **the extra reward we get if we take this action at that state compared to the mean reward we get at that state**.

The extra reward is what’s beyond the expected value of that state.

* If A(s,a) > 0: our gradient is **pushed in that direction**.
* If A(s,a) < 0 (our action does worse than the average value of that state), **our gradient is pushed in the opposite direction**.

The problem with implementing this advantage function is that it requires two value functions — $$Q(s,a)$$  and $$V(s)$$. Fortunately, **we can use the TD error as a good estimator of the advantage function.**

<figure><img src="../assets/image (13).png" alt=""><figcaption></figcaption></figure>

