# Monte Carlo: learning at the end of the episode

## **Monte Carlo approach:**

Monte Carlo waits until the end of the episode, calculates 𝐺𝑡 (return) and uses it as **a target for updating** 𝑉⟮𝑆𝑡⟯**.**

So it requires a **complete episode of interaction before updating our value function.**

<figure><img src="../../assets/image (20).png" alt=""><figcaption></figcaption></figure>

### If we take an example:

<figure><img src="../../assets/image (10).png" alt=""><figcaption></figcaption></figure>

* We always start the episode **at the same starting point.**
* **The agent takes actions using the policy**. For instance, using an Epsilon Greedy Strategy, a policy that alternates between exploration (random actions) and exploitation.
* We get **the reward and the next state.**
* We terminate the episode if the cat eats the mouse or if the mouse moves > 10 steps.

At the end of the episode:

* **We have a list of State, Actions, Rewards, and Next States tuples.**
* **The agent will sum the total rewards** 𝐺𝑡 (to see how well it did).
* It will then **update** 𝑉⟮𝑆𝑡⟯ **based on the formula.**
* Then **start a new game with this new knowledge better.**

> By running more and more episodes, **the agent will learn to play better and better.**

***

For instance, if we train a state-value function using Monte Carlo:

* We initialize our value function **so that it returns 0 value for each state.**
* Our learning rate (lr) is 0.1 and our discount rate is 1 (= no discount).
* Our mouse **explores the environment and takes random actions.**

<figure><img src="../../assets/image (15).png" alt=""><figcaption></figcaption></figure>

* The mouse made more than 10 steps, so the episode ends.&#x20;
* We have a list of state, action, rewards, next\_state, **we need to calculate the return** 𝐺𝑡.
  * 𝐺𝑡=𝑅𝑡**+1**+𝑅𝑡**+2**+𝑅𝑡**+3**...(for simplicity we don’t discount the rewards).
  * 𝐺𝑡=1+0+0+0+0+0+1+1+0+0Gt​=1+0+0+0+0+0+1+1+0+0
  * 𝐺𝑡=3
* We can now update 𝑉⟮𝑆**0**⟯:
  * New 𝑉⟮𝑆**0**⟯=𝑉⟮𝑆**0**⟯+lr∗\[𝐺𝑡—𝑉⟮𝑆**0**⟯)]
  * New 𝑉⟮𝑆**0**⟯=0+0.1∗\[3–0]𝑉⟮𝑆**0**⟯
  * New 𝑉⟮𝑆**0**⟯=0.3V
