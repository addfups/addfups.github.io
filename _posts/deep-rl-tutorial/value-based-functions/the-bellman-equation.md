# The Bellman Equation

Instead of calculating the expected return for each state or each state-action pair, **we can use the Bellman equation.**

The Bellman equation is a recursive equation that works like this:

<figure><img src="../assets/image (22).png" alt=""><figcaption></figcaption></figure>

Instead of starting for each state from the beginning and calculating the return, we can consider the value of any state as:

> **The immediate reward** 𝑅𝑡_**+1**_** + the discounted value of the state that follows (** 𝑔𝑎𝑚𝑚𝑎∗𝑉**(**𝑆𝑡_**+1**_**) ) .**



{% hint style="info" %}
To recap, the idea of the Bellman equation is that instead of calculating each value as the sum of the expected return, **which is a long process**, we calculate the value as **the sum of immediate reward + the discounted value of the state that follows.**
{% endhint %}
