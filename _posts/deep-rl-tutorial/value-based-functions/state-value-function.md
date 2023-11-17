# State Value Function

#### We write the state value function under a policy π like this:

<figure><img src="../assets/image (1).png" alt=""><figcaption></figcaption></figure>

For each state, the state-value function outputs the expected return if the agent **starts at that state** and then follows the policy forever afterward (for all future timesteps, if you prefer).

<figure><img src="../.gitbook/assets/image (7).png" alt=""><figcaption><p>If we take the state with value -7: it's the expected return starting at that state and taking actions according to our policy (greedy policy), so right, right, right, down, down, right, right.</p></figcaption></figure>
