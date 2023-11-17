# Monte Carlo vs Temporal Difference Learning

The idea behind Reinforcement Learning is that **given the experience and the received reward, the agent will update its value function or policy.**

Monte Carlo and Temporal Difference Learning are two different **strategies on how to train our value function or our policy function.** Both of them **use experience to solve the RL problem.**

* On one hand, Monte Carlo uses **an entire episode of experience before learning.**
* On the other hand, Temporal Difference uses **only a step** ⟮𝑆𝑡，𝐴𝑡，𝑅𝑡**+1**，𝑆𝑡**+1**⟯ **to learn.**
