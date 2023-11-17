# The Policy-Gradient Theorem

Policy-gradient is an optimization problem: we want to find the values of θ that maximize our objective function J(θ), so we need to use **gradient-ascent**. It’s the inverse of _gradient-descent_ since it gives the direction of the steepest increase of J(θ).

### Our update step for gradient-ascent is: $$\theta \leftarrow \theta + \alpha * \nabla_\theta J(\theta)$$

We can repeatedly apply this update in the hopes that θ converges to the value that maximizes J(θ).&#x20;

However, there are two problems with computing the derivative of J(θ):

1. We can’t calculate the true gradient of the objective function since it requires calculating the probability of each possible trajectory, which is computationally super expensive. So we want to **calculate a gradient estimation with a sample-based estimate (collect some trajectories)**.
2. To differentiate this objective function, we need to differentiate the state distribution, called the Markov Decision Process dynamics. This is attached to the environment. It gives us the probability of the environment going into the next state, given the current state and the action taken by the agent. The problem is that we can’t differentiate it because we might not know about it

Fortunately we’re going to use a solution called the Policy Gradient Theorem that will help us to reformulate the objective function into a differentiable function that does not involve the differentiation of the state distribution.

So we have: $$\nabla_\theta J(\theta)=\nabla_\theta \sum_\tau P(\tau;\theta)R(\tau)$$

We can rewrite the gradient of the sum as the sum of the gradient: $$= \sum_\tau \nabla_\theta   P(\tau;\theta)R(\tau)$$

We then multiply every term in the sum by $$\frac{ P(\tau;\theta)}{ P(\tau;\theta)}$$ (which is possible since it’s = 1)

$$= \sum_\tau \frac{ P(\tau;\theta)}{ P(\tau;\theta)} \nabla_\theta   P(\tau;\theta)R(\tau) \\ = \sum_\tau P(\tau;\theta) \frac{\nabla_\theta P(\tau;\theta)}{ P(\tau;\theta)}    R(\tau)$$

We can then use the _derivative log trick_ (also called _likelihood ratio trick_ or _REINFORCE trick_), a simple rule in calculus that implies that: $$\nabla_x log f(x) = \frac{\nabla_x f(x)}{f(x)}$$

So this is our likelihood policy gradient:

$$
\nabla_{\theta} J(\theta) = \sum_{\tau} P(\tau;\theta) \nabla_{\theta} \log P(\tau;\theta) R(\tau)
$$

Thanks for this new formula, we can estimate the gradient using trajectory samples (we can approximate the likelihood ratio policy gradient with a sample-based estimate if you prefer)

$$
\nabla_{\theta} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \nabla_{\theta} \log P(\tau^{(i)};\theta) R(\tau^{(i)})
$$

where each $$\tau(i)\$$ is a sampled trajectory.

But we still have some mathematics work to do there: we need to simplify $$\nabla_{\theta} \log P(\tau \mid \theta)$$.

We know that:

$$
\nabla_{\theta} \log P(\tau^{(i)};\theta) = \nabla_{\theta} \log \left[\mu(s_0) \prod_{t=0}^{H} P(s_{t+1}^{(i)} \mid s_t^{(i)}, a_t^{(i)}) \pi_{\theta}(a_t^{(i)} \mid s_t^{(i)})\right]
$$

Where $$\mu(s_0)$$ is the initial state distribution and $$P(s_{t+1}^{(i)} \mid s_t^{(i)}, a_t^{(i)})$$ is the state transition dynamics of the MDP.

We know that the log of a product is equal to the sum of the logs:

$$
\nabla_{\theta} \log P(\tau^{(i)};\theta) = \nabla_{\theta} \left[\log \mu(s_0) + \sum_{t=0}^{H} \log P(s_{t+1}^{(i)} \mid s_t^{(i)}, a_t^{(i)}) + \sum_{t=0}^{H} \log \pi_{\theta}(a_t^{(i)} \mid s_t^{(i)})\right]
$$

We also know that the gradient of the sum is equal to the sum of gradients:

$$
\nabla_{\theta} \log P(\tau^{(i)};\theta) = \\  \nabla_{\theta} \log \mu(s_0) + \nabla_{\theta} \sum_{t=0}^{H} \log P(s_{t+1}^{(i)} \mid s_t^{(i)}, a_t^{(i)}) + \nabla_{\theta} \sum_{t=0}^{H} \log \pi_{\theta}(a_t^{(i)} \mid s_t^{(i)})
$$

Since neither the initial state distribution nor the state transition dynamics of the MDP are dependent on (θ), the derivative of both terms is 0. So we can remove them:

Since: $$\nabla_{\theta} \sum_{t=0}^{H} \log P(s_{t+1}^{(i)} \mid s_t^{(i)}, a_t^{(i)}) = 0$$ and $$\nabla_{\theta} \mu(s_0) = 0$$

$$
\nabla_{\theta} \log P(\tau^{(i)};\theta) = \sum_{t=0}^{H} \nabla_{\theta} \log \pi_{\theta}(a_t^{(i)} \mid s_t^{(i)})
$$

We can rewrite the gradient of the sum as the sum of gradients:

$$
\nabla_{\theta} \log P(\tau^{(i)};\theta) = \sum_{t=0}^{H} \nabla_{\theta} \log \pi_{\theta}(a_t^{(i)} \mid s_t^{(i)})
$$

So, the final formula for estimating the policy gradient is:

$$
\nabla_{\theta} J(\theta) = \hat{g} = \frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^{H} \nabla_{\theta} \log \pi_{\theta}(a_t^{(i)} \mid s_t^{(i)}) R(\tau^{(i)})
$$
