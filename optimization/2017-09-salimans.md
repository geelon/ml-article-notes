---
title: Evolution Strategies as a Scalable Alternative to Reinforcement Learning
date: 2018-01-10
---

Salimans, Tim, et al. "Evolution strategies as a scalable alternative to reinforcement learning." arXiv preprint arXiv:1703.03864 (2017).

## Summary

Usually, learning strategies attempt to minimize/maximize an objective
function $F$ parametrized by $\theta$. Thus, the goal is to search for
the best *point* (or *individual*) inside the parameter space. 

Evolution strategies (ES) frames the search problem in terms of
*populations* instead of individuals. Each parameter $\psi$
represents a population's 'genotype', while the individual's genotype
a mutation/perturbation of $\psi$. The goal is to maximize the
'fitness' of the population instead of the individual.

Mathematically, the 'population genotype' $\psi$ corresponds to a
distribution of 'individual genotypes', $\theta \sim p_\psi$. The goal
is then to maximize the expected objective function:
$$\mathbb{E}_{\theta \sim p_\psi} F(\theta).$$
This may be done using gradient ascent/descent, the gradient is
estimated by:
$$\nabla_\psi \mathbb{E}_{\theta \sim p_\psi} F(\theta) =
\mathbb{E}_{\theta \sim p_\psi} \big\{F(\theta) \nabla_\psi \log
p_\psi(\theta)\big\},$$
where note that the estimator never computes the gradient of $F(\theta)$.
And when $p_\psi$ is factored Gaussian (?? what is meant by factored
??), the estimator is also known as
[simultaneous perturbation stochastic approximation (Spall 1992)](http://www.jhuapl.edu/spsa/pdf-spsa/spall_tac92.pdf),
[parameter exploring policy gradients (Sehnke 2010)](http://kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Neural-Networks-2010-Sehnke_%5b0%5d.pdf),
and [zero-order gradient estimation (Nesterov 2011)](https://core.ac.uk/download/pdf/6340930.pdf).
We can think of this as applying a Gaussian blur to the original
objective function $F$.

In particular, consider the Gaussian centered at $\theta$ with
covariance $\sigma^2 I$, so that the gradient may be defined over
$\theta$, and estimated by
$$\nabla_\theta \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} F(\theta +
\sigma \epsilon) = \frac{1}{\sigma} \mathbb{E}_{\epsilon \sim
\mathcal{N}(0,I)} \big\{F(\theta + \sigma \epsilon) \epsilon\big\},$$
which may be approximated by samples, yielding the natural algorithm
labeled Algorithm 1 in the paper. This algorithm is easily
parallelizable, given as Algorithm 2.

### Empirical Results

In some cases, the Gaussian parameter perturbations did not lead to
adequate exploration of the parameter space; however, they ameliorated
this by virtual batch normalization.

### Comparison of ES and Policy Gradients

Non-smoothness is a fundamental problem in RL, especially in discrete
situations. Generally, according to the parameters $\theta$, the
learner performs an action $\mathbf{a}(\theta)$, which induces
feedback to the learner, $F(\mathbf{a}(\theta))$. Now, to enable
gradient-based methods, both ES and policy gradients add noise: ES
smooths the objective function (by adding noise to the parameter
space) while policy gradients add noise to the action space.

As the dimension $T$ of $\mathbf{a}(\theta)$ grows, the corresponding
policy gradient estimator for $\nabla_\theta F(\theta)$ will have
variance that scales with
$$\sum_{t=1}^T \nabla_\theta \log p(a_t; \theta),$$
so that the variance grows about linearly with $T$. In contrast, the
estimator using ES is independent of $T$.

### Interpretation as Finite Differences

Using the fact that $\mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}
\big\{F(\theta) \epsilon /\sigma\big\} = 0,$ we can rewrite:
$$\mathbb{E}_{\theta \sim \mathcal{N}(0,I)} \big\{F(\theta + \sigma
\epsilon) \epsilon / \sigma\big\} = \mathbb{E}_{\theta \sim
\mathcal{N}(0,I)} \big\{(F(\theta + \sigma \epsilon) -
F(\theta))\epsilon / \sigma\big\},$$
so that we can interpret ES as computing finite differences in a
randomly chosen direction.

This suggests that, as finite differences methods scale poorly with
dimension of $\theta$, ES may also scale poorly. But the
dimension perhaps depends more on the effective dimensionality of the
optimization problem and not of the learning model. (?? why is this
the case ??)

## Discussion

**Question 1:** Mathematically, we want to perform gradient
  descent/ascent on the new objective function $G = F * g$, the
  convolution with a Gaussian in the parameter space: $$\nabla_\psi
  G(\psi) = \nabla_\psi \int_{\Theta} f(\theta) g(\psi - \theta)
  d\theta.$$
  Why is Equation (2) an estimator? That is:
  $$\nabla_\psi G(\psi) = \int_\Theta f(\theta) \nabla_\psi \log g(\psi
  - \theta) d\theta.$$
  See 'log derivative trick' [here](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/), where:
  $$\nabla_x y(x) = y(x) \nabla_x \log x,$$
  according to [Sehnke 2010](http://kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Neural-Networks-2010-Sehnke_%5b0%5d.pdf).

**Question 2:** The current ES algorithm convolves the objective
  function $F$ with the Gaussian $\mathcal{N}(0,\sigma^2I)$. But
  suppose we want to apply ES to a learning model with some
  hierarchical structure (layers of a neural network, sequential 
  actions, etc.). Then consider convolving with a Gaussian that has 
  smaller variance for parameters in earlier layers and greater
  variance for later layers. Thus allowing it to learn parameters of
  the earlier layers first. Then, as trainin  goes on, reduce variance
  of later layers. Is this analogous to the batch normalization
  procedure this paper used? Does this work according to intuition?
  How may this framed in terms of entropy/generalization as discussed
  in [research directions](https://geelon.github.io/projects/files/research_direction.pdf)? 

**Aside 3:** Parameters in different levels/layers as a way to
  generate/estimate fractals. Each layer corresponds to a different
  scale of detail, so deserves different scale of $\sigma$. But in
  learning models where no clear hierarchy, can we determine/learn a
  hierarchy? Which parameters are more/less sensitive? c.f. [influence](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Kabra_Understanding_Classifier_Errors_2015_CVPR_paper.pdf)
  [functions](http://proceedings.mlr.press/v70/koh17a.html)

### Keywords/Further Reading

- policy gradients
- Q-learning
- batch normalization
- importance sampling, [Monte Carlo methods](http://ib.berkeley.edu/labs/slatkin/eriq/classes/guest_lect/mc_lecture_notes.pdf)
- influence functions