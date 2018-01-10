---
title: Evolution Strategies as a Scalable Alternative to Reinforcement Learning
date: 2018-01-10
---

## Summary

Usually, learning strategies attempt to minimize/maximize an objective
function $F$ parametrized by $\theta$. Thus, the goal is to search for
the best *point* (or *individual*) inside the parameter space. 

Evolution strategies (ES) frames the search problem in terms of
*populations* instead of individuals. Each parameter $\tau$
represents a population's 'genotype', but the genotype of each
individual is 'mutated' or perturbed. The goal is to maximize the
'fitness' of the population instead of the individual.

Mathematically, the 'population genotype' $\psi$ corresponds to a
distribution of 'individual genotypes' $\theta \sim p_\psi$. The goal
is then to maximize the expected objective function:
$$\mathbb{E}_{\theta \sim p_\psi} F(\theta).$$
This may be done using gradient ascent/descent, the gradient is
estimated by:
$$\nabla_\psi \mathbb{E}_{\theta \sim p_\psi} F(\theta) \approx
\mathbb{E}_{\theta \sim p_\psi} \big\{F(\theta) \nabla_\psi \log
p_\psi(\theta)\big\}.$$
And when $p_\psi$ is factored Gaussian (?? what is meant by factored
??), the estimator is also known as
[simultaneous perturbation stochastic approximation (Spall 1992)](http://www.jhuapl.edu/spsa/pdf-spsa/spall_tac92.pdf),
[parameter exploring policy gradients (Sehnke 2010)](http://kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Neural-Networks-2010-Sehnke_%5b0%5d.pdf),
and [zero-order gradient estimation (Nesterov 2011)](https://core.ac.uk/download/pdf/6340930.pdf). 

