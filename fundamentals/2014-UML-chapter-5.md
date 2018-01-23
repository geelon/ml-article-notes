---
title: Understanding Machine Learning (Chapter 5)
date: 2018-01-19
---

Shai Shalev-Shwartz & Shai Ben-David, [Understanding Machine Learning](http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)

- [previous chapter](./2014-UML-chapter-4.html): Learning via Uniform
  Convergence 
- [next chapter](./2014-UML-chapter-6.html): VC Dimension

## Summary

### No Free Lunch Theorem

The No Free Lunch theorem shows that there is no universal learner:
that is, for every learning algorithm $A$, there will be a situation
in which the learner will fail. Rigorously,

**Theorem 1.** Let $A$ be a learning algorithm, with the binary
  classification task, 0-1 loss over the domain $\mathcal{X}$. Let $m
  \leq |\mathcal{X}|/2$ be the training sample size. Then, there
  exists distribution $\mathcal{D}$ over $\mathcal{X} \times \{0,1\}$
  such that:
  
  1. There exists a function $f: \mathcal{X} \to \{0,1\}$ with
  $L_\mathcal{D}(f) = 0$.
  2. With probability at least $1/7$ over the choice of $S \sim
  \mathcal{D}^m$, we have $L_\mathcal{D}(A(S)) \geq 1/8$.


### Error Decomposition

Let $h_S$ be an $\mathrm{ERM}_\mathcal{H}$ hypothesis. Then, we can
decompose the true risk into the sum of *approximation error* and
*estimation error*:
$$L_\mathcal{D}(h_S) = \epsilon_{\mathrm{app}} +
\epsilon_{\mathrm{est}}.$$
The *approximation error* is the minimum risk achievable by a
hypothesis in $\mathcal{H}$. That is:
$$\epsilon_{\mathrm{app}} = \min_{h \in \mathcal{H}}
L_\mathcal{D}(h).$$
In other words, if we take $\mathcal{H}$ to be a *model*, then the
approximation error is the error in the model---the *inductive bias*
due to how we restricted the hypothesis class. Approximation error is
independent of the learning algorithm.

The *estimation error* is the error due to the learning
algorithm---the failure to determine the true ERM predictor (as the
empirical risk is only an estimator of the true risk).

This error depends on the size of $\mathcal{H}$ and the sample size
$m$. Recalling the previous
[notes](./2014-UML-chapter-4.html) (Corollary 5), the
relationship is:
$$\epsilon = O\left(\sqrt{\frac{\log |\mathcal{H}|}{m}}\right).$$

Thus, to reduce approximation error, we should increase the size of
$\mathcal{H}$ (i.e. increase the *complexity* of the model). However,
this may lead to an increase in estimation error (in particular,
*overfitting* if the sample size is not sufficiently large). And so,
we face the *bias-complexity tradeoff*.


## Discussion

**Question 1.** Can we prove the analogous No Free Lunch theorem from
  a computational complexity point of view? That is, for any
  polynomial-time learner, there exists a task that the learner will
  fail to learn. Does the complexity point of view generalize the PAC
  learning setting?