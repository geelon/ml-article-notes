---
title: Understanding Machine Learning (Chapter 3)
date: 2018-01-24
---

Shai Shalev-Shwartz & Shai Ben-David, [Understanding Machine Learning](http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)

- [next chapter](./2014-UML-chapter-4.html): Learning via Uniform
  Convergence 

## Summary

In this chapter, PAC learning is formally defined.

**Definition 1 (Realizable PAC learning).** A (realizable) hypothesis
  class $\mathcal{H}$ is PAC learnable if there exists an algorithm
  $A$ such that for all $\epsilon, \delta \in (0,1)$, if we draw
  sufficiently many i.i.d. samples $S$, then for every distribution
  $\mathcal{D}$ over $\mathcal{X}$ and for all labeling function $f :
  \mathcal{X} \to \{0,1\}$, the algorithm returns a hypothesis $h =
  A(S)$ such that with probability $1 - \delta$,
  $L_{(\mathcal{D},f)}(h) \leq \epsilon$.

The minimal numver of samples required
$m_\mathcal{H}(\epsilon,\delta)$ given an $\epsilon$ and $\delta$ is
called the *sample complexity*.

We can generalize by removing the realizability assumption, leading to
*agnostic PAC learning* (i.e. it is not guaranteed that there exists
some hypothesis in $\mathcal{H}$ that can obtain zero risk. In this
setting, we consider a distribution $\mathcal{D}$ over the product
$\mathcal{X} \times \mathcal{Y}$, where $\mathcal{X}$ is the instance
space and $\mathcal{Y}$ is the label set. Samples are drawn from this
distribution, and in general, it is not necessarily the case that each
instance $x$ corresponds with only one label $y$.

The risk is then defined:
$$L_\mathcal{D}(h) = \underset{(x,y) \sim
\mathcal{D}}{\mathbb{P}}[h(x) \ne y].$$
It follows that the optimal classifier is defined to be:
$$f_\mathcal{D}(x) = \begin{cases}
1 & \mathbb{P}[y = 1 | x] \geq 1/2 \\
0 & \textrm{o.w.}
\end{cases}$$
This is called the *Bayes optimal predictor*. However, it is not
always possible to compute this predictor, especially if we do not
know what the distribution $\mathcal{D}$ is.

## Discussion

**Note 1.** Here, 'knowledge' about the relationship between
  $\mathcal{X}$ and $\mathcal{Y}$ is encoded into a distribution
  $\mathcal{D}$ over $\mathcal{X} \times \mathcal{Y}$. The learner
  accesses knowledge through finite i.i.d. samples from that
  distribution.   

From this framework, the question of generalizability is fundamentally
about the relationship between the empirical and the true
distribution. That is, given that we have a finite amount of knowledge
on $\mathcal{X}\times \mathcal{Y}$ (i.e. the empirical distribution),
can we ensure that there is a bound on how different the behavior is
with respect to the true distribution?  

You might ask, the behavior according to what? It is important to note
we only about the behavior facilitated through the hypothesis class
H. So ultimately, we have two questions to answer: what are the
relevant properties on D over $\mathcal{X} \times \mathcal{Y}$? and
what are the relevant properties of H? so that we may prove
generalization bounds.

**Note 2.** This theory seems limited because it only allows for
  i.i.d. samples $S$ from $\mathcal{D}$. Thus, ruling out analysis on
  active learning algorithms. Furthermore, knowledge is understood 
  distributionally, which also limits the theory. Sort of in the sense
  that a hidden Markov process behind cannot be discovered. 