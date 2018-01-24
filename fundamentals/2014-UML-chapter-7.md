---
title: Understanding Machine Learning (Chapter 7)
date: 2018-01-23
---

Shai Shalev-Shwartz & Shai Ben-David, [Understanding Machine Learning](http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)

- [previous chapter](./2014-UML-chapter-6.html): VC dimension

## Summary

This chapter generalizes ERM to SRM (structural risk minimization);
instead of viewing hypothesis class as just a combinatorial object
(i.e. a set), SRM provides a framework to encode priors and structure
into the hypothesis class. As a result, we no longer view hypotheses
*uniformly* (in the sense that we can have priors on $\mathcal{H}$
that aren't uniform, and in the sense that different $h \in
\mathcal{H}$ have different 'learnability' properties).

**Definition 1.** A hypothesis class $\mathcal{H}$ is *nonuniformly
  learnable* if there exists a learning algorithm, $A$, and a function
  $m_\mathcal{H}^\textrm{NUL}: (0,1)^2 \times \mathcal{H} \to
  \mathbb{N}$ such that, for every $\epsilon, \delta \in (0,1)$ and
  for every $h \in \mathcal{H}$, if $m \geq
  m_\mathcal{H}^\textrm{NUL}(\epsilon, \delta, h)$, then for every
  distribution $\mathcal{D}$, with probability at least $1 - \delta$
  over the choice of $S \sim \mathcal{D}^m$, it holds that:
  $$L_\mathcal{D}(A(S)) \leq L_\mathcal{D}(h) + \epsilon.$$

In PAC learning, the sample complexity is independent of the
hypothesis; thus nonuniform learnability is a relaxation of agnostic
PAC learnability.

**Lemma 2.** If $\mathcal{H} = \bigcup_{n \in \mathbb{N}}
  \mathcal{H}_n$, where each $\mathcal{H}_n$ has the uniform
  convergence property, then $\mathcal{H}$ is nonuniformly learnable.

**Theorem 3.** $\mathcal{H}$ is nonuniformly learnable if and only
  if it is a countable union of hypothesis classes with the uniform
  convergence property.

*Proof of Theorem.* If $\mathcal{H}$ is nonuniformly learnable, for
 fixed $\epsilon$ and $\delta$, consider the collection
 $\mathcal{H}_n$ of hypothesis learnable using sample size less than
 $n$ for each $n \in \mathbb{N}$. Then, $\mathcal{H} = \bigcup_{n \in
 \mathbb{N}} \mathcal{H}_n$. The reverse direction is immediate from
 Lemma 2. ☐

## Discussion

**Question 1.** What does a nonuniformly learnable hypothesis class
  look like? Obviously, as a result of Theorem 3 from above, it must
  be an uncountable union of hypothesis classes. But that itself is
  not sufficient, for how do we know that it cannot be written in
  another way as a countable union of uniformly learnable subclasses?

  It seems that at first glance, if $\mathcal{X}$ is uncountable, then
  the hypothesis class $\{0,1\}^\mathcal{X}$ of all possible functions
  is nonuniformly learnable. But in this case, a more fundamental
  problem occurs, where not all functions are measurable. In which
  case, $L_\mathcal{D}(h)$ may not be well-defined. (?? what about
  when $\mathcal{X}$ is countably infinite??) 

  So perhaps the largest possible hypothesis class is all measurable
  functions. How could one prove whether or not this is nonuniformly
  learnable?

**Note 2.** Perhaps we could instead think of the collection of all
  possible distributions...approximation with respect to this?
  c.f. approximating measurable functions with simple functions.

**Further reading:**

- [Brian 2015](https://arxiv.org/abs/1504.00134): From Haar to
  Lesbegue via domain theory. (To answer question whether the Cantor
  space with the Haar measure is "equivalent" (?) to the unit interval
  $[0,1]$ with the Lebesgue measure; thus, this can produce a Vitali
  set on the Cantor space).
