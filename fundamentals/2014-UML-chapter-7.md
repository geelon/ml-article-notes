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


### Nonuniform Learnability

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

### Structural Risk Minimization

Now, that we've generalized the problem, can we also generalize ERM?
We want to be able to say something like, *with probability $1 -
\delta$, the learning algorithm will produce a hypothesis in
$\bigcup_{n \in \mathbb{N}} \mathcal{H}_n$ with risk bounded above by
$\epsilon$ using at most sample size $m$.*

Naturally, we need therefore to say something about each hypothesis
subclass $\mathcal{H}_n$, and then union bound over all $n$. So, for
each $\mathcal{H}_n$, we will need confidence $\delta_n$, such that:
$$\sum_{n\in \mathbb{N}} \delta_n = \delta.$$
Or equivalently, define a *weight function* $w: \mathbb{N} \to
[0,1]$ such that $\sum_n w(n) = 1$, and let $\delta_n = w(n) \cdot
\delta$.

However, as $\delta_n$ approaches zero, then either the sample
complexity $m_{\mathcal{H}_n}^\mathrm{UC}$ goes up or the allowed risk
$\epsilon_n$ goes down. Let's take the latter approach, and define:
$$\epsilon_n(m,\delta) = \min \{\epsilon \in (0,1) :
m_{\mathcal{H}_n}^\mathrm{UC}(\epsilon,\delta) \leq m\},$$
the best upper bound on the gap between true and empirical risk using
$m$ samples.

Based on this setup, with the weight function $w$, the hypothesis
class $\mathcal{H} = \bigcup_{n \in \mathbb{N}} \mathcal{H}_n$, and
the error bound $\epsilon_n(m, \delta)$, we have:

**Theorem 4.** For all $\delta \in (,01)$ and distribution
  $\mathcal{D}$, with probability at least $1 - \delta$, over the
  choice of $S \sim \mathcal{D}^m$, then the following holds for all
  $n \in \mathbb{N}$ and $h \in \mathcal{H}_n$:
  $$|L_\mathcal{D}(h) - L_S(h)| \leq \epsilon_n(m, w(n) \cdot
  \delta).$$

**Corollary 5.** For $\delta$ and $\mathcal{D}$ as above, then for all
  $h \in \mathcal{H}$,
  $$L_\mathcal{D}(h) \leq L_S(h) + \min_{n :h \in \mathcal{H}_n}
  \epsilon_n(m,w(n) \cdot \delta).$$

The SRM paradigm attempts to minimize this bound (instead of the ERM
bound).

### Minimum Description Length

Suppose $\mathcal{H}$ were countable. Then, $\mathcal{H} = \bigcup_{n
\in \mathbb{N}} \{h_n\}$, where each $\mathcal{H}_n$ are singleton. It
follows from [Theorem 4](./2014-UML-chapter-4.html#finite-classes-are-agnostic-pac-learnable)
of Chapter 4 that the sample complexity for uniform convergence is
$m^\mathrm{UC}(\epsilon,\delta) = \frac{\log
(2/\delta)}{2\epsilon^2}$, and so the SRM rule is:
$$\underset{h_n \in \mathcal{H}}{\mathrm{argmin}}\left[L_S(h) +
\sqrt{\frac{-\log w(n) + \log(2/\delta)}{2m}}\right].$$

Viewing each $w(n)$ as the prior probability that $h_n$ is the target
hypothesis, the term $- log w(n)$ is just the number of bits needed to
optimally represent each $h_n$ with respect to the prior $w$. Let
$|h|$ denote the *description length* of $h$. Then, this learning
paradigm that balances the tradeoffs between empirical risk and
description length is called *minimum description length*, minimizing:
$$L_S(h) + \sqrt{\frac{|h| + \ln (2/\delta)}{2m}}.$$
This is analogous to Occam's razor, which says that a short
explanation tends to be more valid than a long explanation.

## Consistency

The notion of a consistent learner relaxes a nonuniform
learner. Instead of having bounds that hold over all distributions
$\mathcal{D}$ (for a fixed $\sigma$-algebra on $\mathcal{X}$), we just
consider bounds holding over some collection of distributions
$\mathcal{D} \in \mathcal{P}$, and allow the sample complexity to
depend on the choice of distribution (if $\mathcal{P}$ is all
distributions, then we say that the learner is *univerally
consistent*). But as a result, no upper bound on true risk is possible
from empirical risk (since sample complexity is dependent on the
specific distribution). 

- an example of a consistent learner is ```Memorize```, which
  memorizes training data and guesses on unseen data.



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
