---
title: Understanding Machine Learning (Chapter 6)
date: 2018-01-20
---

Shai Shalev-Shwartz & Shai Ben-David, [Understanding Machine Learning](http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)

## Summary

The main result in this chapter is that finite VC dimension
characterize PAC learnability. In addition, it gives upper and lower
bounds on sample complexity.

**Definition 1.** The *VC-dimension* of a hypothesis class
  $\mathcal{H}$, denoted $\mathrm{VCdim}(\mathcal{H})$, is the maximal
  size of a set $C \subset \mathcal{X}$ that can be shattered by
  $\mathcal{H}$. If $\mathcal{H}$ can shatter sets of arbitrarily
  large size we say that $\mathcal{H}$ has infinite VC-dimension.

Here is one direction of how VC dimension characterizes
PAC-learnability:

**Theorem 2.** If $\mathcal{H}$ has infinite VC dimension, then it is
  not PAC learnable.

*Proof.* For all sample size $m$, consider a subset $C \subset
 \mathcal{X}$ of size $2m$ that is shattered by $\mathcal{H}$. Then,
 by the No Free Lunch theorem in [notes](./2014-UML-chapter-5.html) of
 the previous chapter, there exists a distribution on $\mathcal{X}
 \times \{0,1\}$ such that $\mathcal{H}$ is realizable but the
 learning algorithm will fail to produce hypothesis with low true
 risk. Thus, no finite sample size is sure to be sufficient to learn
 the model $\mathcal{H}$. ☐

To prove the other direction, we need to define the *growth function*
of a hypothesis class and show Sauer's lemma:

**Defintion 3.** The *growth function* of $\mathcal{H}$,
  $\tau_\mathcal{H} : \mathbb{N} \to \mathbb{N}$, is defined as:
  $$\tau_\mathcal{H}(m) = \max_{C\subset \mathcal{X}: |C| = m}
  \big|\mathcal{H}_C\big|.$$

**Lemma 3 (Sauer-Shelah-Perles).** Let $\mathrm{VCdim}(\mathcal{H})
  \leq d < \infty$. Then, for all $m$, $\tau_\mathcal{H}(m)
  \leq\sum_{i=0}^d \binom{m}{i}$.  In particular, if $m > d + 1$ then
  $\tau_\mathcal{H}(m) \leq (em/d)^d$.


