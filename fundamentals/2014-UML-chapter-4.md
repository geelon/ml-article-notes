---
title: Understanding Machine Learning (Chapter 4)
date: 2018-01-18
---

Shai Shalev-Shwartz & Shai Ben-David, [Understanding Machine Learning](http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)

- [next chapter](./2014-UML-chapter-5.html): No Free Lunch and Error
  Decomposition 

## Summary

**Notation:** $\mathcal{H}$ is a *hypothesis class* on the domain $Z$,
  with loss function $\ell$ and distribution $\mathcal{D}$. Denote the
  *empirical risk* of a hypothesis $h \in \mathcal{H}$ on a sample $S
  \subset Z$ by $L_S(h)$ and the *true risk* by
  $L_\mathcal{D}(h)$. Let $h^*$ be the *target hypothesis*, minimizing
  $L_D$. Let $h_S$ be the *empirical risk minimizer*,
  $\mathrm{ERM}(S)$. 

**Definition 1.** A training set $S$ is called
  *$\epsilon$-representative* if the empirical risk $L_S$ is always
  within $\epsilon$ of the true risk $L_\mathcal{D}$:
  $$\forall h \in \mathcal{H}, |L_S(h) - L_\mathcal{D}(h)| \leq
  \epsilon.$$

It follows that if $S$ is $\frac{\epsilon}{2}$-representative,
then by definition, $L_S(h^*)$ is less than $L_\mathcal{D}(h^*) +
\frac{\epsilon}{2}$. The emipirical risk minimizer $h_S$ thus has
empirical risk at most $L_\mathcal{D}(h^*) + \frac{\epsilon}{2}$ as
well. And again, as $S$ is $\frac{\epsilon}{2}$-representative, the
true risk of $h_S$ is less than $L_\mathcal{D}(h^*) + \epsilon$.

*This shows that if with probability $1 - \delta$, we are sure to
 obtain an $\frac{\epsilon}{2}$-representative, then we may agnostic
 $(\epsilon,\delta)$-PAC learn.* 

**Definition 2.** A hypothesis class $\mathcal{H}$ has *uniform
  convergence property* if for all distributions and all $\epsilon,
  \delta \in (0,1)$, then w.p. $1 - \delta$, any sufficiently 
  large i.i.d. sample $S$ will be $\epsilon$-representative.

Concretely, this implies there is a function
$m_{\mathcal{H}}^{\mathrm{UC}} : (0,1)^2 \to \mathbb{N}$ such that if
the sample size is greater than
$m_{\mathcal{H}}^{\mathrm{UC}}(\epsilon,\delta)$, then $S$ is
$\epsilon$-representative with probability $1 - \delta$. The function
$m_{\mathcal{H}}^{\mathrm{UC}}$ is the (minimal) sample complexity of
obtaining the UC property.

**Corollary 3.** If $\mathcal{H}$ has uniform convergence property,
  corresponding to the UC sample complexity
  $m_\mathcal{H}^\mathrm{UC}$, then $\mathcal{H}$ is agnostic-PAC
  learnable with sample complexity:
  $$m_\mathcal{H}(\epsilon,\delta)\leq m_\mathcal{H}^\mathrm{UC}(\epsilon/2,\delta).$$

### Finite Classes are Agnostic PAC Learnable

**Theorem 4.** Suppose $\mathcal{H}$ is finite. Then $\mathcal{H}$ has uniform
convergence property with UC sample complexity:
$$m_\mathcal{H}^\mathrm{UC}(\epsilon,\delta) \leq \left\lceil
\frac{\log (2|\mathcal{H}|/\delta)}{2\epsilon^2}\right\rceil.$$

This theorem combined with Corollary 3 immediately implies that
$\mathcal{H}$ is agnostic PAC-learnable. The proof of Theorem 4 is
simple: Hoeffding's then union bound:

- To be able to apply union bound, we require that for all $h \in
  \mathcal{H}$,
  $$\mathrm{Pr}\big[|L_S(h) - L_\mathcal{D}(h)| > \epsilon\big] <
  \frac{\delta}{|\mathcal{H}|}.$$
- Because $L_S$ (as an estimator of $L_\mathcal{D}$) is computed
  by taking the average of $\ell(h,x_i)$ over the i.i.d. samples
  $x_1,\dotsc, x_m$, Hoeffding's implies that 
  $$\mathrm{Pr}\big[\left|L_S(h) - L_\mathcal{D}(h) \right| >
  \epsilon\big] \leq 2 \exp\left(-2 m\epsilon^2\right).$$
  
These two equations show that $m_\mathcal{H}^\mathrm{UC}$ should
satisfy Equation (3).

**Corollary 5.** If $\mathcal{H}$ is finite, then $\mathcal{H}$ is
  $(\epsilon,\delta)$ PAC-learnable, with sample complexity:
  $$m_\mathcal{H} =
  O\left(\frac{1}{\epsilon^2} \left(\log|\mathcal{H}| + \log \frac{1}{\delta}\right)\right).$$ 