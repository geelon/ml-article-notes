---
title: Understanding Machine Learning (Chapter 6)
date: 2018-01-22
---

Shai Shalev-Shwartz & Shai Ben-David, [Understanding Machine Learning](http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)

- [previous chapter](./2014-UML-chapter-5.html): No Free Lunch and
  Error Decomposition
- [next chapter](#): Nonuniform Learnability

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

Next, Sauer's lemma will show that $|\mathcal{H}_C|$ will grow only
polynomially in $|C|$, if $\mathcal{H}$ has finite VC dimension:

**Lemma 3 (Sauer-Shelah-Perles).** Let $\mathrm{VCdim}(\mathcal{H})
  \leq d < \infty$. Then, for all $m$, $\tau_\mathcal{H}(m)
  \leq\sum_{i=0}^d \binom{m}{i}$.  In particular, if $m > d + 1$ then
  $\tau_\mathcal{H}(m) \leq (em/d)^d$.

*Proof.* Let $C = \{c_1,\dotsc, c_m\} \subset \mathcal{X}$ be any
sample of size $m$. We can show the stronger statement: 
$$\tau_\mathcal{H}(m) \leq \big|\{B \subset C : \mathcal{H}_C
\textrm{ shatters } B\}\big| \leq \sum_{i=0}^d \binom{m}{i}.$$
The max number of possible subsets of $C$ that are shattered by
$\mathcal{H}$ is given by the right hand side, if the size of the
shattered set is at most $d$. This shows the second inequality.

To prove the first inequality, we use induction. The base case is
trivial (note that $\mathcal{H}$ always shatters $\emptyset \subset
\mathcal{X}$). To show the inductive step, let $C' = \{c_1, \dotsc,
c_{m-1}\}$. Consider $\mathcal{H}_{C'}$, which partitions
$\mathcal{H}_C$ into classes of sizes 1 or 2 (partitioned by the
equivalence relation, $h_0 \sim h_1$ iff $h_0\big|_{C'} =
h_1\big|_{C'}$). Notice that the classes of size 2 are only possible
if there exists $h_0 \sim h_1$ that disagree on $c_m$. 
Denote by $\mathcal{H}'$ the union of such equivalence classes with 2
elements, so that:
$$|\mathcal{H}_C| = |\mathcal{H}_{C'}| + |\mathcal{H}_{C'}'|.$$
Now, we can apply the inductive hypothesis:
$$\begin{eqnarray*}
|\mathcal{H}_{C'}| \leq \big|\{B \subset C' : \mathcal{H}_{C'}
\textrm{ shatters } B\}\big| = \big|\{B \subset C : \mathcal{H}_C
\textrm{ shatters } B \wedge c_m \notin B\}\big|,\\
|\mathcal{H}_{C'}'| \leq \big|\{B \subset C' : \mathcal{H}_{C'}'
\textrm{ shatters } B\}\big| = \big|\{B \subset C : \mathcal{H}_C
\textrm{ shatters } B \wedge c_m \in B\}\big|.
\end{eqnarray*}$$

We conclude by appealing to Equation (4); we sum the two above
inequalities, and since:
$$\begin{gather*}\{B \subset C: \mathcal{H}_C \textrm{ shatters } B\}
\\
=\\
\big|\{B \subset C : \mathcal{H}_C \textrm{ shatters } B \wedge c_m
\notin B\}\big| \cup \big|\{B \subset C : \mathcal{H}_C
\textrm{ shatters } B \wedge c_m \in B\}\big|,\end{gather*} $$
we obtain our desired inequality. (See Lemma A.1 below for analytic
bound). ☐

The following lemma helps us bound the estimation error, as a function
of how fast the effective size of the hypothesis class grows with
respect to the sample size:

**Lemma 4.** For every $\mathcal{D}$ and $\delta \in (0,1)$, with
  probability at least $1 - \delta$ over the choice of $S \sim
  \mathcal{D}^m$, we have:
  $$|L_\mathcal{D}(h) - L_S(h)| \leq \frac{4 + \sqrt{\log
  \tau_\mathcal{H}(2m)}}{\delta \sqrt{2m}}.$$
  

**Theorem 5 (Fundamental Theorem of Statistical Learning).** Let
  $\mathcal{H}$ be a hypothesis class of funtions from $\mathcal{X}$
  to $\{0,1\}$ with the 0-1 loss. Then, $\mathcal{H}$ has uniform
  convergence property if and only if $\mathcal{H}$ has finite VC
  dimension. 

*Proof of Theorem 5.* Reiterating, if $\mathcal{H}$ has infinite VC
 dimension, then it is not PAC learnable; hence, it does not have
 uniform convergence property. What's left to show is that if
 $\mathcal{H}$ has finite VC dimension, then it has uniform
 convergence property.
 
 So, let us just plug the upper bound on $\tau_\mathcal{H}$ from
 Sauer's lemma into Lemma 4 (Equation 4):[^simplification]
 $$|L_S(h) - L_\mathcal{D}(h)| \leq \frac{1}{\delta} \sqrt{\frac{2d
 \log(2em/d)}{m}}.$$
 This term may be bounded above by arbitrarily small epsilon, given
 the appropriate choice of $m$. But notice that it was crucial that
 $\tau_\mathcal{H}$ is polynomial and not exponential in
 $m$. Otherwise, a bound by an arbitrary $\epsilon > 0$ would have
 been impossible. ☐

*Proof of Lemma 4.* We'll just show that:
$$\underset{S\sim \mathcal{D}^m}{\mathbb{E}}\left[\sup_{h \in
\mathcal{H}}|L_S(h) - L_\mathcal{D}(h)|\right] \leq \frac{4 +
\sqrt{\log\tau_\mathcal{H}(2m)}}{\sqrt{2m}},$$
where applying Markov's inequality will immediately yield the lemma.

Notice that $L_\mathcal{D}(h) = \mathbb{E}_{S' \sim \mathcal{D}^m}
L_{S'}(h)$. So, in fact, we can replace and apply Jensen's inequality:
$$\begin{align*}
\underset{S\sim \mathcal{D}^m}{\mathbb{E}}\left[\sup_{h \in
\mathcal{H}}|L_S(h) - L_\mathcal{D}(h)|\right] &=
\underset{S\sim \mathcal{D}^m}{\mathbb{E}}\left[\sup_{h \in
\mathcal{H}}|L_S(h) - \underset{S' \sim \mathcal{D}^m}{\mathbb{E}}
L_{S'}(h)|\right] \\
&\leq  \underset{S, S' \sim \mathcal{D}^m}{\mathbb{E}} \left[\sup_{h
\in \mathcal{H}} |L_S(h) - L_{S'}(h)|\right] .
\end{align*}$$
But since $S = \{z_1,\dotsc, z_m\}$ and $S' = \{z_1',\dotsc, z_m'\}$
are i.i.d., we can sample $S$ and $S'$ and swap them without affecting
the expectation. Indeed, we can swap the individual $z_i$ and $z_i'$
as well. So, for each index $1 \leq i \leq m$, uniformly at random
choose choose $\sigma_i\in_R \{\pm 1\}$. We have:
$$\begin{align*}
  \underset{S, S' \sim \mathcal{D}^m}{\mathbb{E}} \left[\sup_{h
\in \mathcal{H}} |L_S(h) - L_{S'}(h)|\right]
&= \underset{\sigma \sim
U_{\pm}^m}{\mathbb{E}} \underset{\ S, S' \sim \mathcal{D}^m\ }{\mathbb{E}} \left[\sup_{h\in \mathcal{H}}
\frac{1}{m}\left|\sum_{i=1}^m \sigma_i(\ell(h,z_i) -
\ell(h,z_i'))\right|\right]\\
&= \underset{\ S, S' \sim \mathcal{D}^m\ }{\mathbb{E}} \left[\underset{\sigma \sim
U_{\pm}^m}{\mathbb{E}} \max_{h\in \mathcal{H}}
\underbrace{\frac{1}{m}\left|\sum_{i=1}^m \sigma_i(\ell(h,z_i) -
\ell(h,z_i'))\right|}_{\theta_h}\right]
\end{align*}$$
Call the braced term $\theta_h$. Since $\mathbb{E}[\theta_h] = 0$ is
the average of independent variables, Hoeffding's implies that for all
$\rho > 0$,
$$\mathbb{P}\big[|\theta_h|> \rho\big] \leq 2 \exp(-2 m\rho^2).$$
Applying union bound implies
$$\mathbb{P}\left[\max_{h \in \mathcal{H}_C}|\theta_h|> \rho\right] \leq
2|\mathcal{H}_C| \exp(-2 m\rho^2).$$
Using Lemma A.2 below, this implies Equation (6).  ☐


[^simplification]: Note that we assume that $\sqrt{d \log (2em/d)}
\geq 4$ for simplicity.

### Aside
**Lemma A.1.** Let $m,d$ be two positive integers such that $d \leq
m-2$. Then,
$$\sum_{k=0}^d \binom{m}{k} \leq \left(\frac{em}{d}\right)^d.$$
See text for proof; note also that it uses Stirling's formula.

**Lemma A.2.** Let $X$ be a random variable and $x' \in \mathbb{R}$ be
  a scalar. Assume that there exists $a > 0$ and $b \geq e$ such that
  for all $t \geq 0$, we have $\mathbb{P}\big[|X - x'| > t\big] \leq
  2be^{-t^2/a^2}$. Then,
  $$\mathbb{E}\big[|X - x'|\big] \leq a(2 + \sqrt{\log b}).$$




## Discussion

**Note 1.** In the statistical setting, we assume an upper limit in
  how many samples $m$ we see. Then, bounds are shown by exhibiting 
  distributions for which such a limited sample size will fail to
  reduce error. If we want to take a complexity view, though, the
  limit is in the number of computations. This should give rise to a
  sample complexity bound lower than that given by PAC learning (a PAC
  learning algorithm is computable). 