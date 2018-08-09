---
title: Nonconvex Optimization for Machine Learning (Chapter 1)
date: 2018-08-05
---

Prateek Jain & Purushottam Kar, [Non-convex Optimization for Machine Learning](https://arxiv.org/pdf/1712.07897.pdf)

- current chapter: Introduction
- [next chapter](./2017-nonconvex-chapter-2.html): Mathematical Tools

## Summary

The generic form of an analytic optimization problem is:
$$\begin{align*}
\min_{\mathbf{x} \in\mathbb{R}^p} &\quad f(\mathbf{x})\\
\mathrm{s.t.} &\quad \mathbf{x} \in \mathcal{C},
\end{align*}$$
where $f$ is the *objective function* and $\mathcal{C}$ is the
*constraint set*. An optimization problem is convex only if both the
objective and constraint are convex.

**Example 1 (Sparse Regression).** Given a set of $n$
  covariate/response pairs $(\mathbf{x}_1, y_1), \dotsc,
  (\mathbf{x}_n, y_n)$, where $\mathbf{x}_i \in \mathbb{R}^p$ and $y_i
  \in \mathbb{R}$, the linear regression approach makes the modeling
  assumption
  $$\begin{align*}
  y_i = \mathbf{x}_i^\top \mathbf{w}^* + \eta_i,
  \end{align*}$$
  where $\mathbf{w}^* \in \mathbb{R}^p$ is the underlying linear model
  and $\eta_i$ is additive noise.

  Sometimes, we expect only a few of the $p$ features/covariates to be
  relevant. Other times, we are in a data-starved setting, with $n \ll
  p$. Standard statistical approaches require at least $n \geq p$ data
  points for consistency. The *sparse recovery* approach optimizes:
  $$\begin{align*}
  \hat{\mathbf{w}}_\mathrm{sp} = \mathrm{arg\ min}_{\mathbf{w}\in
  \mathbb{R}^p} & \quad \sum_{i=1}^n \left(y_i - \mathbf{x}_i^\top
  \mathbf{w}\right)^2 \\ 
  \mathrm{s.t.}& \quad \mathbf{w} \in \mathcal{B}_0(s),
  \end{align*}$$
  where $\mathcal{B}_0(s) = \{\mathbf{w} \in \mathbb{R}^p:
  ||\mathbf{w}||_0 \leq s\}$ is the set of vectors in $\mathbb{R}^p$ 
  with no more than $s$ non-zero entries. Spare recovery requires
  $\Omega(s\log p)$ data points, compared to the $\Omega(p)$. However,
  sparse recovery is NP-hard (Natarajan 1995).
  

**Example 2 (Recommender Systems).** Given a collection of $m$ users
  $u_1,\dotsc, u_m$ and $n$ items, $a_1,\dotsc, a_n$, let $A$ be the
  $m\times n$ preference matrix, where $A_{ij}$ is the $i$th user's
  preference for the $j$th item. However, there are only a small
  number of entries that are available; we would like to recover the
  remaining entries: a *matrix completion* problem. 

  It is a common to make the structural assumption that $A$ is
  low-rank. Then, let $\Omega \subset [m] \times [n]$, the set of
  observed entries of $A$. Then the low-rank matrix completion problem
  is:
  $$\begin{align*}
  \hat{A}_{\mathrm{lr}} = \mathrm{arg\ min}_{X \in \mathbb{R}^{m
  \times n}} & \quad \sum_{(i,j) \in \Omega} (X_{ij} - A_{ij})^2 \\
  \mathrm{s.t.} & \quad \mathrm{rank}(X) \leq r.
  \end{align*}$$
  Here, the constraint is also non-convex, and the problem is also
  NP-hard. This problem is equivalently written as:
  $$\begin{align*}
  \hat{A}_\mathrm{lr} = \mathrm{arg\ min}_{\substack{U \in
  \mathbb{R}^{m \times r} \\ V \in \mathbb{R}^{n\times r}}}
  \sum_{(i,j) \in \Omega} \left(U_i^\top V_j - A_{ij}\right)^2. 
  \end{align*}$$
  Here, there are no constraints, but the objective function over the
  pair $(U,V)$ is no longer convex.

### Approaches

One method to tackle nonconvex problems is through *relaxing* the
problem into a convex optimization problem. For example, spare
regression can be relaxed to lasso regression.

However, the solutions to the relaxed problem may be poor
approximations to the original problem. Sometimes, the *relaxation
gap* may be small or absent, with nice enough structure.

Other methods attempt to solve nonconvex optimization problems
directly, for example, *projected gradient descent*, *alternating
minimization*, the *expectation-maximization* algorithm, and
*stochastic optimization*.


## Exercises

**Problem 1.** *[Exercise 2.6]* Show that the set of sparse vectors
  $\mathcal{B}_0(s) \subset \mathbb{R}^p$ is non-convex for any $s < 
  p$. What happens when $s = p$?

*When $0 < s < p$, then $\mathcal{B}_0(s)$ contains the coordinate
 axes. The convex hull of the coordinate axes is the whole space
 $\mathbb{R}^p$, so it follows that $\mathcal{B}_0(s)$ for $0 < s < p$
 is not a convex set. When $s = p$, then $\mathcal{B}_0(s) =
 \mathbb{R}^p$. *  

**Problem 2.** *[Exercise 2.7]* Show that
  $\mathcal{B}_\mathrm{rank}(r) \subset \mathbb{R}^{n\times n}$, the
  set of $n \times n$ matrices with rank at most $r$, is non-convex
  for any $r < n$. What happens when $r = n$?

*This follows immediately from Problem 1, since
 $\mathcal{B}_0(1) \subset \mathcal{B}_\mathrm{rank}(r)$ for any $0 <
 r < p$. When $r = n$, then $\mathcal{B}_\mathrm{rank}(r) =
 \mathbb{R}^{n\times n}$. *

**Problem 3.** *[Exercise 3.3]* Show that the ratings matrix is at
  most rank $r$ iff for every user $i \in [m]$, there is an associated
  vector $\mathbf{u}_i \in\mathbb{R}^r$ describing that user, and with
  every item $j \in [n]$, there is a vector $\mathbf{v}_i \in
  \mathbb{R}^r$ such that $A_{ij} = \mathbf{u}_i^\top \mathbf{v}_j$.  

*If $A$ is a rank $r$ matrix, then there is a singular value
 decomposition of $A = U \Sigma V^\top$, with $U \in \mathbb{R}^{m
 \times m}$, $V \in \mathbb{R}^{n\times n}$ and  $\Sigma \in
 \mathbb{R}^{m \times n}$ is a diagonal where only top $r$ entries are
 nonzero. It follows that we can truncate the matrices, so that we're
 left with $\hat{U} \in \mathbb{R}^{m \times r}$, the first $r$
 columns of $U$, $\hat{\Sigma} \in \mathbb{R}^{r\times r}$ the top $r$
 rows and columns, and $\hat{V} \in \mathbb{R}^{n\times r}$, the first
 $r$ columns of $V$. Still, we have $A = \hat{U}
 \hat{\Sigma}\hat{V}^\top$. We may let $\mathbf{u}_i$ be the $i$th row
 of $\hat{U}$, and $\mathbf{v}_j$ be the $j$th column of
 $\hat{\Sigma}\hat{V}^\top$, so that $A_{ij} = \mathbf{u}_i^\top
 \mathbf{v}_j$.*

*Conversely, if such $\mathbf{u}_i$ and $\mathbf{v}_j$ exist, then let
 $U$ be the $m \times r$ matrix whose rows are $\mathbf{u}_i^top$ and
 $V$ be the $r \times n$ matrix whose columns are
 $\mathbf{v}_j$. The rank of either of these matrices are bounded
 above by $r$, so their product $A= UV$ is also at most rank $r$.*

**Problem 4.** *[Exercise 4.1]* Show that the objective in the
  low-rank matrix completion problem:
  $$\begin{align*}
  \hat{A}_\mathrm{lr} = \mathrm{arg\ min}_{\substack{U \in
  \mathbb{R}^{m \times r} \\ V \in \mathbb{R}^{n\times r}}}
  \sum_{(i,j) \in \Omega} \left(U_i^\top V_j - A_{ij}\right)^2, 
  \end{align*}$$
  is not jointly convex in $U$ and $V$. Show also that the objective
  is marginally convex in both variables.

*For simplicity, we'll show that the problem is not jointly convex in
 $U$ and $V$ in the special case of $m = n = 1$  and $A = 0$. The same
 argument easily extends to arbitrary $m, n > 0$ and $A \in
 \mathbb{R}^{m\times n}$. Here, the objective simply becomes:*
 $$\begin{align*}
 \hat{A}_\mathrm{lr} = \mathrm{arg\ min}_{\mathbf{u}, \mathbf{v} \in
 \mathbb{R}^m} & \quad \left(\mathbf{u}^\top \mathbf{v}\right)^2
 \end{align*}$$
 *Let* $\mathbf{e}_1 \in \mathbb{R}^m$ *be the first basis
 vector. Consider two cases:*
 $$\begin{align*}
 (\mathbf{u},\mathbf{v}) &= (2\mathbf{e}_1, 0)\\
 (\mathbf{u}',\mathbf{v}') &= (0, 2\mathbf{e}_1).
 \end{align*}$$
 *In both cases, their squared dot product is zero. However, their
 convex combination:*
 $$\begin{equation*}
 \frac{1}{2} (\mathbf{u}, \mathbf{v}) + \frac{1}{2}
 (\mathbf{u}',\mathbf{v}') = (\mathbf{e}_1,\mathbf{e}_1)
 \end{equation*}$$
 *has positive squared dot product. Thus, the objective is not
 convex.*

 *In contrast, the objective is marginally convex: fixing $V$, the
  function $\left(U_i^\top V_j - A_{ij}\right)^2$ is convex in
  $U_i$. So, the sum over all $i,j$ is still convex in $U$.*
 
 
## Discussion

**Note 1.** This section states that "it turns out that problem
structures that allow nonconvex approaches to avoid NP-hardness
results are very similar to those that allow their convex relaxation
counterparts to avoid distortions and a large relaxation gap". Should
understand what this means more.


### Further Reading
- [[Natarajan 1995](https://epubs.siam.org/doi/pdf/10.1137/S0097539792240406)] Sparse Approximation Solutions to Linear Systems.
