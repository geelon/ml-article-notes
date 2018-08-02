---
title: Learning Theory, an approximation viewpoint (Chapter 1)
date: 2018-07-28
---

Felipe Cucker & Ding Xuan Zhou, [Learning Theory: an approximation viewpoint](http://www.cambridge.org/gb/academic/subjects/computer-science/pattern-recognition-and-machine-learning/learning-theory-approximation-theory-viewpoint)

- [next chapter](./2007-LT-chapter-2.html): Basic Hypothesis Spaces

## Summary

This chapter introduces the standard statistical learning framework
and discusses a number of ways to decompose generalization error.

### Notation
In the following, we let:

- $X$ is a compact metric space, $Y = \mathbb{R}^k$
- $Z = X \times Y$ with Borel probability measure $\rho$

We also denote by $\rho(y|x)$ the *conditional probability measure* on 
$Y$. We also have $\rho_X$ as the *marginal probability measure* on
$X$. That is, if $\pi: X \times Y \to X$ is the projection, then
$$\begin{equation*}\rho_X(S) = \rho(\pi^{-1}(S)).\end{equation*}$$
Fubini's theorem states that if $\phi: X \times Y \to \mathbb{R}$ is
integrable, then:
$$\begin{equation*}
\int_{X \times Y} \phi(x,y) \ d\rho = \int_X \left(\int_Y \phi(x,y)
\ d\rho(y|x)\right) \ d\phi_X.\end{equation*}$$
To make notation more concise, we write:
$$\begin{align*}
\mathbb{E}_Z[\phi(x,y)] &:= \int_Z \phi(x,y) \ d\rho,\\
\mathbb{E}_Y[\phi(x,y)|x] &:= \int_Y \phi(x,y) \ d\rho(y|x),\\
\mathbb{E}_X[f(x)] &:= \int_X f(x) \ d\rho_X.
\end{align*}$$


### The Formal Setting

**Definition 1.** The *generalization error* (alsk called *error* or
  *risk*) of a function $f: X \to Y$ is defined as:
  $$\mathrm{err}(f) := \mathbb{E}_Z \left[\big(f(x) -
  y\big)^2\right].$$ 

The goal is to find the $f$ that minimizes $\mathrm{err}(f)$. To do
this, define $f_\rho:X \to Y$ to be the *regression function* or the
$L^2$ *regressor* as:
$$f_\rho(x) = \mathbb{E}_Y[y| x ].$$
The value $f_\rho(x)$ is just the average value of $y$ on $\{x\}
\times Y$. We assume that $f_\rho$ is bounded. This function is the
*Bayes estimator* in the sense that it minimizes $\mathrm{err}(f)$. In
fact, we can prove something stronger:

**Proposition 2.** For every $f: X \to Y$,
$$\mathrm{err}(f) = \mathbb{E}_X \left[ \big(f(x) -
f_\rho(x)\big)^2\right]  + \mathrm{err}(f_\rho).$$

*Proof.* We can decompose the error of $f$ as:
$$\begin{align*}
\mathrm{err}(f) &= \mathbb{E}_Z \left[\big(f(x) - f_\rho(x) +
f_\rho(x) - y\big)^2\right]\\
&= \mathbb{E}_Z \left[\big(f(x) - f_\rho(x)\big)^2 + 2 \big(f(x) -
f_\rho(x)\big)\big(f_\rho(x) - y\big) +\big(f_\rho(x) - y\big)^2\right].
\end{align*}$$
So we just need to show that the expectation of $2 \big(f(x) -
f_\rho(x)\big)\big(f_\rho(x) - y\big)$ is zero. Indeed, because $f(x)
- f_\rho(x)$ is constant in $Y$, Fubini's gives us:
$$\begin{equation*}\mathbb{E}_X\bigg[2\big(f(x) - f_\rho(x)\big) \cdot
\mathbb{E}_Y\left[ f_\rho(x) - y \ \big|\ x\right]\bigg] =
\mathbb{E}_X \big[2\big(f(x) - f_\rho(x)\big) \cdot 0\big] = 0.
\end{equation*}$$
This proves the desired equation.
<div align="right">☐</div>

Not only does this show that $f_\rho$ obtains is the best possible
classifier, we can rephrase the goal of learning as finding the best
possible approximator of $f_\rho$ using random samples from $Z$.

Let $\mathbf{z} = (z_1, \dotsc, z_m) \in Z^m$ be $m$ samples drawn
independently from $Z$ according to $\rho$. Then, we define the
*empirical mean* of a random variable $\xi$ on $Z$ by:
$$\hat{\mathbb{E}}_\mathbf{z} (\xi) := \frac{1}{m} \sum_{i=1}^m
\xi(z_i).$$
And so, we can define the *empirical error* or the *empirical risk* of
$f$ using the sample $\mathbf{z}$ as: 
$$\widehat{\mathrm{err}}_\mathbf{z}(f) :=
\hat{\mathbb{E}}_\mathbf{z}\left[\big(f(x) - y\big)^2\right] =
\frac{1}{m} \sum_{i=1}^m \big(f(x_i) - y_i\big)^2.$$

### Hypothesis Space

Learning occurs over a *hypothesis space*, a subset $\mathcal{H}
\subset \mathscr{C}(X)$ of the continuous functions on $X$. We work
with the $L^\infty$-norm,
$$\begin{equation*} ||f||_\infty = \sup_{x \in X}
|f(x)|. \end{equation*}$$
The hypothesis spaces studied in this book tends to be compact,
infinite-dimensional subsets of $\mathscr{C}(X)$. We also consider
closed balls in finite-dimensional subspaces and whole linear spaces.

The *target function* $f_\mathcal{H}$ is the function $f \in
\mathcal{H}$ that minimizes the error:
$$f_\mathcal{H} := \mathrm{arg\ min}_{f \in \mathcal{H}} \mathbb{E}_Z
\left[\big(f(x) - y \big)^2\right].$$
By Proposition 2, the optimizer of $\mathrm{err}(f)$ over $f \in
\mathcal{H}$, by Proposition 2, is also an optimizer of:
$$f_\mathcal{H} = \mathrm{arg\ min}_{f\in \mathcal{H}}
\mathbb{E}_X\left[\big(f - f_\rho\big)^2\right].$$
However, we cannot optimize over $\mathrm{err}(f)$ directly, since we
don't know the distribution $\rho$. But we can sample from the
distribution and optimize for the empirical error.

The *empirical target function* using a collection of random samples
$\mathbf{z} \in Z^m$ is similarly defined to be the optimizer of the
empirical error:
$$f_\mathbf{z} := f_{\mathcal{H},\mathbf{z}} := \mathrm{arg\ min}_{f
\in \mathcal{H}} \frac{1}{m} \sum_{i=1}^m \left[f(x_i) -
y_i\right]^2.$$
This function in $\mathcal{H}$ is also called the *empirical risk
minimizer* of $\mathcal{H}$.

We now need guarantees on the existence of the optimizers
$f_\mathcal{H}$ and $f_\mathbf{z}$ and bounds on how different the
empirical target function is from the actual target function. To show
existence, we first show that $\mathrm{err}$ and
$\widehat{\mathrm{err}}_\mathbf{z}$ are continuous on $\mathcal{H}$
with the $L^\infty$-norm. And by our assumption that $\mathcal{H}$ is
compact, optimizers exist.

**Lemma 3.** Let $f_1, f_2$ satisfy $|f_i(x) - y| \leq M$ almost
  everywhere in $Z$. Then both of:
  $$\begin{align*}
    \big|\ \mathrm{err}(f_1) - \mathrm{err}(f_2)\ \big| \quad \textrm{
  and }\quad    \big|\ \widehat{\mathrm{err}}_\mathbf{z}(f_1) -
  \widehat{\mathrm{err}}_\mathbf{z}(f_2)\ \big|
  \end{align*}$$
  are less than $2 M ||f_1 - f_2||_\infty$.

*Proof.* Notice that we can decompose $\big(f_1(x) - y\big)^2 -
 \big(f_2(x) - y \big)^2$ into:
 $$\begin{equation*}
 \big(f_1(x) + f_2(x) - 2y\big) \big(f_1(x) - f_2(x)\big).
 \end{equation*}$$
 This implies:
 $$\begin{align*}
   \left|\mathrm{err}(f_1) - \mathrm{err}(f_2)\right| &= \left| \int_Z
 \big(f_1(x) - y\big)^2 -  \big(f_2(x) - y \big)^2 \ d\rho \right| \\
 &=\left|\int_Z
 \big(f_1(x) + f_2(x) - 2y\big) \big(f_1(x) - f_2(x)\big) \
 d\rho\right|\\
 &\leq \int_Z \big|f_1(x) + f_2(x) - 2y\big| \cdot ||f_1 -
 f_2||_\infty \ d\rho\\
 &\leq 2M ||f_1 - f_2||_\infty.
 \end{align*}$$
 A similar calculation shows the same for the empirical error.
 <div align="right">☐</div>

Further note that the bound for the empirical error requires  $||
\cdot ||_\infty$ as a condition. In contrast, a weaker condition that
is say, distribution-dependent, may still give the same result for the
true error. In any case, this lemma implies that $\mathrm{err}$ and
$\widehat{\mathrm{err}}_\mathbf{z}$ are continuous.

### Sample, Approximation, and Generalization Errors

Given any function $f$, we saw earlier that $\mathrm{err}(f)$ is
always lower bounded by $\mathrm{err}(f_\rho)$. And so, we call their
difference the *excess generalization error*:
$$\mathrm{err}(f) - \mathrm{err}(f_\rho).$$

However, given a hypothesis class $\mathcal{H}$, the minimum
attainable error may be greater than $\mathrm{err}(f_\rho)$. In
particular, the error of any $f \in \mathcal{H}$ is lower bounded by
the best-in-class $f_\mathcal{H}$. We can say that our model
$\mathcal{H}$ will always incur an *approximation error*,
$$\mathrm{err}(\mathcal{H}) := \mathrm{err}(f_\mathcal{H}).$$

And so, given $f_\mathbf{z}$, we can decompose its error into a sum of
the *sample/estimation error* and the *approximation error*::
$$\mathrm{err}(f_\mathbf{z}) = \underbrace{\mathrm{err}(f_\mathbf{z})
-\mathrm{err}(f_\mathcal{H})}_{\textrm{estimation error}} +
\underbrace{\mathrm{err}(f_\mathcal{H})}_{\textrm{approximation error}}.$$ 
This decomposition highlights the **bias-variance problem**, described
as follows.

When our hypothesis space $\mathcal{H}$ is small, then the
approximation error may be very large; this contributes to the
*bias*. However, when $\mathcal{H}$ is large, then it may be possible
to attain a very low excess generalization error. However, the same
number of samples $\mathbf{z}$ may have large estimation error; there
is much more *variance* of the sample error. When bias is large, we
tend to *underfit* the sample data, while a large variance lets us
*overfit* sample data.

In the following chapters, we'll see that under certain condition of
$\rho$ and $\mathcal{H}$, the excess generalization error may approach
zero as the number samples tend to infinity. The sort of bounds $B(m,
\mathcal{H})$  we might hope for are:[^other]

- error bound: $\mathrm{err}(f_\mathbf{z}) \leq
  \widehat{\mathrm{err}}_\mathbf{z}(f_\mathbf{z}) + B(m, \mathcal{H})$
- error bound relative to best in class: $\mathrm{err}(f_\mathbf{z})
  \leq \mathrm{err}(f_\mathcal{H}) + B(m, \mathcal{H})$
- error bound relative to Bayes risk: $\mathrm{err}(f_\mathbf{z}) \leq
  \mathrm{err}(f_\rho) + B(m,\mathcal{H})$.


[^other]: this classification of error bounds is from (Bosquet
Boucheron Lugosi 2004).

## Further Reading

- [[Kearns Vazirani 1994](https://mitpress.mit.edu/books/introduction-computational-learning-theory)]
  An Introduction to Computational Learning Theory, a standard
  reference to PAC learning.
- [[Bosquet Boucheron Lugosi 2004](http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/pdfs/pdf2819.pdf)] Introduction to Statistical Learning Theory.