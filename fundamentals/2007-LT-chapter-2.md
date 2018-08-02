---
title: Learning Theory, an approximation viewpoint (Chapter 2)
date: 2018-07-30
---

Felipe Cucker & Ding Xuan Zhou, [Learning Theory: an approximation viewpoint](http://www.cambridge.org/gb/academic/subjects/computer-science/pattern-recognition-and-machine-learning/learning-theory-approximation-theory-viewpoint)

- [previous chapter](./2007-LT-chapter-1.html): The Framework of Learning
- [next chapter](./2007-LT-chapter-3.html): Estimating the Sample Error

## Summary

### Review of Mathematics

**Review of Fourier transforms.** The *Fourier transform*
  $\mathcal{F}: L^1(\mathbb{R}^n) \to L^1(\mathbb{R}^n)$ is defined
  by: $$\mathcal{F}(f)(w) = \int_{\mathbb{R}^n} e^{-iw \cdot x} f(x) \
  dx.$$ 
  The Fourier transform on $L^1(\mathbb{R}^n)$ satisfies:
  $$\mathcal{F}(f * g) = \mathcal{F}(f) \mathcal{F}(g),$$
  where $f * g$ is the *convolution* of $f$ and $g$:
  $$(f * g)(x) = \int_{\mathbb{R}^n} f(x - u) g(u) \ du.$$

  The Fourier transform extends to $L^2(\mathbb{R}^n)$, though  with a
  bit more work, for if we take some $f \in L^2(\mathbb{R}^n)$, the
  same integrand as defined above may not be integrable.  However, the
  space $\mathscr{C}_0$ of continuous functions with compact support
  is contained in $L^1(\mathbb{R}^n)$ and is dense in
  $L^2(\mathbb{R}^n)$. It turns out that the Fourier transform of any
  Cauchy sequence in $\mathscr{C}_0$ converging to $f \in
  L^2(\mathbb{R}^n)$ is itself a Cauchy sequence convering to some
  element in $L^2(\mathbb{R}^n)$. We define that element as the
  Fourier transform of $f$, denoted as $\hat{f}$ or $\mathcal{F}(f)$.

**Theorem 1 (Plancherel's theorem).** For $f \in L^2(\mathbb{R}^n)$,

1. $\mathcal{F}(f)(w) = \lim_{k \to \infty} \int_{[-k,k]^n} e^{-iw
\cdot w} f(x) \ dx$, with convergence in $L^2$.
2. $||\mathcal{F}(f)|| = (2\pi)^{n/2} ||f||.$
3. $f(x) = \lim_{k \to \infty} \frac{1}{(2\pi)^n} \int_{[-k,k]^n}
e^{iw\cdot x} \mathcal{F}(f)(w) \ dw$, with convergence in $L^2$.
4. $\mathcal{F}: L^2(\mathbb{R}^n) \to L^2(\mathbb{R}^n)$ is an
isomorphism of Hilbert spaces.

**Review of Functional Analysis.** A subset $S \subset \mathscr{C}(X)$
  is *equicontinuous at* $x\in X$ if for all $\epsilon > 0$, there
  exists a neighborhood $V$ of $x$ such that $|f(x) - f(y)| <
  \epsilon$ for all $y \in V$. The set $S$ is *equicontinuous* if this
  holds for all $x \in S$.

**Theorem 2 (Arzelá-Ascoli Theorem).** Let $X$ be compact and $S
  \subset \mathscr{C}(X)$. Then $S$ is compact if and only if $S$ is
  closed, bounded, and equicontinuous.

**Remark 3.** Closed balls $B$ in a (infinite-dimensional) Hilbert
  space are not compact, though they are *weakly compact*. That is,
  every sequence $\{f_n\}_{n \in \mathbb{N}}$ in a closed ball $B$ has
  a weakly convergent subsequence $\{f_{n_k}\}_{k \in \mathbb{N}}$,
  where there exists some $\tilde{f} \in B$ such that:
  $$\lim_{k \to\infty} \langle f_{n_k}, g\rangle = \langle \tilde{f},
  g\rangle$$
  for all $g \in \mathcal{H}$.

**Definition 4.** Let $L :\mathbb{E} \to \mathbb{F}$ be a linear map
  between the Banach spaces $\mathbb{E}$ and $\mathbb{F}$. We say that
  $L$ is *bounded* if the image of the unit ball in $\mathbb{E}$ is
  bounded in $\mathbb{F}$. The *operator norm* is defined as:
  $$||L||_\mathrm{op} := \sup_{||x|| = 1} ||Lx||.$$
  We say that $L$ is *compact* if for all bounded sets $B \subset
  \mathbb{E}$, the closure $\overline{L(B)}$ is compact.

**Review of Completely Monotonic Functions.** We say that a function
  $f: [0,\infty) \to \mathbb{R}$ is *completely monotonic* if it is
  continuous on $[0,\infty)$ and $C^\infty$ on $(0,\infty)$, and for
  all $r > 0$, $k \geq 0$, $(-1)^k f^{(k)}(r) \geq 0$. These functions
  are characterized by:

**Proposition 5.** A function $f: [0,\infty) \to \mathbb{R}$ is
  completely monotonic if and only if for all $t \in (0,\infty)$,
  $$f(t) = \int_0^\infty e^{-t\sigma} \ d \nu(\sigma),$$
  where $\nu$ is a finite Borel measure on $[0,\infty)$.

### Reproducing Kernel Hilbert Spaces

**Definition 6.** Let $X$ be a metric space. A map $K : X \times X \to
  \mathbb{R}$ is *symmetric* if $K(x,t) = K(t,x)$ for all $x,t \in
  X$. It is *positive semidefinite* if for all finite sets
  $\mathbf{x} = \{x_1,\dotsc, x_n\}$, the $n \times n$ matrix
  $K_\mathbf{x}$ whose $ij$th entry is  $K(x_i, x_j)$ is positive
  semidefinite. We say that $K$ is a *Mercer kernel* if it is
  continuous, symmetric, and positive semidefinite. We call
  $K_\mathbf{x}$ the *Gramian* of $K$ at $\mathbf{x}$.

Fix $X$ a compact metric space and $K$ a Mercer kernel. Because $K$ is
positive semidefinite, $K(x,x) \geq 0$. We can define:
$$\mathbf{C}_K := \sup_{x \in X} \sqrt{K(x,x)}.$$
In analogy to Cauchy-Schwarz, the following holds:
$$K(x,t)^2 \leq K(x,x) K(t,t).$$
To see this, just note that the Gramian at $\{x, t\}$ is a $2\times 2$ 
matrix whose determinant must be nonnegative due to positive
semidefiniteness. It follows from this that:
$$\begin{equation*}
\mathbf{C}_K = \sup_{x,t \in X} \sqrt{|K(x,t)|}.
\end{equation*}$$
We also define for each $x \in X$, the function $K_x: X \to
\mathbb{R}$ by $K_x: t \mapsto K(x,t)$.

**Theorem 7.** There exists a unique Hilbert space $\mathcal{H}_K$ of
  functions on $X$ satisfying the following conditions:

  1. for all $x \in X$, $K_x \in \mathcal{H}_K$,
  2. the span of $\{K_x : x \in X\}$ is dense in $\mathcal{H}_K$,
  3. for all $f \in \mathcal{H}_K$ and $x \in X$, $f(x) = \langle K_x,
  f\rangle_{\mathcal{H}_K}$.

  Moreover, $\mathcal{H}_K$ consists of continuous functions and the
  inclusion $I_K: \mathcal{H}_K \to \mathscr{C}(X)$ is bounded with
  $||I_K|| \leq \mathbf{C}_K$.

*Proof.* Let $H_0$ be the span of $\{K_x : x \in X\}$. Define the
 inner product on the generators by $\langle K_x, K_t\rangle =
 K(x,t)$, and extend by linearity. Because $K$ is symmetric and
 positive semidefinite, this defines an inner product on $H_0$.

 Define $\mathcal{H}_K$ as the completion of $H_0$ with the norm
 induced by this inner product, and this $\mathcal{H}_K$ satisfies the
 three above conditions. By construction, $\mathcal{H}_K$ is unique.

 To see that elements of $\mathcal{H}_K$ are continuous, consider:
 $$|f(x)| = |\langle K_x, f\rangle_{\mathcal{H}_K}| \leq
 ||f||_{\mathcal{H}_K} \sqrt{K(x,x)}.$$
 This implies $$||f||_\mathrm{\infty}  \leq \mathbf{C}_K\cdot
 ||f||_{\mathcal{H}_K}.$$
 This shows that $||I_K||_\mathrm{op} \leq \mathbf{C}_K$ and so,
 convergence in $||\cdot ||_{\mathcal{H}_K}$ implies convergence in
 $||\cdot ||_\infty$. And as $f \in \mathcal{H}_K$ is the limit of
 linear combinations of continuous functions $K_x$, $f$ is
 continuous.
 <div align="right">☐</div>

**Definition 8.** The Hilbert space $\mathcal{H}_K$ in the previous
  theorem is called a *reproducing kernel Hilbert space* (RKHS), where
  property (3) is called the *reproducing property*.

### Examples Mercer Kernels

**Example 9.** Let $X = \{x \in \mathbb{R}^n : ||x|| \leq R\}$. The
  *dot product kernel* is a function $K: X \times X \to \mathbb{R}$
  with:
  $$K(x,y) = \sum_{i=1}^\infty a_d (x \cdot y)^d,$$
  where $a_d \geq 0$ and $\sum a_d R^{2d} < \infty$.

**Example 10.** If $k$ is an even function on $\mathbb{R}^n$, so that
  $k(-x) = k(x)$, then $K(x,y) = k(x - y)$ is a *translation invariant
  kernel*. The Fourier transform $\hat{k}$ is *nonnegative*
  (respectively, *positive*) if it is real-valued and $\hat{k}(\xi)
  \geq 0$ (respectively, $\hat{k}(\xi) > 0$) for all $\xi \in
  \mathbb{R}^n$. 

**Proposition 11.** Let $k \in L^2(\mathbb{R}^n)$ be continuous and
  even. If the Fourier transform of $k$ is nonnegative, then the
  kernel $K(x,y) = k(x - y)$ is a Mercer kernel on $\mathbb{R}^n$.

**Proposition 12.** Let $f: [0,\infty) \to \mathbb{R}$ be a completely
  monotonic function. Then $K(x,y) = f(||x - y||^2)$ is positive
  definite.

**Corollary 13.** Let $c,\alpha > 0$. Then the following are Mercer
  kernels:
  
1. (Gaussian) $K(x,t) = \mathrm{exp}\left(-||x - t||^2/c^2\right)$.
2. (Inverse multiquadrics) $K(x,t) = (c^2 + ||x - 1||^2)^{-\alpha}$.


### Hypothesis space associated with an RKHS

Here, we show that if $K$ is a Mercer kernel, and $X$ is a compact
metric space, then  closed balls of $\mathcal{H}_K$ are compact in
$\mathscr{C}(X)$. First, we need that it is closed:

**Lemma 14.** Let $K$ be a Mercer kernel on a compact metric
  space $X$, and $\mathcal{H}_K$ be its RKHS. For all $R > 0$, the
  ball $B_R := \{f \in \mathcal{H}_K : ||f||_K \leq R\}$ is a closed
  subset of $\mathscr{C}(X)$.

*Proof.* Let $\{f_n\} \subset B_R$ converge to some $f \in
 \mathscr{C}(X)$. Since $B_R$ is weakly compact, there is a
 subsequence $\{f_{n_k}\}$ such that:
 $$\lim_{k \to \infty} \langle f_{n_k}, g\rangle_K = \langle
 \tilde{f}, g\rangle_K$$
 for all $g \in \mathcal{H}_K$. In particular, letting $g = K_x$ for
 all $x \in X$ shows that $\tilde{f}(x) = f(x)$ for all $x$; thus,
 $f$ is in $B_R$.
 <div align="right">☐</div>

Now, we show that $\mathcal{H}_K$ is equicontinuous.

**Lemma 15.** Let $K$ be a Mercer kernel on a compact metric
  space $X$, and $\mathcal{H}_K$ be its RKHS. For all $R > 0$, the
  inclusion $I_K(B_R)$ is equicontinuous.

*Proof.* Since $K: X \times X \to \mathbb{R}$ is continuous, and $X
 \times X$ is compact, $K$ is uniformly continuous. In particular, for
 all $\epsilon > 0$, there exists $\delta > 0$ such that for all $x
 \in X$, if $y, y' \in X$ where $d(y, y') \leq \delta$:
 $$\begin{align*}
   |K(x,y) - K(x,y')| < \epsilon.
 \end{align*}$$
 Let $f \in B_R$, and $y,y'$ as before. Then:
 $$\begin{align*}
   |f(y) - f(y')| \leq |\langle f, K_y - K_{y'}\rangle_K| \leq ||f||_K
 ||K_y - K_{y'}||_K \leq R \sqrt{ 2 \epsilon}.
 \end{align*}$$
 <div align="right">☐</div>

Arzelá-Ascoli now implies:

**Proposition 16.** Let $K$ be a Mercer kernel on a compact metric
space $X$, and $\mathcal{H}_K$ be its RKHS. Then closed balls $B_R$ 
in $\mathcal{H}_K$ are compact in $\mathscr{C}(X)$.
<div align="right">☐</div>

And so, these are compact metric hypothesis spaces.

**Example 17.** We'll consider more deeply a specific dot product
  kernel on $\mathbb{R}^{n+1}$:
  $$K(x,y) = \langle x,y\rangle^d.$$
  First, we need some notation. Let
  $\alpha = (a_0, \dotsc, a_n)$ be nonnegative integers such that
  $|\alpha| := \sum \alpha_i = d$. These are multi-indices, where
  given an $x \in S^n \subset \mathbb{R}^{n+1}$ in the unit
  $n$-sphere, we let $x^\alpha$ denote the value
  $x_0^{\alpha_0}\dotsm x_n^{\alpha_n}$. We also define the
  multinomial coefficients as:   
  $$C_\alpha^d = \frac{d!}{\alpha_1! \dotsm \alpha_n!},$$
  so that if $x, y \in \mathbb{R}^{n+1}$, then:
  $$\begin{align*}
     \langle x, y\rangle^d &= \sum_{|\alpha| = d} C_\alpha^d x^\alpha
  y^\alpha \\
  &= \sum_{|\alpha| = d} (C_\alpha^d)^{1/2} x^\alpha
  (C_\alpha^d)^{1/2} y^\alpha.
  \end{align*}$$
  The first line suggests that $K_x \in \mathbb{R}[t_0,\dotsc, t_n]_d$
  is a homogeneous polynomial of degree $d$ that evaluates points $t
  \in \mathbb{R}^{n+1}$ via:
  $$K_x(t) = \sum_{|\alpha| = d} C_\alpha^d x^\alpha t^\alpha.$$
  We'll denote the linear space of homogeneous polynomials of degree
  $d$ by $\mathcal{H}_d$. The second line is written suggestively as
  the form of a dot product inside a space of dimension
  $$\begin{equation*}
     N = \binom{n + d}{n},
  \end{equation*}$$
  a dimension for each possible $\alpha$ where $|\alpha|= d$ (see
  Stars and Bars theorem). In particular, if we let $\Phi:
  \mathbb{R}^{n+1} \to \mathbb{R}^N$ be defined by:
  $$\Phi(x) = \left(x^\alpha (C_\alpha^d)^{1/2}\right)_{|\alpha|=d},$$ 
  then we obtain $K(x,y) = \langle \Phi(x), \Phi(y)\rangle.$

  Together, this shows that $\mathcal{H}_K$ is contained in the
  subspace of homogeneous polynomials of degree $d$ where the inner
  product must satisfy:
  $$\begin{align*}
  \langle K_x, K_y\rangle_{\mathcal{H}_K} &=
  \left\langle \sum (C_\alpha^d)^{1/2} x^\alpha t^\alpha ,
  \sum (C_\alpha^d)^{1/2} y^\alpha t^\alpha \right\rangle_{\mathcal{H}_K} \\
  &= \sum_{|\alpha| = d} C_\alpha^d x^\alpha y^\alpha.
  \end{align*}$$
  It follows that we can define an inner product structure on
  $\mathcal{H}_d$ as follows so that the inner product on
  $\mathcal{H}_K$ coincides with the inner product on $\mathcal{H}_d$:
  if $f = \sum w_\alpha t^\alpha$ and $g = \sum v_\alpha t^\alpha$ are
  in $\mathcal{H}_d$, then:
  $$\langle f, g\rangle_{\mathcal{H}_d} = \sum_{|\alpha| = d}
  (C^d_{\alpha})^{-1} w_\alpha v_\alpha.$$
  This inner product structure on $\mathcal{H}_d$ is called the *Weyl
  inner product*. At this point, we have $\mathcal{H}_K \subset
  \mathcal{H}_d$ as inner product spaces.

  On the other hand, we can write the Veronese embedding $\mathcal{V}: 
  \mathbb{R}^{n+1} \to \mathbb{R}^N$ as:
  $$\mathcal{V}(x) = \left(x^\alpha\right)_{|\alpha| = d}.$$
  Thus, we can decompose:
  $\Phi = \mathrm{diag}\left((C_\alpha^d)^{1/2}\right)\mathcal{V}$.
  From algebraic geometry, we know that the Veronese variety spans
  $\mathcal{H}_d$, so this implies $\mathcal{H}_d = \mathcal{H}_K$. We
  have just shown:

**Proposition 18.** Let $X = S^n$ be the $n$-sphere in
  $\mathbb{R}^{n+1}$ and $K(x,y) = \langle x,y\rangle^d$ be a kernel
  on $X$. Then, $\mathcal{H}_K = \mathcal{H}_d$, where $\mathcal{H}_d$
  is the inner product space of homogenous polynomials in
  $\mathbb{R}[t_0,\dotsc, t_n]_d$ of degree $d$ given the Weyl inner
  product structure.

**Fact 19.** The Weyl inner product also has an invariance
  property. If $O_{n+1}$ is the orthogonal group on
  $\mathbb{R}^{n+1}$, then its action induces an action on
  $\mathcal{H}_d$ where $\sigma f(x) := f(\sigma^{-1} x)$ for $f \in
  \mathcal{H}_d$. The Weyl inner product $\langle \cdot ,\cdot
  \rangle_W$ is invariant to this action:
  $$\langle \sigma f, \sigma g\rangle_W = \langle f, g\rangle_W.$$

### Computing the Empirical Target Function

**Review of optimization.** The *general nonlinear programming
problem* is finding $x \in \mathbb{R}^n$ that optimizes:
$$\begin{align}
\min \ & \ f(x)\nonumber \\
\textrm{s.t. } & \ g_i(x) \leq 0, \quad i = 1,\dotsc, m, \\
& \ h_j(x) = 0, \quad j = 1,\dotsc, p, \nonumber
\end{align}$$
where $f, g_i, h_j : \mathbb{R}^n \to \mathbb{R}$. The function $f$ is
the *objective function*, and $g_i$ and $h_j$ are inequality/equality
constraints, respectively. Any point satisfying the constraints are
*feasible*, comprising the *feasible set*. The following types of
optimization problems have efficient algorithms:


- *linear programming:* objective and constraints are linear.
- *convex programming:* $f$ and $g_i$ are convex and $h_j$ are linear.
- *convex quadratic programming:* convex programming problem where $f$
  and $g_i$ are quadratic. 


If $\mathcal{H} = I_K(B_R)$ as in Proposition 16, then computing the
empirical target function $f_\mathbf{z}$ is a convex optimization
problem.

Let $\mathcal{H}_{K,\mathbf{z}}$ be the subspace of $\mathcal{H}_K$
spanned by $\{K_{x_1}, \dotsc, K_{x_m}\}$. Let $\Pi: \mathcal{H}_K \to
\mathcal{H}_{K,\mathbf{z}}$ be the orthogonal projection onto the
subspace. Then, the following shows that a minimizer of the empirical
risk is contained in the span of $K_{x_i}$:

**Proposition 20.** Let $B \subset \mathcal{H}_K$. If $f \in
  \mathcal{H}_K$ is a minimizer of $\widehat{\mathrm{err}}_\mathbf{z}$
  over $B$, then $\Pi f$ minimizes $\widehat{\mathrm{err}}_\mathbf{z}$
  over $\Pi (B)$.

*Proof.* If $f \in \mathcal{H}_K$ is a minimizer of empirical error,
then $f$ decomposes into the sum of orthogonal components---the
component $\Pi f$ contained in the span of $K_{x_i}$ and those
perpendicular. And since:
$$f(x_i) = \langle f, K_{x_i}\rangle,$$
the orthogonal component has no contribution, implying
$$\Pi f(x_i) = f(x_i).$$
It follows $\Pi f$ also minimizes the empirical error.
<div align="right">☐</div>

So, when $B = B_R$, a solution will be of the form $f_\mathbf{z} =
\sum_{i=1}^m c_i^* K_{x_i}$, minimizing:
$$\begin{align*}
\min \ & \ \frac{1}{m} \sum_{j=1}^m \left(\sum_{i=1}^m c_i K(x_i,x_j)
- y_j\right)^2\\
\textrm{s.t. }& \ c^T K_\mathbf{z} c \leq R^2.
\end{align*}$$



## Discussion

**Question 1.** Let $K$ be the Gram matrix for data points $\mathcal{X} =
  \{x_1,\dotsc, x_n\}$. Then, there exists some embedding $\Phi:
  \mathcal{X} \to \mathbb{R}^n$ such that $K_{ij} = \langle
  \Phi(x_i),\Phi(x_j)\rangle$. Then, $a^2 K$ corresponds to the Gram
  matrix to the embedding $a\Phi : \mathcal{X} \to \mathbb{R}^n$.

  Let $K_1$ and $K_2$ be two Gram matrices for $\mathcal{X}$. Then,
  there are two embeddings $\Phi_1 : \mathcal{X} \to \mathbb{R}^n$ and
  $\Phi_2 : \mathcal{X} \to \mathbb{R}^n$ as above. Let $\iota_i :
  \mathbb{R}^n \to \mathbb{R}^n \oplus \mathbb{R}^n$ be the natural
  inclusion maps into the first and second terms of the direct sum. 

  We should be able to give an intepretation to the sum of the kernels
  $a^2 K_1 + b^2 K_2$ as the kernel generated by the map $\Phi :
  \mathcal{X} \to \mathbb{R}^{2n}$, given by:
  $$\Phi := \sqrt{a^2 + b^2} \cdot \left(\frac{a}{\sqrt{a^2 + b^2}}
  \iota_1 \circ \Phi_1 + \frac{b}{\sqrt{a^2+b^2}} \iota_2 \circ
  \Phi_2\right).$$

  Does this imply that if points become linearly separable using
  $K_1$, then they remain linearly separable under $a^2 K_1 + b^2
  K_2$ when $a^2 > 0$?

**Question 2.** Can we give a similar geometric interpretation of the
  (pointwise) product of two kernels, $K_1 \cdot K_2$?

## Further Reading

- [[Smola Schölkopf Herbricht 2001](./)] A generalized representer
  theorem. This has more general forms of these theorems.
- [[Smola Schölkopf Müller 1998](./)] The connection between
  regularization operators and support vector kernels.