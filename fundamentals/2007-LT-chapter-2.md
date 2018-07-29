---
title: Learning Theory, an approximation viewpoint (Chapter 2)
date: 2018-07-29
---

Felipe Cucker & Ding Xuan Zhou, [Learning Theory: an approximation viewpoint](http://www.cambridge.org/gb/academic/subjects/computer-science/pattern-recognition-and-machine-learning/learning-theory-approximation-theory-viewpoint)

- [previous chapter](./2007-LT-chapter-1.html): The Framework of Learning

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
 continuous.  ☐

**Definition 8.** The Hilbert space $\mathcal{H}_K$ in the previous
  theorem is called a *reproducing kernel Hilbert space* (RKHS), where
  property (3) is called the *reproducing property*.