---
title: Nonconvex Optimization for Machine Learning (Chapter 5)
date: 2018-08-08
---

Prateek Jain & Purushottam Kar, [Non-convex Optimization for Machine Learning](https://arxiv.org/pdf/1712.07897.pdf)

- [previous chapter](./2017-nonconvex-chapter-4.html): Alternating Minimization
- current chapter: The EM Algorithm


## Summary

Denote by $f(\cdot | \theta)$ or $f_\theta(\cdot)$ a parametric
distribution over a domain $\mathcal{X}$, with parameter $\theta$. We
assume that i.i.d. samples $\mathbf{x}_1,\dotsc, \mathbf{x}_n$ are
sampled from an unknown distribution $f^*$. Assume that $f^*$ belongs
to a *parametric family* of distributions $\mathcal{F} = \{f_\theta :
\theta \in \Theta\}$.

The goal is to recover the true parameter $\theta^*$ using only the
samples. We can write the probability that we drew that data given a
if the distribution were parametrized by $\theta$:
$$f(\mathbf{x}_1,\dotsc, \mathbf{x}_n | \theta) = \prod_{i=1}^n
f(\mathbf{x}_i | \theta).$$
So, we can define the *likelihood* of a parameter $\theta$ given our
data as:
$$\mathcal{L}(\theta| \mathbf{x}_1,\dotsc, \mathbf{x}_n) =
f(\mathbf{x}_1,\dotsc, \mathbf{x}_n | \theta).$$
The *maximum likelihood estimate* (MLE) of $\theta^*$ is just the
parameter that maximizes $\mathcal{L}$:
$$\hat{\theta}_\mathrm{MLE} := \mathrm{arg\ max}_{\theta \in \Theta}
\mathcal{L}(\theta| \mathbf{x}_1,\dotsc, \mathbf{x}_n). $$
Other estimation techniques include *maximum a posteriori* (MAP),
which incoporates a prior distribution over $\theta$.

The probabilistic model becomes more challenging with the presence of
*latent variables*. For example, consider a statistical model that
generates two random variables, $Y \in \mathcal{Y}$ and $Z \in
\mathcal{Z}$, from a parametric family $\mathcal{F} = \{f_\theta =
f(\cdot, \cdot| \theta): \theta \in \Theta\}$. However, we only
observe $Y$ components. So, the corresponding marginal likelihood
function is then:
$$\mathcal{L}(\theta, \mathbf{y}_1, \dotsc, \mathbf{y}_n) =
\prod_{i=1}^n \sum_{\mathbf{z}_i \in \mathcal{Z}} f(\mathbf{y}_i,
\mathbf{z}_i|\theta).$$
Notice that when expanded, this is a sum of $|\mathcal{Z}|^n$ terms,
so is often NP-hard.

### Alternating Maximization

MLE with latent variables becomes intractable; however, if we had
them, then it would have been simple to find the MLE solution:
$$\hat{\theta}_\mathrm{MLE} = \mathrm{arg\ max}_{\theta \in \Theta}
\log \mathcal{L}(\theta; \{(\mathbf{y}_i, \mathbf{z}_i)\}_{i=1}^n).$$
On the other hand, if we knew $\theta^*$, we could also estimate the
latent variables:
$$\hat{\mathbf{z}}_i = \mathrm{arm\ max}_{\mathbf{z} \in \mathcal{Z}}
f(\mathbf{z} |\mathbf{y}_i ,\theta^*).$$
And so, we could apply a gAM-style algorithm to solve this MLE problem
with latent variables:

+------------------+--+--------------------------------------------------------------------------------------------+
| Algorithm 1      |  | AltMax for Latent Variable Models (AM-LVM)                                                 |
|                  |  |                                                                                            |
+=================:+==+:===========================================================================================+
|            Input |  | Data points $\mathbf{y}_1,\dotsc, \mathbf{y}_n$                                            |
+------------------+--+--------------------------------------------------------------------------------------------+
|           Output |  | An approximate MLE $\hat{\theta} \in \Theta$                                               |
+------------------+--+--------------------------------------------------------------------------------------------+
|                1 |  | $\theta^1 \leftarrow \mathsf{INITIALIZE}()$                                                |
+------------------+--+--------------------------------------------------------------------------------------------+
|                2 |  | **for** $t = 1, 2, \dotsc$  **do**                                                         |
+------------------+--+--------------------------------------------------------------------------------------------+
|                3 |  | $\quad$ **for $i = 1,2, \dotsc, n$ **do**                                                  |
+------------------+--+--------------------------------------------------------------------------------------------+
|                4 |  | $\quad \quad \hat{\mathbf{z}}_i^t \leftarrow                                               |
|                  |  | \mathrm{arg\ max}_{\mathbf{z}                                                              |
|                  |  | \in \mathcal{Z}} f(\mathbf{z} |\mathbf{y}_i,                                               |
|                  |  | \theta^t)$                                                                                 |
+------------------+--+--------------------------------------------------------------------------------------------+
|                5 |  | $\quad$ **end for**                                                                        |
+------------------+--+--------------------------------------------------------------------------------------------+
|                6 |  | $\quad \theta^{t+1} \leftarrow                                                             |
|                  |  | \mathrm{arg\ max}_{\theta \in \Theta}$                                                     | 
+------------------+--+--------------------------------------------------------------------------------------------+
|                7 |  | **end for**                                                                                |
+------------------+--+--------------------------------------------------------------------------------------------+
|                8 |  | **return** $\mathbf{w}^t$                                                                  |
+------------------+--+--------------------------------------------------------------------------------------------+

*At step 4, we estimate the latent variables, and at step 6, we update
 the parameters.*


At each time step $t$, AM-LVM makes a "hard assignment", each
datapoint $\mathbf{y}_i$ gets assigned to one $\mathbf{z}_i$. This is
a drawback, especially when $\mathcal{Z}$ is a large space; there may
be other values $\mathbf{z}' \in \mathcal{Z}$ such that
$f(\mathbf{z}'|\mathbf{y}_i, \theta^t)$ is also large neglected by
AM-LVM.

### Variational Lower Bound: prelude to EM algorithm

Let's consider a single sample $\mathbf{y}$. The log-likelihood is
then:
$$\begin{equation}
\log \mathcal{L}(\theta | \mathbf{y}) = \log f(\mathbf{y}|\theta) =
\log \sum_{\mathbf{z} \in \mathcal{Z}} f(\mathbf{y}, \mathbf{z} | \theta).
\end{equation}$$
It is hard to work with the log of a sum; however, Jensen's inequality
gives us a way to move the log past a sum. In doing so, we might be
able to give a lower bound on the log-likelihood. Recall that Jensen's
states that for any random variable $X$:  
$$\begin{equation}
\mathbb{E}\left[\log X\right] \leq \log \mathbb{E}[X]. \tag{Jensen's inequality}
\end{equation}$$
Indeed, notice that for any distribution $q(\mathbf{z})$ with support
over $\mathcal{Z}$, we have:
$$\begin{align*}
\log \sum_{\mathbf{z} \in\mathcal{Z}} f(\mathbf{y},\mathbf{z}|\theta)
= \log \mathbb{E}_{\mathbf{z} \sim q} \left[\frac{f(\mathbf{y},
\mathbf{z}|\theta)}{q(\mathbf{z})}\right].
\end{align*}$$
Thus, it is always true that:
$$\begin{align}
\log \mathcal{L}(\theta | \mathbf{y}) &\geq \mathbb{E}_{\mathbf{z}
\sim q} \left[\log f(\mathbf{y},\mathbf{z}|\theta) - \log
q(\mathbf{z})\right] \nonumber \\
&= \mathbb{E}_{\mathbf{z} \sim q} \left[\log
f(\mathbf{y},\mathbf{z}|\theta)\right] + H(\mathbf{z}). \label{elbo}
\end{align}$$
This value is called the *variational lower bound* or *evidence lower
bound* (ELBO), as it lower bounds the log likelihood. It follows that
to maximize the log likelihood, we can also attempt to maximize the
variational lower bound.

One question we should ask is how large is the gap between the log
likelihood and this lower bound. It turns out that this inequality is
actually tight. When $q(\mathbf{z}) = f(\mathbf{z}| \mathbf{y},
\theta)$, then we have: 
$$\begin{align*}
\log \mathbb{E}_{\mathbf{z} \sim f(\cdot | \mathbf{y} ,\theta)}
\left[\frac{f(\mathbf{y},
\mathbf{z}|\theta)}{f(\mathbf{z}|\mathbf{y},\theta)}\right] =
\underbrace{\log \mathbb{E}_{\mathbf{z} \sim
f(\cdot|\mathbf{y},\theta)} \big[f(\mathbf{y}|\theta)\big]}_{(*)} =
\log f(\mathbf{y}|\theta). 
\end{align*}$$
But what about for other distributions over the latent variable? To
obtain a bound, let's analyze the term $(*)$, where we can change the
expectation from the distribution over $f(\cdot | \mathbf{y},\theta)$
to any other distribution $q$, by:
$$\begin{align}
\log \mathbb{E}_{\mathbf{z} \sim
f(\cdot|\mathbf{y},\theta)} \big[f(\mathbf{y}|\theta)\big] &= \log
\mathbb{E}_{\mathbb{z} \sim q} \left[\frac{f(\mathbf{y}|\theta)
f(\mathbf{z}|\mathbf{y}, \theta)}{q(\mathbf{z})}\right]\nonumber \\
&\geq \mathbb{E}_{\mathbf{z} \sim q} \left[\log
f(\mathbf{y}|\theta)\right]  - \mathbb{E}_{\mathbf{z} \sim q}
\left[\log\frac{q(\mathbf{z})}{f(\mathbf{z}|\mathbf{y},
\theta)}\right] \nonumber\\
&= \log f(\mathbf{y}|\theta) - \mathrm{KL}\big(q(\mathbf{z})\
\big|\big|\ f(\mathbf{z}|\mathbf{y}, \theta)\big). 
\end{align}$$
This shows that the gap between the log-likelihood and the variational
lower bound using the distribution $q$ is the KL-divergence between
$q$ and the actual conditional distribution $f(\cdot | \mathbf{y},
\theta)$ of $\mathbf{z}$ given the observed data $\mathbf{y}$.

Now that we have a better understanding of the variational lower
bound, let's use it to derive the EM algorithm.


### EM Algorithm
Because the inequality in Equation $\ref{elbo}$ is tight when $q =
f(\cdot | y, \theta)$, this implies that optimizing the log-likelihood
$\mathcal{L}(\theta | \mathbf{y})$ is equivalent to the optimization
problem:
$$\begin{equation}
(q^*, \theta^*) = \mathrm{arg\ max}_{\substack{q \in \mathcal{Q}\\
\theta \in \Theta}} \ \mathbb{E}_{\mathbf{z} \sim q}[\log
f(\mathbf{y}, \mathbf{z} | \theta)] + H(\mathbf{z}),\label{objective}
\end{equation}$$
when $\mathcal{Q} = \{ f(\cdot | \mathbf{y}, \theta) : \theta \in
\Theta\}$ is this family of distributions over $\mathcal{Z}$. 

We can attempt to optimize this in an alternating-maximization-style
algorithm. Fixing $q \in \mathcal{Q}$ to be $f(\cdot |\mathbf{y},
\theta')$ for some (fixed) $\theta'$, we just need to optimze $\theta$
over the first term: 
$$\begin{equation*}
\hat{\theta} = \mathrm{arg\ max}_{\theta \in \Theta}
\mathbb{E}_{\mathbf{z} \sim q} [\log f(\mathbf{y},\mathbf{z} |
\theta)].
\end{equation*}$$
The objective here is called the *pointwise $Q$-function*
(depending on the point $\mathbf{y}$):
$$Q_\mathbf{y}(\theta | \theta') = \mathbb{E}_{\mathbf{z} \sim f(\cdot
| \mathbf{y}, \theta')} [\log f(\mathbf{y},\mathbf{z} | \theta)].$$
This new $\hat{\theta}$ obtained then gives us an updated distribution
for the latent variable, $f(\cdot | \mathbf{y}, \hat{\theta})$. The
construction of the objective function $Q$ is called the E-step, while
the optimization over $\theta$ is called the M-step. Together, they
make up the EM algorithm:

+------------------+--+--------------------------------------------------------------------------------------------+
| Algorithm 2      |  | Expectation Maximiation (EM)                                                               |
|                  |  |                                                                                            |
+=================:+==+:===========================================================================================+
|            Input |  | Implementations of the E-step $E(\cdot)$ and M-step $M(\cdot)$                             |
+------------------+--+--------------------------------------------------------------------------------------------+
|           Output |  | A good parameter $\hat{\theta} \in \Theta$                                                 |
+------------------+--+--------------------------------------------------------------------------------------------+
|                1 |  | $\theta^1 \leftarrow \mathsf{INITIALIZE}()$                                                |
+------------------+--+--------------------------------------------------------------------------------------------+
|                2 |  | **for** $t = 1, 2, \dotsc$  **do**                                                         |
+------------------+--+--------------------------------------------------------------------------------------------+
|                3 |  | $\quad$ $Q_t(\cdot | \theta^t) \leftarrow E(\theta^t)$                                     |
+------------------+--+--------------------------------------------------------------------------------------------+
|                4 |  | $\quad$ $\theta^{t+1} \leftarrow M(\theta^t, Q_t)$                                         |
+------------------+--+--------------------------------------------------------------------------------------------+
|                5 |  | **end for**                                                                                |
+------------------+--+--------------------------------------------------------------------------------------------+
|                6 |  | **return** $\theta^t$                                                                      |
+------------------+--+--------------------------------------------------------------------------------------------+



We haven't specified the E-step and M-step because there are a few
ways we can construct $E(\cdot)$ and $M(\cdot)$. If we have a single
sample, $\mathbf{y}$, then a natural way to construct the $Q$ function
from a parameter $\theta'$ in the E-step is just by returning
$Q_\mathbf{y}(\cdot | \theta')$.

**E-step Constructions.** For the purpose of theoretical analysis,
suppose we could use this pointwise $Q$-function to create the
*population $Q$-function*---if the true parameter is $\theta^*$, then
from the $\theta'$ we can generate: 
$$\begin{align*}
Q^\mathrm{pop}(\theta| \theta') &= \mathbb{E}_{\mathbf{y} \sim
f_{\theta^*}} Q_\mathbf{y} (\theta | \theta')\\
&= \sum_{\mathbf{y} \in \mathcal{Y}} \sum_{\mathbf{z} \in \mathcal{Z}}
f(\mathbf{y}|\theta^*) \cdot f(\mathbf{z}|\mathbf{y}, \theta') \log
f(\mathbf{y},\mathbf{z} | \theta). 
\end{align*}$$
Of course, this construction is infeasible. But we could construct the
*sample $Q$-function*, using $n$ data points $\mathbf{y}_1,\dotsc,
\mathbf{y}_n$, by:
$$\begin{align*}
Q^\mathrm{sam}(\theta | \theta') &= \frac{1}{n} \sum_{i=1}^n
Q_{\mathbf{y}_i} (\theta | \theta') \\
&= \frac{1}{n} \sum_{i=1}^n \sum_{\mathbf{z} \in \mathcal{Z}}
f(\mathbf{z} | \mathbf{y}, \theta') \log f(\mathbf{y}_i, \mathbf{z} |
\theta). 
\end{align*}$$
More generally, for any family of distributions $\mathcal{Q}$ with
support over $\mathbf{Z}$, given a sample of $n$ i.i.d. observed data,
we've ended up optimizing over the objective:
$$\frac{1}{n} \sum_{i=1}^n \sum_{\mathbf{z} \in \mathcal{Z}}
q(\mathbf{z}) \left[ \log f(\mathbf{y}_i, \mathbf{z}|\theta)\right].$$ 
This shows where the computational speed-up occurs. Instead of
calculating $f(\mathbf{z} | \mathbf{y}_i, \theta)$ for each $1 \leq i
\leq n$ and $\mathbf{z} \in \mathcal{Z}$, we just need to calculate
the values $q(\mathbf{z})$ once; these are reused for all $n$ data
points. Compared the $|\mathcal{Z}|^n$ terms we had before, we now
just have $n |\mathcal{Z}|$.

**M-step Constructions.** At the M-step, we can choose to perform a
  descent step instead of fully optimizing $\theta$. When the M-step
  returns the maximizer of the given $Q$-function, we say that this is
  the *fully corrective* version of EM:
  $$\begin{equation*}
  M^{\mathrm{fc}} (\theta', Q) = \mathrm{arg\ max}_{\theta \in \Theta}
  Q(\theta | \theta').
  \end{equation*}$$
  The *gradient descent* verion of EM performs a gradient step,
  $$\begin{equation*}
  M^{\mathrm{grad}}(\theta',Q) = \theta' + \alpha \cdot \nabla
  Q(\theta'|\theta').\end{equation*}$$

**Stochastic EM Construction.** The *stochastic update* version of EM
  uniformly chooses a sample $\mathbf{y}_i$ at random and performs a
  gradient step on the pointwise $Q$-function at $\mathbf{y}_i$. That
  is, the M-step is:
  $$\begin{equation*}
  M^{\mathrm{sto}}(\theta') = \theta' + \alpha \cdot \nabla
  Q_Y(\theta' | \theta'),
  \end{equation*}$$
  where $Y \sim \mathrm{Uniform}(\mathbf{y}_1,\dotsc, \mathbf{y}_n)$.



### Exercises

**Problem 1.** *[Exercise 5.1]* Show that for Gaussian mixture models
  with a balanced isotropic mixture, the AM-LVM algorithm exactly
  recovers Lloyd's algorithm for $k$-means clustering. Note that
  AM-LVM in this case prescribes setting $w_t^0(\mathbf{y}) = 1$ if
  $||\mathbf{y} - \mathbf{\mu}^{t,0}||_2 \leq ||\mathbf{y} -
  \mathbf{\mu}^{t,1}||_2$ and 0 otherwise, and also setting
  $w_t^1(\mathbf{y}) = 1 - w_t^0(\mathbf{y})$. 

## Discussion

Perhaps can also become more intuitive to describe log likelihood
using number of bits to represent data...

### Further Reading

 - [[Huber Ronchetti 2009](https://www.researchgate.net/file.PostFileLoader.html?id=563870c360614b09c78b456f&assetKey=AS%3A291621783523328%401446539459191)] *Robust Statistics*, on $M$-estimators
 - [[Blei Kucukelbir McAuliffe 2018](https://arxiv.org/pdf/1601.00670.pdf)] Variational Inference: A Review of Statisticians
- [[Yang 2017](http://legacydirs.umiacs.umd.edu/~xyang35/files/understanding-variational-lower.pdf)] Understanding the Variational Lower Bound


