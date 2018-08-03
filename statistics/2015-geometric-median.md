---
title: Geometric median and robust estimation in Banach spaces
date: 2018-08-01
---

Minsker, Stanislav, et al. "Scalable and robust Bayesian inference via
the median posterior." International conference on machine
learning. 2014. 


## Summary

Given $n$ points in a Banach space, any point that is close to a
constant fraction of those points must also be close to the geometric
median of those points (this is formally stated below). Using this
fact, suppose we have $n$ empirical estimates $\hat{\mu}_1,\dotsc,
\hat{\mu}_n$, and that we know that with high probability, each of
these are within some distance of the true value. Then, their median
will also be within some distance of the true value with high
probability. The median of these means has a tighter concentration
bound around the true parameter. 

### Problem Introduction
Let $\mu = \mathbb{E}X$ be the mean for a real-valued random variable
with bounded variance. Using i.i.d. samples $X_1,\dotsc, S_n$, can we
construct and estimator $\hat{\mu}$ of the mean such that for all $t >
0$:
$$\mathrm{Pr}\big[|\hat{\mu} - \mu| > t\big] \leq
\mathrm{exp}\left(- \frac{C n^2 t^2}{\mathrm{Var}(X)}\right),$$ 
where $C$ is some absolute constant, without any addition assumption
on the distribution? 

It turns out that this is possible, as seen, for example in [Oliveira
Lerasle 2011], where the sample is split into $V \approx t$ blocks,
the sample mean is computed on each of those blocks, and the median of
the means is the new empirical estimator. Similar ideas were used in
[Alon Matias Szegedy 1996] to approximate frequency moments of a
sequence and [Bubeck Cesa-Bianchi Lugosi 2013] in multi-armed bandits.
Another approach in [Cantoni 2012] used PAC-Bayesian truncation.

This paper extends the *median of the means* to general Banach
spaces.

### Geometric Median

If $\mathbb{X}$ is a Banach space with norm $|| \cdot ||$ and
probability measure $\mu$, the *geometric median* (or *Fermat-Weber
point*) is any $x_*$ that satisfies:[^other]
$$x_* = \mathrm{arg\ min}_{y \in \mathbb{X}} \int_{\mathbb{X}} (||y -
x|| - ||x||) \ \mu (\mathrm{d}x).$$
A sufficient condition on $\mathbb{X}$ for the existence of a
geometric median:$\mathbb{X}$ is separable and reflexive. We'll assume
this from now on.

The particular geometric median we're interested in corresponds to the
empirical measure on a collection $x_1, \dotsc, x_k \in
\mathbb{X}$, where:
$$x_* = \mathrm{med}(x_1,\dotsc, x_k) := \mathrm{arg\ min}_{y \in
\mathbb{X}} \sum_{j=1}^k ||y - x_j||.$$
If $\mathbb{X}$ is strictly convex, then $x_*$ is unique unless $x_1,
\dotsc, x_n$ are collinear. 

The geometric result we need is:

**Lemma 1.** Let $x_1, \dotsc, x_k \in \mathbb{X}$ and let $x_*$ be
  their geometric median. Let $\alpha \in (0, \frac{1}{2})$ and $r >
  0$. There is a constant $C_\alpha$ such that whenever a point $z \in
  \mathbb{X}$ is far away from the geometric median:
  $$||x_* - z|| > C_\alpha r,$$
  then it must be far from an $\alpha$-fraction of $x_1,\dotsc, x_k$:
  $$||x_j - z|| > r$$
  for $j \in J \subset [k]$ where $|J| > \alpha k$. Respectively,
  when $\mathbb{X}$ is a general Banach space or a Hilbert space:
  $$C_\alpha = \frac{2(1 - \alpha)}{1 - 2\alpha} \quad \quad C_\alpha
  = (1 - \alpha) \sqrt{\frac{1}{1 - 2 \alpha}}.$$
<div align="right">☐</div>

*Note:* in the Banach space case, $C_\alpha \to 2$ as $\alpha \to 0$,
 while in the Hilbert space case, $C_\alpha \to 1$ as $\alpha \to 0$.

[^other]: See [Small 1990] for a survey of other multidimensional
medians. 

### Boosting the Confidence

**Theorem 2.** Assume that $\mu \in \mathbb{X}$ is a parameter of
  interest and $\hat{\mu}_1,\dotsc, \hat{\mu}_k \in \mathbb{X}$ are a
  collection of independent estimators of $\mu$. Fix $\alpha \in
  (0,\frac{1}{2})$. Let $0 < p < \alpha$ and $\epsilon > 0$. If for
  all estimators $\hat{\mu}_j$,
  $$\mathrm{Pr}\big[||\hat{\mu}_j - \mu || > \epsilon \big]\leq p,$$
  then setting $\hat{\mu} = \mathrm{med}(\hat{\mu}_1,\dotsc, \hat{\mu}_k)$,
  $$\mathrm{Pr}\big[||\hat{\mu} - \mu|| >
  C_\alpha \epsilon\big] \leq \mathrm{exp}\big(- k \cdot
  \mathrm{KL}(\alpha || p)\big)$$

*Proof.* If the event $||\hat{\mu} - \mu|| > C_\alpha \epsilon$
 occurs, then Lemma 1 shows that for at least $\alpha k$ estimators
 $\hat{\mu}_j$, the event $||\hat{\mu}_j - \mu || >
 \epsilon$ occurred. In contrast, the expected number of estimators
 for which this occurs is at most $pk$. A Chernoff bound gives:
 $$\mathrm{Pr}\left[\sum_{j=1}^k \mathbf{1} \left\{||\hat{\mu}_j
 - \mu|| > \epsilon\right\} \geq \alpha k \right] \leq \mathrm{exp} \big( - k \cdot
 \mathrm{KL}(\alpha || p)\big).$$
 <div align="right">☐</div>

*Note:* if the error $\epsilon$ is of the form:
$$\begin{equation*}
\epsilon = \textrm{approximation error} + \textrm{random error},
\end{equation*}$$
the first term is constant while the second term goes to zero as
sample size increases. The difference, therefore, between the Hilbert
space and Banach space cases may be large, since $C_\alpha \to 1$ in
the first case, while $C_\alpha \to 2$ in the second (as $\alpha \to
0$). 


### Example: Mean in Hilbert Space

Our setting here is as follows: let $X_1, \dotsc, X_n$ where $n \geq
2$ be i.i.d. estimators in a Hilbert space with mean $\mu = \mathbb{E}
X$. Let $\Sigma = \mathbb{E}[(X - \mu)\otimes (X - \mu)]$ be the
covariance operator, with $\mathrm{tr}(\Sigma) < \infty$.

**Corollary 3.** Let $0 < \delta < 1$ be the confidence
  parameter. Divide $X_1,\dotsc, X_n$ into $k(\delta)$ disjoint groups
  of size $\left\lfloor \frac{n}{k}\right\rfloor$ each,[^parameters] 
  and let $\hat{\mu}_j$ be the mean of each of these groups. Let the
  median be $\hat{\mu} = \mathrm{med}(\hat{\mu}_1,\dotsc,
  \hat{\mu}_k)$. Then,
  $$\mathrm{Pr}\left[||\hat{\mu} - \mu || \geq 11
  \sqrt{\frac{\mathrm{tr}(\Sigma)\log \frac{7}{5\delta}}{n}}\right]
  \leq \delta.$$ 
<div align="right">☐</div>

When the Hilbert space is $\mathbb{R}^D$, then would could also
perform a coordinate-wise median means. As we then need to union bound
over the coordinates, this leads to a dimension-dependent bound.[^bound]


It turns out that the coordinate-wise bound outpeforms the geometric
median bound for low dimensional Euclidean space. However, as $D$
becomes very large (or even infinite), the latter bound becomes
trivial.  In particular, the former becomes better as $D \geq 165$
(when $\delta = 0.1)$, and as $D \geq 15,806$ (when $\delta = 0.01$).


[^parameters]: Let $\alpha_* = \frac{7}{18}$ and $p_* = 0.1$ and $k =
\left\lfloor\frac{\log \frac{1}{\delta}}{\mathrm{KL}(\alpha_* || p_*)}
\right\rfloor.$

[^bound]: A coordinate-wise median leads to the following bound:
$$\mathrm{Pr}\left[||\hat{\mu}_* - \mu|| \geq \frac{22}{5}
\sqrt{\mathrm{tr}(\Sigma) \frac{\log \frac{8 D}{5\delta}}{n -
\frac{12}{5} \log \frac{8D}{5\delta}}}\right] \leq \delta.$$

### Example: Robust PCA

Our setting is as follows: let $X, X_1, \dotsc, X_n \in \mathbb{R}^D$
be i.i.d. random vectors, where $\mu = \mathbb{E}X$, $\Sigma =
\mathbb{E}[(X - \mu)(X - \mu)^T]$, and $\mathbb{E}||X||^4 <
\infty$. For simplicity, assume that all positive eigenvalues of
$\Sigma$ have algebraic multiplicity 1. Let the eigenvalues be
$\lambda_1 > \lambda_2 \geq \dotsm \geq 0$. PCA estimates the
projection operator $\mathrm{Proj}_m$ onto the subspace spanned by the
top $m$ eigenvectors. 


There are examples of robust PCA the rely on certain assumptions about
the data. For example, if the observations are contained in a
low-dimensional subspace, with noise corruption, the low-dimensional
subspace may be recovered exactly, see [Candès Li Ma Wright 2011] and
[Zhang Lerman 2014].

When there are no further assumptions about the data, then the
geometric median approach computes sample covariances. We can group
the data $X_1,\dotsc, X_n$ into $k$ groups, and calculate the sample
covariances within each of those groups:
$$\hat{\Sigma}_j := \frac{1}{|G_j|} \sum_{i\in G_j} X_i X_i^T.$$
The median $\hat{\Sigma}$ is computed with respect to the Frobenius
norm. For example, if we suppose the data is zero-centered (i.e. $\mu
= 0$). Let $Y_j = X_j X_j^T$. Then,
$$\begin{equation*}\mathbb{E}||Y - \mathbb{E}Y||^2 = \mathbb{E}||X|^4
- \mathrm{tr}(\Sigma^2)\end{equation*}.$$ 
Plugging directly into Corollary 3, we get:
$$\begin{equation*}
\mathrm{Pr}\left[||\Sigma - \hat{\Sigma}||_F \geq 11
\sqrt{\frac{\left[\mathbb{E}||X||^4 - \mathrm{tr}(\Sigma^2)\right]
\log \frac{7}{5 \delta}}{n}} \right]\leq \delta. 
\end{equation*}$$

From $\hat{\Sigma}$, we can estimate the projection operator. To pass
a bound on the covariance estimate to the projection operator
estimate, we rely on the Davis-Kahan perturbation theorem:  

**Theorem 4 (Davis-Kahan, see [Zwald Blanchard 2005]).** Let $A$ be a
  symmetric positive Hilbert-Schmidt operator[^hs] on a Hilbert
  space. Suppose it has nonzero  eigenvalues $\lambda_1 > \lambda_2 >
  \dotsm$, and let $\Delta_m = \lambda_m - \lambda_{m+1}$ be the $m$th
  eigengap. Let $B$ another symmetric operator such that $||B|| <
  \frac{\Delta_m}{4}$ and $(A+B)$ is still positive. Let
  $\mathrm{Proj}^A_m$ be the projector onto the first $m$
  eigendirections of $A$, and similarly for $A+B$. Then:
  $$|| \mathrm{Proj}^A_m - \mathrm{Proj}^{A+B}_m|| \leq \frac{2
  ||B||}{\Delta_m}.$$
<div align="right">☐</div>

Explicitly, if the eigengap $\Delta_m = \lambda_m - \lambda_{m+1}$ of 
$\Sigma$ is sufficiently large, so that $\Delta_m > 4 ||\Sigma -
\hat{\Sigma}||_F$, then Davis-Kahan directly yields a bound between
$\widehat{\mathrm{Proj}}_m$ and $\mathrm{Proj}_m$. That is:

**Corollary 5.** Let $X_1,\dotsc, X_n$ be drawn from a zero-mean
distribution with covariance $\Sigma$, and $\mathbb{E}||X||^4 <
\infty$. Let:
$$\epsilon = 11 \sqrt{\frac{\left[\mathbb{E}||X||^4 -
\mathrm{tr}(\Sigma^2)\right] \log \frac{7}{5 \delta}}{n}}.$$
If the eigengap of $\Sigma$ is at least $\Delta_m > 4 \epsilon$, then:
$$\mathrm{Pr}\left[||\widehat{\mathrm{Proj}}_m -
\mathrm{Proj}_m||_\mathrm{F} \geq \frac{2 \epsilon}{\Delta_m} \right] \leq \delta.$$
<div align="right">☐</div>

*Note:* when the data is not zero-centered, then the covariance
estimator is:
$$\begin{equation*}
\hat{\Sigma}_j = \frac{1}{|G_j|} \sum_{i \in G_j} (X_i - \mu)(X_i -
\mu)^T - (\mu - \hat{\mu}_j)(\mu - \hat{\mu}_j)^T,
\end{equation*}$$
which gives a worse dependence with $\epsilon =
O\left(\frac{\mathrm{tr}(\Sigma) \log \frac{1}{\delta}}{n}\right)$;
see paper for details. 

[^hs]: Note that the Hilbert-Schmidt norm on Euclidean space is also
called the Frobenius norm.

## Discussion

**Question 1.** If $\hat{\mu}_1,\dotsc, \hat{\mu}_k$ are a collection
  of i.i.d. estimators with $\mu = \mathbb{E}\hat{\mu}_i$, is
  $\mathrm{med}(\hat{\mu}_1,\dotsc, \hat{\mu}_k)$ an unbiased
  estimator of $\mu$? It seems like it should not be, for otherwise
  the geometric median and mean should coincide.

**Question 2.** What if instead of minimizing the '$\ell^1$-median':
$$\begin{equation*}
\mathrm{arg\ min}_{y \in \mathbb{X}} \sum_{j =1}^k ||y - x_j||,
\end{equation*}$$
we minimized an '$\ell^p$-median':
$$\begin{equation*}
\mathrm{arg\ min}_{y \in \mathbb{X}} \sum_{j =1}^k ||y - x_j||^p.
\end{equation*}$$
As $p$ takes on values between $1$ and $\infty$, what 'median' do we
obtain? For $p \in (1,2)$, are there different tradeoffs? Perhaps
$C_\alpha$ relaxes (so that it is easier to produce the individual
estimators), while the concentration bounds are also looser (though
perhaps still tighter than using all of our sample size 'budget' in
one go). This question seems related to $M$-estimation.
  
### Further Reading and Keywords

- [[Oliveira Lerasle 2011](https://arxiv.org/abs/1112.3914)] Robust empirical mean estimators.
- [[Alon Matias Szegedy 1996](https://www.tau.ac.il/~nogaa/PDFS/amsz4.pdf)] The space complexity of
  approximating the frequency moments.
- [[Bubeck Cesa-Bianchi, Lugosi 2013](https://arxiv.org/abs/1209.1727)] Bandits with heavy tail.
- [[Catoni 2012](https://arxiv.org/abs/1009.2048)] Challenging the empirical mean and empirical
  variance: a deviation study.
- [[Huber Ronchetti 2009](https://www.wiley.com/en-us/Robust+Statistics%2C+2nd+Edition-p-9780470129906)] *Robust Statistics*.
- [[Hubert Rousseeuw Aelst 2008](https://arxiv.org/pdf/0808.0657.pdf)] High-breakdown robust
   multivariate methods.
- [[Small 1990](https://www.jstor.org/stable/pdf/1403809.pdf)] A survey of multidimensional medians.
- [[Candès Li Ma Wright 2011](https://arxiv.org/pdf/0912.3599.pdf)] Robust principal component analysis?
- [[Zhang Lerman 2014](https://arxiv.org/abs/1112.4863)] A novel $M$-estimator for robust PCA.
- [[Cohen Lee Miller 2016](https://arxiv.org/pdf/1606.05225.pdf)] Geometric median in nearly linear time
- [[Zwald Blanchard 2005](http://ljk.imag.fr/membres/Laurent.Zwald/paper/Nips2005mod.pdf)] On the convergence of eigenspaces in kernel principla component analysis.
- M-estimation
- U-statistic, e.g. see [Theory of U-Statistics](https://link.springer.com/content/pdf/10.1007%2F978-94-017-3515-5.pdf)
- Robust estimation
- Empirical processes