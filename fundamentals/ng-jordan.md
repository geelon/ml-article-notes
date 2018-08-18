---
title: On Discriminative vs. Generative classifiers, A comparison of logistic regression and naive Bayes
date: 2018-08-15
---

## Summary

Let $x$ be inputs and $y$ be labels. *Generative classifiers* learn a
model of the joint probability $p(x,y)$. In contrast, *discrimative
classifiers* learn $p(y|x)$ directly.

In general, it is thought that learning the discriminative model is
preferable. In addition to the sample complexity in the discriminative
case being linear in the VC dimension, citing Vapnik:

>One should always solve the [classification] problem directly and
>never solve a more general problem as an intermediate step [such as
>modeling *p*(*x*|*y*)]. 

When learning a model, we can either maximize the joint likelihood of
inputs and labels $p(x,y)$, or we can maximize the conditional
likelihood $p(y|x)$ directly or the 0-1 loss by thresholding $p(y|x)$
to make predictions. Let the resulting classifiers be called
$h_\mathrm{Gen}$ and $h_\mathrm{Dis}$, respectively. The two are
called a *generative-discriminative pair*.

### Setting

Consider the binary classification setting on
$\mathcal{X}$ and labels $\mathcal{Y} = \{0,1\}$. Let $\mathcal{D}$ be
a joint distribution over $\mathcal{X} \times \mathcal{Y}$ from which
a training set $S = \{x^{(i)},y^{(i)}\}_{i=1}^m$ is drawn.

**Generative Classifier.** The generative classifier $h_\mathrm{Gen}$
returns the argmax over $y$ of the joint probability $p(x_1,\dotsc,
x_n, y) = p(x_1,\dotsc, x_n| y) p(y)$. That is, $h_\mathrm{Gen}(x) =
1$ if and only if the following is positive:
$$\begin{equation*}
l_\mathrm{Gen}(x) = \log \frac{p(x_1,\dotsc, x_n, y=1)}{p(x_1,\dotsc,
x_n, y=0)} = \log \frac{p(x_1,\dotsc, x_n| y=1)}{p(x_1,\dotsc, x_n|
y=0)} + \log \frac{p(y = 1)}{p(y=0)}.
\end{equation*}$$
The naive Bayes assumption is that $p(x_1,\dotsc, x_n|y) =
\prod_{i=1}^n p(x_i|y)$.

*Discrete case.* Let $\mathcal{X} = \{0,1\}^n$.  We can compute
estimates $\hat{p}(x_i|y)$ and $\hat{p}(y)$ based on samples:
$$\begin{equation*}
\hat{p}(x_i = 1 | y = b) = \frac{\#_S\{x_i = 1, y=b\} + \ell}{\#_S\{y
= b\} + 2\ell}.
\end{equation*}$$
When $\ell = 0$, this is the usual empirical estimate. But $\ell$ may
also be set to a positive number, applying the Laplace smoothing to
the probabilities.

*Continuous case.* Here, the only difference is that we let
$\mathcal{X} = [0,1]^n$ and the class conditional distributions
$\hat{p}(x_i | y = b)$ be parametrized by a univariate distribution
$\mathcal{N}(\hat{\mu}_{i|y = b} , \hat{\sigma}_i^2)$.


**Discriminative Classifier.** The discriminative classifier
$h_\mathrm{Dis}$ 

## Discussion
### Further Reading