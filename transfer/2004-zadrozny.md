---
title: Learning and Evaluating Classifiers under Sample Selection Bias
date: 2018-08-22
---

Zadrozny, Bianca. "Learning and evaluating classifiers under sample selection bias." Proceedings of the twenty-first international conference on Machine learning. ACM, 2004.


## Summary

We often learn on training data that is sampled from a different
distribution than the test data. In econometrics, this is called the
*sample selection bias*. And in the context of linear regression
models, there is work by Heckman 1979 to correct the bias. In
statistics, this is related to the *missing data* problem. This paper
presents a method to correct the bias using a form of rejection
sampling. 

### Setting

Consider a classification setting where samples $(x,y,s) \in
\mathcal{X}\times \mathcal{Y} \times \mathcal{S}$ are drawn from a
distribution $\mathcal{D}$. Here, $s$ is a binary variable that
indicates whether the datapoint $(x,y)$ is seen in training.

There are four cases regarding the dependence of $s$ on the example
$(x,y)$:

1. $s$ is independent of $x$ and independent of $y$. In statistics,
this is the *missing completely at random* (MCAR) case. Here, the
selected sample is not biased.
2. $s$ is independent of $y$ given $x$:
$$\begin{equation*} P(s| x,y) = P(s|x).\end{equation*}$$
In statistics, this is the *missing at random* (MAR) case.
3. $s$ is independent of $x$ given $y$:
$$\begin{equation*} P(s| x,y) = P(s|y).\end{equation*}$$
This corresponds to a change in the prior probabilities of the labels,
and this setting is well-studied in the machine learning
literature. See (Elkan 2001) and (Bishop 1995), for example.
4. No independence assumption holds between $x$, $y$ and $s$. This is
the *not missing at random* (NMAR) case. In this case, there is no way
to provide an unbiased estimate, unless we have access to additional
features $x_s$ such that $p(s|x_s,x,y) = p(s|x_s)$. This is the usual
situation in econometrics, where $x_s$ contain all the features that
are relevant to sample selection.

In the classification setting, we mostly care about prediction and not
the underlying generative model. And so, this paper focuses on case
(2). In practice, to make $p(s|x,y) = p(s|x)$ true, $x$ must contain
all the features used for sample selection.

### Local and Global Learners

This paper introduces two types of classifier learners:

**Definition 1.** A learner is *local* if its output is depends
  asymptotically only on $P(y|x)$. Otherwise, the learner is *global*
  when it depends on $P(y|x)$ and $P(x)$.

Notice that sample selection bias does not affect local learners
because their classification depends only on the distribution at a
given input $x$, and by definition, $P(y|x) = P(y|x,s=1)$.

**Example 2 (Bayes Classifier and Logistic Regression).** Bayes
  classifiers depend only on the posterior probability $P(y|x)$ to
  make a decision. They are local learners. Similarly, logistic
  regression fits parametric model using MLE on:
  $$\begin{equation*}
  P(y= 1| x) = \frac{1}{1 + \mathrm{exp}(\beta_0 + \beta^\top x)}.
  \end{equation*}$$
  If $P(s=1| x) > 0$ for all $x$ in the support of $P(x)$, then
  logistic regression is also local.

**Example 3 (Naive Bayes Classifier).** The naive Bayes classifier
  uses MLE to maximize $P(y|x_1) \dotsm P(y|x_n)$. However, in
  general, it is not the case that:
  $$\begin{equation*}
  \prod_{i=1}^n P(y|x_i) = \prod_{i=1}^n P(y|x_i, s=1),
  \end{equation*}$$
  so these are global learners.

**Example 4 (Decision Trees).** Decision trees recursively split the
  space $\mathcal{X}$ along a single feature, based on the impurity
  after the split. Thus, these classifiers depend on $P(y|t)$, where
  $t$ is a test that uses one feature of $x$. And in general, $P(y|t)
  \ne P(y|t, s=1)$. So, these are also global learners.

**Example 5 (Hard SVM).** Assuming separable data, if the selection
  probability $P(s=1 | x) > 0$ for all $x$ such that $P(x) > 0$, then
  asymptotically, the decision boundary resulting from SVM will
  converge to the expecte decision boundary without sampling selection
  bias. Thus, these are local learners.

**Example 6 (Soft SVM).** On the other hand, soft-SVM takes into
  account the density of positive vs. negative points, and so they are
  global learners.

### Correcting the Bias

When learning a classifier, we often rely on estimating the risk,
$$\begin{equation*}
\mathbb{E}_{(x,y) \sim \mathcal{D}}\left[\ell(x,y)\right] =
\sum_{(x,y)} \ell(x,y) \cdot P(x,y).
\end{equation*}$$
However, with sampling selection bias, we instead estimate:
$$\begin{equation*}
\mathbb{E}_{(x,y,s) \sim \mathcal{D}} \left[\ell(x,y)\big| s =
1\right] = \sum_{(x,y)} \ell(x,y) \cdot P(x,y|s=1). 
\end{equation*}$$
To correct the bias, we should then reweight the sample $(x,y)$ by
$P(x,y)/P(x,y|s=1)$. And as we have $P(y|x) = P(y|x, s=1)$, this is
equivalent to reweighting the sample by:
$$\begin{align}
\frac{P(x,y)}{P(x,y|s=1)} &= \frac{P(y|x)P(x)}{P(y|x, s=1)
P(x|s=1)}\nonumber \\
&= \frac{P(x)}{P(x| s=1)}\nonumber \\
&= \frac{P(s=1)}{P(s=1|x)} 
\end{align}$$
We obtain an unbiased estimate of the risk by reweighting the cost by
this value. In practice, we would need to estimate $P(s=1 | x)$;
however, this is often feasible (e.g. in medical trials, the outcome
$y$ of a patient $x$ is known only if they undergo a trial $s$).

## Discussion

**Question 1.** What is rejection sampling? Is it related to
  importance sampling?

### Further Reading

- [[Heckman 1979](./)] Sample selection bias as a specification error
- [[Elkan 2001](./)] The foundations of cost-sensitive learning
- [[Bishop 1995](./)] Neural networks for pattern recognition