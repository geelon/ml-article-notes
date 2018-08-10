---
title: Machine Learning, a probabilistic approach (Chapter 3)
date: 2018-08-09
---

Murphy, K. "Machine Learning: A Probabilistic Approach" *Massachusetts Institute of Technology* (2012).

- current chapter: Generative Models for Discrete Data

## Summary

### Setting
In *concept learning*, we begin with a *hypothesis space*
$\mathcal{H}$ of concepts, with the goal of learning a specific
concept from data/examples $D$. The subset of $\mathcal{H}$ consistent
with $D$ is called the *version space*.

For example, consider a *number game* from Tenenbaum's PhD thesis
(Tenenbaum 1999), where the learner is presented with a sequence of
randomly drawn positive examples of class of numbers $C$ between 1 and
100. For example, we might see a sequence of data $8, 2, 64$.

### Likelihood
From this data, we might decide to learn the concept "powers of two"
or "even numbers", as both concepts are consistent with the
evidence. However, intuitively, it seems unlikely that if the class
were "even numbers", then we'd see only powers of twos. We want to
avoid *suspicious coincidences*.

More formally, we take the *strong sampling
assumption*, where the data we see are samled uniformly from the
*extension* of a concept (i.e. the positive instances of a
concept). Then, the probability of independently sampling $N$ items
from a concept $h$ is:
$$p(D|h) = \left[\frac{1}{|h|}\right]^N,$$
where $|h|$ is the size of the extension. This is *Occam's razor*.

For example, suppose we see data $D = \{16\}$. Then, the likelihood
that the concept is "powers of two" is $p(D|h_{\mathrm{two}}) = 1/6$, 
since there are 6 powers of two less than 100. Similarly,
$p(D|h_{\mathrm{even}}) = 1/50$.

After four examples, $D = \{16, 8, 64, 2\}$, their respective
likelihoods are $p(D|h_\mathrm{two}) = (1/6)^4$ and
$p(D|h_\mathrm{even}) = (1/50)^4$, with a *likelihood ratio* of almost
5000:1 in favor of $h_\mathrm{two}$.

### Prior

Notice however that the concept "powers of two except 32" will be more
likely than "powers of two". We'd like to say that the former concept
is 'unnatural' with respect to a *prior*. This *subjective* aspect of
Bayesian reasoning is quite controversial, for two individuals with
different priors will reach different answers.

Without priors, then it is information-theoretically impossible to
learn from small samples.

### Posterior

The *posterior distribution* after seeing data $D$  is the likelihood
of that data times the prior $p$, normalized. That is,
$$p(h|D) = \frac{p(D|h) p(h)}{\sum_{h' \in \mathcal{H}} p(D, h')}.$$
The concept $h$ with the highest posterior $p(h|D)$ becomes the
*maximum a posteriori (MAP) estimate*.
Notice that as the likelihood term depends exponentially on the number
samples, $N$, the MAP estimate converges toward the *maximum
likelihood estimator* (MLE),
$$\hat{h}_\mathrm{MLE} := \mathrm{arg\ max}_h p(D | h).$$
In other words, with enough data, the data overwhelms the prior. If
the true hypothesis is in $\mathcal{H}$, then the MAP estimate/MLE
will converge to the true hypothesis.

Here, Bayesian inference and MLE are consistent estimators, and we say
that the hypothesis space is *identifiable in the limit*.

### Naive Bayes Classifiers

Consider vectors of discrete-valued features, $\mathbf{x} \in
\{1,\dotsc, K\}^D$, where $K$ is the number of values for each
feature, and $D$ is the number of features.

Taking a generative approach, we specify the class conditional
distribution $p(\mathbf{x}|y = c)$. If we make the (naive) assumption
that the features are conditionally independent given the class, the
resulting model is a *naive Bayes classifier* (NBC). We have the class
condition density:
$$p(\mathbf{x}| y = c, \theta) = \prod_{j=1}^D p(x_j | y = c,
\theta_{jc}). $$

Examples of NBCs:

- Gaussian distributions, $p(\mathbf{x} | y = c, \theta) =
  \prod_{j=1}^D \mathcal{N}(x_j | \mu_{jc}, \sigma_{jc}^2)$.
- Multivariate Bernoulli naive Bayes model, for binary features:
  $p(\mathbf{x}|y = c,\theta) = \prod_{j=1}^D \mathrm{Ber}(x_j |
  \mu_{jc})$.
- Multinoulli distribution, where $x_j \in \{1,\dotsc, K\}$, and
  $p(\mathbf{x}|y = c, \theta) = \prod_{j=1}^D \mathrm{Cat}(x_j
  |\mathbf{\mu}_{jc})$. 

### Model Fitting

We can use MLE to estimate the parameters. However, it can overfit the
data. One solution to overfitting is to be Bayesian, using a prior
$\mathbf{\pi}$.

## Discussion

**Question 1.** What does it mean to "take a generative approach"?
  Does this mean that we assume samples are generated according the
  class they come from?

**Note 2.** One model for document classification is using the
  multinomial classifier. However, it does not take into account the
  *burstiness* of a word: here, given a document, most words will
  never appear. But if it appears once, it is much more likely to
  appear more than once; i.e. words occur in bursts. One way to
  account for this is to use the  Dirichlet Compound Multinomial (DCM)
  density. If the multinomial model corresponds to drawing a ball from
  an urn with $K$ colors, the DCM model draws a ball then replaces it
  while also adding an additional copy. This is *Polya's urn*.

**Note 3.** Should go back and work out some more examples.



