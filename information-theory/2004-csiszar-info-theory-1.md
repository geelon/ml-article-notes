---
title: Information Theory and Statistics, a tutorial (Chapter 1)
date: 2018-12-12
---

Csiszár, I, and Shields, P.C. [Information Theory and Statistics: A Tutorial](https://users.renyi.hu/~csiszar/Publications/Information_Theory_and_Statistics:_A_Tutorial.pdf)

- [this chapter](#): Preliminaries
- [next chapter](./2004-csiszar-info-theory-2.html): Preliminaries: Large deviations, hypothesis testing

## Summary

Brief review of information theory:

- Let $A = \{a_1,\dotsc, a_{|A|}\}$ be a finite space. Let $P,Q$ be
  distributions over $A$. The entropy of $P$ is:
  $$H(P) = - \sum_{a \in A} P(a) \log P(a).$$
- We interpret $0 \log 0$ as $0$.
- The KL-divergence (a.k.a. information divergence or $I$-divergence
  or relative entropy) of $P$ and $Q$ is:
  $$D(P||Q) = \sum_{a \in A} P(a) \log \frac{P(a)}{Q(a)}.$$
- The log-sum inequality shows that KL-divergence is nonnegative. It
  is also finite if $\mathrm{supp}(P) \subset \mathrm{supp}(Q)$.
- A *code* for $A$ with image alphabet $B$ is an injective mapping $C:
  A \to B^*$, where $B^*$ are all finite sequences of letters in
  $B$. If a sequence $s = uv$ is formed by the concatenation of two 
  subsequences $u$ and $v$, we write $u \prec s$ to denote that $u$ is
  a *prefix* of $s$. Here, we let $B = \{0,1\}$.
- A *prefix-free code* (or *prefix code*) is an encoding of $A$ such
  that $C(a) \prec C(a')$ if and only if $a = a'$.

**Lemma (Kraft's inequality).** A function $L : A \to \mathbb{N}$ is
  the length function of a prefix-free code if and only if it
  satisfies the Kraft inequality:
  $$\sum_{a \in A} 2^{-L(a)} \leq 1.$$

*Proof.* Given a prefix-free code, each $a \in A$ maps to a dyadic
 number $t(a)$:
 $$\begin{equation*}a \mapsto 0.b_1\dotsc b_{L(a)}.\end{equation*}$$
 The prefix-free condition implies that the intervals $[t(a), t(a) +
 2^{-L(a)})$ are disjoint over $a \in A$. Thus, $\sum 2^{-L(a)} \leq
 1$.

 Conversely, suppose $\sum 2^{-L(a)} \leq 1$. Assume that $L(a_i) \leq
 L(a_{i+1})$. Then, represent $t(a_i)$ by $\sum_{j < i} 2^{-L(a_j)}$.
 <div align="right">☐</div>

**Theorem (Shannon's noiseless coding).** *Let $P$ be a probability
  distribution on $A$. Then each prefix-free code has expected length:*
  $$\mathbb{E}[L] \geq H(P).$$
  *Furthermore, there exists a prefix-free code with length function
  $\lceil - \log P(a) \rceil$; its expected length satisfies:*
  $$\mathbb{E}[L] < H(P) + 1.$$

*Proof.* By Kraft's inequality, we have:
$$\begin{align*}
\mathbb{E}[L] - H(P) &= \sum_{a \in A} p(a) \log
\frac{p(a)}{2^{-L(a)}}
 \geq \log \frac{1}{ \sum_{a \in A} 2^{-L(a)}} \geq 0.\end{align*}$$

On the other hand, $L(a) = \lceil - \log P(a)\rceil$ satisfies Kraft's
 inequality. 
 <div align="right">☐</div>




## Technical Notes

**Lemma (Log-sum inequality).** Let $p_1,\dotsc, p_t$ and $q_1,\dotsc
  q_t$ be arbitrary nonnegative numbers. Let $p = \sum_i p_i$ and $q =
  \sum_i q_i$. Then:
  $$ \sum_{i=1}^t p_i \log \frac{p_i}{q_i} \geq p \log \frac{p}{q}.$$

*Proof.* This follows from the convexity of $f(x) = x \log x$ and
 Jensen's inequality.
 <div align="right">☐</div>