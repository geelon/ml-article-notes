---
title: A Group-Theoretic Approach to Abstraction, hierarchical, interpretable and task-free clustering
date: 2018-08-05
---

Yu, Haizi, Igor Mineyev, and Lav R. Varshney. "A Group-Theoretic Approach to Abstraction: Hierarchical, Interpretable, and Task-Free Clustering." arXiv preprint arXiv:1807.11167 (2018).

## Summary

This paper approaches abstraction through clustering. In particular,
the partitioning of a space into partitions of equivalent objects. 

### Mathematical Formalism

Let $X$ be the instance space (the set over which to form the
abstraction). An *abstraction* is defined to be a partition of the
space $X / \sim$ with respect to some equivalence relation
$\sim$. Note that the space $\mathfrak{P}_X^*$ of all partitions over
$X$ is a poset; the *join* of two partitions is the coarsest common
refinement, while the *meet* is the finest common coarsening.

**Definition 1.** An *abstraction universe* for a set $X$ is a
  sublattice of $\mathfrak{P}_X^*$. An *abstraction join-semiuniverse*
  (resp. *meet-semiuniverse*) is a join-semilattice
  (resp. meet-semilattice) of $\mathfrak{P}_X^*$ (closed under joins,
  resp. meets). Thus, an abstraction universe is both an
  join-semiuniverse and meet-semiuniverse.

**Remark 2.** Let $\mathsf{F}(X)$ denote the space of permutations on
  $X$ (i.e. bijections from $X$ to $X$). Let $H \leq \mathsf{F}(X)$ be
  any subgroup. The orbit of $x \in X$ under $H$ is the set $Hx :=
  \{h(x) : h \in H\}$. The quotient space $X/H$ is the space of
  abstractions that respect $H$-symmetry or $H$-invariance. 

**Definition 3.** An *abstraction generating function* is a mapping
  $\pi : \mathcal{H}_{\mathsf{F}(X)}^* \to \mathfrak{P}_X^*$ where
  $\mathcal{H}_{\mathsf{F}(X)}^*$ is the collection of all subgroups
  of $\mathsf{F}(X)$. In particular, $\pi(H) = X/H$.

**Proposition 4.** The abstraction generating function is surjective
  but not necessarily injective.

[See paper for additional propositions]

### Information Theory

**Definition 5.** An *information element* is an equivalence class of
  random variables (of a common sample space) with respect to the
  `being informationally equivalent' relation. An *information
  lattice* is a lattice of information elements, where partial order
  is defined by $x \leq y \Leftrightarrow H(x|y) = 0$, where $H$ is
  entropy.

## Discussion

### Further Reading

- [[Li Chong 2011](https://pdfs.semanticscholar.org/f20a/51915b922c83cb91202580d4e533149c509e.pdf)] On a connection between information and group lattices.
- [[Saitta Zucker 2013](https://link.springer.com/content/pdf/10.1007/978-1-4614-7052-6.pdf)] *Abstraction in Artificial Intelligence and Complex Systems.*