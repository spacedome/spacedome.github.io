---
author: "Julien"
desc: "Hyper Parameter Optimization"
keywords: "ML"
lang: "en"
title: "Learning Hyper-Parameters with Bayesian Optimization"
---

It is well known in the Machine Learning literature that a Grid Search for optimizing parameters is sub-optimal, and generally outperformed by a Random Search.
This project explored using Bayesian Optimization as a sequential search strategy, testing the method on the hyper-parameters of a relatively simple Neural Network performing image classification.

The main conclusion of this research is that incredible computational resources are needed to characterize NN hyper-parameter optimization methods, see the Google Vizier "paper" for context.
Despite this, Bayesian Optimization is a promising technique for sequential, potentially stochastic, optimization problems where the cost of sampling is a limiting factor.
For example, see my paper with Alex Dunn, *Rocketsled*, on applications of Bayesian Optimization to Computational Materials Science.

In a broader context, stop searching for parameters by hand, and consider random search over grid search if the relative importance of each parameter is unknown (for example some may not be important at all).
See the [paper](http://www.jmlr.org/papers/v13/bergstra12a.html) by Bergstra and Bengio for reference.
There is no reason to search parameters manually, there are many easy to use optimization codes to choose from, for example [skopt](https://scikit-optimize.github.io/).
# [Poster PDF](/pdf/poster_CS682.pdf)
