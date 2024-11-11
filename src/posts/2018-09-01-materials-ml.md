---
author: "Julien"
desc: "Methodology in Materials Science"
keywords: "ML, Materials"
lang: "en"
title: "Cross-Validation Methodology in Materials Science"
---

Cross-validation is a critical part of statistical methodology, for ad hoc models cross-validation may be the only indication of model performance, and without a reasonable cross-validation methodology serious over-fitting can go undetected.
This issue is particularly relevant to domains where small data-sets with a comparatively large number of features is common, for example Materials Science or Genomics.
If the cross-validation method does not take into consideration the feature selection (that is, considering feature selection as part of model selection), a significant selection bias can occur, see [this paper](https://doi.org/10.1073/pnas.102102699) for an example with gene-expression data.
In general, a reasonably robust validation methodology should be chosen before model selection, and final hold-out sets should be used when possible.  

# [Poster PDF](/pdf/poster_SULI.pdf)
