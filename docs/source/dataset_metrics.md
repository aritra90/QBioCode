(data-complexity-measures)=

# Dataset complexity metrics

QProfiler describes every dataset it runs before it fits a single model. For each pass,
meaning a (dataset, embedding, split) combination, {py:func}`qbiocode.evaluation.dataset_evaluation.evaluate`
computes **141 complexity measures** on the data the models see. They are what
{doc}`QSage <apps/sage>` learns from and what the benchmark's
{ref}`meta-analysis <bench-meta>` regresses the classical-vs-quantum contrast on. So
what they measure bounds what either can conclude.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {octicon}`tools;1.4em` Native measures
:class-card: sd-border-primary
:link: metrics-native
:link-type: ref

**10 columns**, computed by QBioCode because pyMFE has no equivalent: intrinsic
dimension, conditioning, Fisher ratio, sparsity, kernel density, Isomap error,
fractal dimension.

:::

:::{grid-item-card} {octicon}`package;1.4em` pyMFE meta-features
:class-card: sd-border-info
:link: metrics-pymfe
:link-type: ref

**115 columns** (`mfe.*`) from a curated set of 76 pyMFE meta-features: the Lorena
et al. complexity suite, landmarking, statistics, information theory, trees,
clustering.

:::

:::{grid-item-card} {octicon}`pulse;1.4em` Target spectrum (label-based)
:class-card: sd-border-success
:link: metrics-task
:link-type: ref

**16 columns** (`task.*`), new: where the label `y` sits in the geometric spectrum
of `X`, smooth, oscillatory or noise. Each comes with a permutation-null z-score.

:::
::::

```{figure} _static/IEComplexity.png
:align: center
:width: 75%

Intrinsic complexity lives in the data: class overlap, nonlinear boundaries,
higher-order interactions, noise. Extrinsic complexity comes from the pipeline:
preprocessing, and a mismatch between the model's inductive bias and the data. The
measures below describe the intrinsic side, so that the extrinsic side, which model
won, can be explained by it.
```

## Where the columns appear

| File | Rows | What it holds |
|---|---|---|
| `RawDataEvaluation.csv` | one per dataset | the measures on the **raw**, unembedded matrix |
| `ModelResults.csv` | one per (pass, model) | the measures of that pass's **training** features (after embedding), beside every score |
| `collated/ModelResults.csv` | the whole run | the same, merged across jobs, and checked to agree across jobs to a relative 1e-6 |

Column names carry their provenance. Native measures keep their historical names
(`Fisher Discriminant Ratio`), pyMFE columns are `mfe.<feature>.<summary>`
(`mfe.f1.mean`, `mfe.n1`), and target-spectrum columns are `task.<feature>` with a
`task.<feature>_z` twin. Select a whole block with one `startswith`.

```{tip}
**Every measure is computed on the features the models actually saw.** For an embedded
pass that is the 8-component PCA or UMAP matrix, not the raw data. The raw-data view
is in `RawDataEvaluation.csv`. Compare like with like when you read a measure next to
a score.
```

```{tip}
**Some measures read the label.** The target-spectrum block and several pyMFE families
(landmarking, the Lorena F/L/N families, `mfe.c2`) use `y`. That is fine for describing
a dataset. But in a meta-analysis that explains a label-derived outcome, a label-reading
feature can be partly circular: see {ref}`bench-meta`.
```

(metrics-native)=

## Native measures

Computed in {py:mod}`qbiocode.evaluation.dataset_evaluation` because pyMFE has none of
them, or none with the same meaning.

| Column | What it measures | Reading |
|---|---|---|
| `Intrinsic_Dimension` | local-PCA estimate (`skdim` lPCA) of the manifold dimension the rows lie on | well below the feature count: simpler structure than the width suggests |
| `Condition number` | {math}`\kappa(X) = \sigma_{\max} / \sigma_{\min}` of the data matrix | above about {math}`10^3`: ill-conditioned, near-collinear features |
| `Fisher Discriminant Ratio` | multivariate {math}`\mathrm{tr}(S_B) / \mathrm{tr}(S_W)`, between- over within-class scatter | larger is **easier**. pyMFE's `f1` is per-feature and inverted, so it is not a substitute |
| `Coefficient of Variation %`, `std_co_of_v` | mean and SD over features of {math}`\sigma / \mu \times 100` | dispersion comparable across scales |
| `# Low variance features` | features below the 25th percentile of the variance distribution | likely uninformative columns |
| `# Non-zero entries` | count of non-zero values | sparsity |
| `Mean Log Kernel Density` | mean log-likelihood under a Gaussian KDE | higher: more concentrated data |
| `Isomap Reconstruction Error` | residual of a 2-D Isomap embedding (geodesic vs Euclidean) | larger: more curved, less flat manifold. Seeded, so identical across jobs |
| `Fractal dimension` | Higuchi fractal dimension | between 1 and 2; higher means more irregular |

:::{dropdown} Formulas for the native measures
:icon: code

```{math}
\kappa(\mathbf X) = \frac{\sigma_{\max}(\mathbf X)}{\sigma_{\min}(\mathbf X)},
\qquad
\mathrm{FDR} = \frac{\mathrm{tr}(\mathbf S_B)}{\mathrm{tr}(\mathbf S_W)},
\qquad
CV = \frac{\sigma}{\mu}\times 100\,\%,
```

```{math}
\overline{\log p} = \frac{1}{n}\sum_{i=1}^{n}
\log\!\left(\frac{1}{n h^{d}}\sum_{j=1}^{n} K\!\left(\frac{x_i-x_j}{h}\right)\right),
\qquad
D_f = \lim_{\epsilon\to 0}\frac{\log N(\epsilon)}{\log(1/\epsilon)} .
```

References: Fukunaga & Olsen (1971) on intrinsic dimension; Fisher (1936); Golub & Van
Loan (2013) on conditioning; Silverman (1986) on kernel density; Higuchi (1988) on
fractal dimension.
:::

(metrics-task)=

## Target spectrum (label-based)

Everything else either describes `X` alone, or describes `y` only through a
classifier's view of it. This block, from {py:mod}`qbiocode.evaluation.task_spectrum`,
answers a different question: **given the geometry `X` induces, is `y` a smooth
function on it, a structured oscillatory one, or broadband noise?** Parity,
checkerboards and alternating-sign targets are deterministic and noise-free, yet sit
almost entirely in the high-frequency part of the geometry. Low-pass learners fail on
them for reasons unrelated to label noise, and nothing else in the 141 columns tells
that failure apart from ordinary difficulty.

**Construction.** Build a mutual k-NN graph on the rows of `X`, with locally scaled
weights, and take its normalized Laplacian
{math}`L_{\mathrm{sym}} = I - D^{-1/2} W D^{-1/2}`. Its eigenvectors are a Fourier basis of
the data geometry: small eigenvalues are smooth modes, large ones oscillate between
neighbours. Expanding the centred label in that basis gives the target's power over
geometric frequency:

```{math}
\alpha_j = u_j^\top \tilde y, \qquad p_j = \frac{\alpha_j^2}{\sum_l \alpha_l^2}.
```

| Column (`task.`) | What it measures |
|---|---|
| `graph_dirichlet` | {math}`\sum_j \lambda_j p_j`: the mean geometric frequency of the target |
| `graph_hf_mass` | share of target power in modes with {math}`\lambda \ge 1`, where neighbours systematically disagree |
| `graph_spec_entropy` | normalized entropy of {math}`p_j`: a few modes (low) against broadband (high) |
| `graph_bandwidth90` | fraction of frequency-ordered modes needed to hold 90 % of the target power |
| `diffusion_half_life` | how long the target survives diffusion on the graph, relative to the graph's slowest timescale, in (0, 1] |
| `pca_tail_signal` | share of target power **outside** the PCs that explain 90 % of `X`'s variance. High means PCA would discard the signal |
| `h0_fragmentation` | within-class MST weight over pooled MST weight: about 1 for contiguous classes, rising towards 2 when they interleave |
| `purity_auc` | chance-corrected neighbourhood label purity, integrated over log scale |
| `*_z` | each of the above as a z-score against random relabelling with class proportions fixed |

How the two headline features separate the cases:

| Target | `graph_hf_mass` | `graph_spec_entropy` | Reading |
|---|---|---|---|
| smooth, simple | low | low | easy geometric target |
| complex, smooth | low to moderate | high | many low-frequency modes |
| **structured oscillatory** | **high** | **low to moderate** | parity- or checkerboard-like |
| noisy | high | high | broadband target, or noise |

```{tip}
**Read the `_z` column, not the raw value, on wide data.** In sparse, high-dimensional
data random labels look high-frequency by default, so a raw `graph_hf_mass` of 0.4 says
little on its own. The `_z` twin compares it with the same statistic under random
relabelling. It is cheap: the graph is diagonalized once per `k`, and each permutation
is one matrix product.
```

```{tip}
**The interesting signature is a conjunction, not one column.** The pattern worth
testing is a structured high-frequency target (high `graph_hf_mass` and
`graph_hf_mass_z`) of low spectral rank (low `graph_spec_entropy` and
`graph_bandwidth90`), with signal outside the dominant variance (high
`pca_tail_signal`), that cheap classical landmarks cannot express. Any one of those
alone is weak.
```

Graph features are the median over `k` in (5, 10, 20), so that "frequency" is not an
artefact of one neighbourhood size. Pass `task_spectrum=False` to `evaluate` to skip
the block on an unusually tall dataset; the cost is {math}`O(n^3)`, about 12 s at
`n = 800`.

(metrics-pymfe)=

## pyMFE meta-features

Most columns come from [pyMFE](https://pymfe.readthedocs.io/), whose
[meta-feature list](https://pymfe.readthedocs.io/en/latest/auto_pages/meta_features_description.html)
documents each one. QBioCode uses a **curated 76 of pyMFE's roughly 105**, summarized
by mean and SD where pyMFE returns several values. The rest are excluded on measured
grounds: undefined (a silent `NaN`), constant, or intractable when `p >> n`. Every
exclusion is recorded with its reason in {py:mod}`qbiocode.evaluation.mfe_features`.
The headliners:

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Classification complexity (Lorena et al. 2019)
- `mfe.f1`, `mfe.f2`, `mfe.f3`, `mfe.f4`: feature overlap between classes
- `mfe.l1`, `mfe.l2`, `mfe.l3`: distance from linear separability
- `mfe.n1`: fraction of points on the class boundary (MST-based)
- `mfe.n3`: leave-one-out 1-NN error
- `mfe.t3`, `mfe.t4`: PCA dimension ratios
- `mfe.c2`: class-imbalance ratio
:::

:::{grid-item-card} Landmarking
The accuracy of deliberately cheap learners on the data itself:
- `mfe.one_nn`, `mfe.elite_nn`
- `mfe.naive_bayes`, `mfe.linear_discr`
- `mfe.best_node`, `mfe.worst_node`

A landmark *is* a cheap measurement of model performance, which makes these the most
directly useful columns for QSage.
:::

:::{grid-item-card} General and statistical
- `mfe.nr_inst`, `mfe.nr_attr`, `mfe.attr_to_inst`: shape and width-to-height ratio
- `mfe.cor`, `mfe.nr_cor_attr`: mean absolute correlation, and the share of pairs above 0.5
- `mfe.skewness`, `mfe.kurtosis`, `mfe.var`, `mfe.sd`
- `mfe.can_cor`, `mfe.w_lambda`: canonical correlation with the class
:::

:::{grid-item-card} Information, trees and clusters
- `mfe.class_ent`, `mfe.joint_ent`, `mfe.mut_inf`, `mfe.ns_ratio`: entropy and mutual information
- `mfe.tree_depth`, `mfe.leaves`, `mfe.nodes`: shape of an induced decision tree
- `mfe.sil`, `mfe.ch`, `mfe.vdu`: how well labels match the data's own clusters
:::
::::

```{tip}
**The L family, `mfe.f2` and `mfe.f4` degenerate when there are more features than
samples.** Such data is almost always linearly separable, so L1–L3 and F4 go to zero
however hard the problem is. They are informative on embedded passes (8 components).
Read them next to `mfe.attr_to_inst`.
```

## What changed from the old column set

QProfiler used to write 23 hand-rolled measures. Ten survive as the native block. The
rest moved to pyMFE, which computes them more carefully: `# Samples` became
`mfe.nr_inst`, `# Features` became `mfe.nr_attr`, `Feature_Samples_ratio` became
`mfe.attr_to_inst`, and `Mutual information`, `Variation`, `Skewness` and `Kurtosis`
became their `mfe.` counterparts. `Total Correlations` was the unnormalized
{math}`\sum_{i\ne j}|\rho_{ij}|`, which grows with {math}`p^2` and so mostly measured width. It is
replaced by the mean `mfe.cor.mean` and the share `mfe.nr_cor_attr`. `std_entropy` was
always exactly 0 (the SD of a scalar) and is gone.

```{seealso}
- Lorena, A. C., et al. (2019). How complex is your classification problem? A survey on
  measuring classification complexity. *ACM Computing Surveys*, 52(5), 1–34.
- Pfahringer, B., Bensusan, H., & Giraud-Carrier, C. (2000). Meta-learning by
  landmarking various learning algorithms. *ICML*, 743–750.
- Alcobaça, E., et al. (2020). MFE: Towards reproducible meta-feature extraction.
  *JMLR*, 21(111), 1–5.
- The data-complexity slides of the {doc}`ISMB 2025 workshop <workshops/ISMB_2025>`.
```
