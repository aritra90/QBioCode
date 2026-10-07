(data-complexity-measures)=

# Dataset complexity metrics

QProfiler describes every dataset before it fits a model. For each (dataset, embedding,
split) pass, {py:func}`qbiocode.evaluation.dataset_evaluation.evaluate` computes
**141 measures** on the features the models see. {doc}`QSage <apps/sage>` learns from
them, and the benchmark's {ref}`meta-analysis <bench-meta>` regresses the
classical-vs-quantum contrast on them.

::::{grid} 1 1 3 3
:gutter: 3
:class-container: sd-mb-3

:::{grid-item-card} Native · 10
:link: metrics-native
:link-type: ref
Geometry and conditioning that pyMFE does not cover.
:::

:::{grid-item-card} pyMFE · 115
:link: metrics-pymfe
:link-type: ref
A curated 76 of pyMFE's meta-features, as `mfe.*` columns.
:::

:::{grid-item-card} Target spectrum · 16
:link: metrics-task
:link-type: ref
Label-based (`task.*`): is `y` smooth, oscillatory or noise on `X`?
:::
::::

## Where the columns appear

| File | One row per | Measures computed on |
|---|---|---|
| `RawDataEvaluation.csv` | dataset | the raw, unembedded data |
| `ModelResults.csv` | (pass, model) | that pass's training features, after any embedding |

Native measures keep their historical names (`Fisher Discriminant Ratio`). pyMFE columns
are `mfe.<feature>.<summary>` (`mfe.f1.mean`), and target-spectrum columns are
`task.<feature>`, each with a `task.<feature>_z` twin.

```{tip}
Several measures read the label `y`: the target spectrum, landmarking, and the Lorena
complexity families. That is right for describing a dataset. But when a meta-analysis
explains a label-derived outcome, they can be partly circular.
```

(metrics-native)=

## Native measures

Computed in {py:mod}`qbiocode.evaluation.dataset_evaluation`, because pyMFE has none of
them, or none with the same meaning.

| Column | What it measures |
|---|---|
| `Intrinsic_Dimension` | local-PCA estimate of the dimension of the manifold the rows lie on |
| `Condition number` | {math}`\sigma_{\max}/\sigma_{\min}` of the data matrix; above about {math}`10^3` is ill-conditioned |
| `Fisher Discriminant Ratio` | multivariate {math}`\mathrm{tr}(S_B)/\mathrm{tr}(S_W)`; larger is **easier** (pyMFE's `f1` is per-feature and inverted) |
| `Coefficient of Variation %`, `std_co_of_v` | mean and SD over features of {math}`\sigma/\mu`, in percent |
| `# Low variance features` | features below the 25th percentile of variance |
| `# Non-zero entries` | how sparse the data is |
| `Mean Log Kernel Density` | mean log-likelihood under a Gaussian KDE; higher is more concentrated |
| `Isomap Reconstruction Error` | geodesic-vs-Euclidean residual of a 2-D Isomap; larger means a more curved manifold |
| `Fractal dimension` | Higuchi fractal dimension, between 1 and 2 |

:::{dropdown} Formulas and references
:icon: code

```{math}
\kappa(\mathbf X) = \frac{\sigma_{\max}(\mathbf X)}{\sigma_{\min}(\mathbf X)}, \qquad
\mathrm{FDR} = \frac{\mathrm{tr}(\mathbf S_B)}{\mathrm{tr}(\mathbf S_W)}, \qquad
\overline{\log p} = \frac{1}{n}\sum_{i}\log\Big(\frac{1}{n h^{d}}\sum_{j}K\big(\tfrac{x_i-x_j}{h}\big)\Big)
```

Fukunaga & Olsen (1971), intrinsic dimension · Fisher (1936) · Golub & Van Loan (2013),
conditioning · Silverman (1986), kernel density · Higuchi (1988), fractal dimension.
:::

(metrics-task)=

## Target spectrum (label-based)

Everything else describes `X`, or `y` only through a classifier's view of it. This
block, from {py:mod}`qbiocode.evaluation.task_spectrum`, asks a different question:
**given the geometry `X` induces, is `y` smooth on it, oscillatory, or noise?** Parity
and checkerboard targets are deterministic, yet they sit almost entirely in the
high-frequency part of the geometry. Low-pass learners fail on them for reasons that
have nothing to do with label noise.

The construction uses the normalized Laplacian of a mutual k-NN graph on the rows of `X`.
Its eigenvectors are a Fourier basis of the data geometry, and expanding the centred
label in that basis gives the target's power at each geometric frequency,
{math}`p_j = (u_j^\top \tilde y)^2 / \lVert \tilde y \rVert^2`.

| Column (`task.`) | What it measures |
|---|---|
| `graph_dirichlet` | the target's mean geometric frequency |
| `graph_hf_mass` | share of target power in modes where neighbours disagree ({math}`\lambda \ge 1`) |
| `graph_spec_entropy` | spread of that power: a few modes, or broadband |
| `graph_bandwidth90` | fraction of modes needed to hold 90 % of the power |
| `diffusion_half_life` | how long the target survives smoothing, relative to the graph's slowest mode |
| `pca_tail_signal` | share of target power outside the PCs that explain 90 % of the variance |
| `h0_fragmentation` | within-class over pooled MST weight; near 1 for contiguous classes, near 2 for interleaved |
| `purity_auc` | chance-corrected neighbourhood label purity, across scales |
| `*_z` | each of the above, as a z-score against random relabelling |

| Target | `graph_hf_mass` | `graph_spec_entropy` | Reading |
|---|---|---|---|
| smooth | low | low | easy geometric target |
| complex smooth | low to moderate | high | many low-frequency modes |
| **structured oscillatory** | **high** | **low to moderate** | parity- or checkerboard-like |
| noisy | high | high | broadband target, or noise |

```{tip}
On wide data, read the `_z` column. In high dimensions random labels look
high-frequency by default, so a raw `graph_hf_mass` means little on its own.
```

(metrics-pymfe)=

## pyMFE meta-features

These are a curated 76 of [pyMFE](https://pymfe.readthedocs.io/)'s roughly 105
meta-features, summarized by mean and SD. The others are left out because they
are undefined, constant, or intractable when there are far more features than samples.
{py:mod}`qbiocode.evaluation.mfe_features` records the reason for each exclusion, and
pyMFE's
[meta-feature table](https://pymfe.readthedocs.io/en/latest/auto_pages/meta_features_description.html)
defines every feature. The headliners:

| Group | Headliners | What they capture |
|---|---|---|
| Complexity (Lorena et al.) | `f1`–`f4`, `l1`–`l3`, `n1`, `n3`, `t3`, `t4`, `c2` | class overlap, distance from linear separability, boundary points, PCA ratios, imbalance |
| Landmarking | `one_nn`, `naive_bayes`, `linear_discr`, `best_node`, `elite_nn` | how well cheap learners do on the data itself |
| General, statistical | `nr_inst`, `nr_attr`, `attr_to_inst`, `cor`, `skewness`, `kurtosis`, `can_cor` | shape, correlation, distribution, class correlation |
| Information, trees, clusters | `class_ent`, `mut_inf`, `tree_depth`, `leaves`, `sil`, `ch` | entropy, an induced tree's shape, how well labels match clusters |

```{tip}
The L family, `f2` and `f4` go to zero when there are more features than samples,
because such data is almost always linearly separable. They are informative on the
8-component embedded passes.
```

:::{dropdown} What changed from the old column set
:icon: history

QProfiler used to write 23 hand-rolled measures. Ten survive in the native block, and
the rest moved to pyMFE:
- `# Samples`, `# Features` and `Feature_Samples_ratio` became `mfe.nr_inst`,
  `mfe.nr_attr` and `mfe.attr_to_inst`;
- `Mutual information`, `Variation`, `Skewness` and `Kurtosis` became their `mfe.`
  counterparts;
- `Total Correlations`, an unnormalized sum that mostly measured width, became the mean
  `mfe.cor.mean`;
- `std_entropy` was always exactly 0 and is gone.
:::

```{seealso}
Lorena et al. (2019), *How complex is your classification problem?*, ACM Computing
Surveys 52(5). Pfahringer et al. (2000), *Meta-learning by landmarking*, ICML.
Alcobaça et al. (2020), *MFE: towards reproducible meta-feature extraction*, JMLR 21(111).
```
