# qHDC smoke-test data

`qhdc_smoke.csv` is a synthetic binary-classification fixture with 64 rows,
four numerical features and the target in the last column. It contains no
biological or patient data. It was generated with:

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=64, n_features=4, n_informative=3, n_redundant=0,
    n_clusters_per_class=1, class_sep=1.2, flip_y=0,
    random_state=42,
)
```

Install QBioCode and the optional hdlib source dependency, then run from the
repository root:

```bash
python -m pip install -e .
python -m pip install -r requirements/requirements-qhdc.txt
qprofiler --config-dir=tutorial/QProfiler/configs --config-name=qhdc_smoke
```

The config compares logistic regression, random forest, PQK and qHDC over two
splits and two feature representations (`none` and `pca`): **16 result rows**.
It uses CPU simulation and needs no credentials. Results go into
`results/qhdc_smoke/<timestamp>/ModelResults.csv`; raw-data complexity metrics,
prediction records and PQK projections are stored beside it. This small fixture
checks the workflow, not whether one model performs better on real-world data.

For an even shorter qHDC-only run:

```bash
qprofiler --config-dir=tutorial/QProfiler/configs --config-name=qhdc_smoke \
  'model=[qhdc]' 'embeddings=[none]' iter=1
```
