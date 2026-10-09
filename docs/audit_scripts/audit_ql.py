import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.svm import SVC
from audit_diag import bloch
d = pd.read_csv('/tmp/rep/x_view/ql_zz_n4_tau1_s0.csv'); X = d.iloc[:, :-1].to_numpy(float); y = d.iloc[:, -1].to_numpy(int)
qbc = {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1], "kernel": ["linear", "rbf", "poly", "sigmoid"]}
rng = np.random.default_rng(0)
for lab, B in (("aligned reps2", bloch(X, 2, "linear", 0, rng)), ("reps4 pairwise", bloch(X, 4, "linear", 0, rng))):
    a = [RandomizedSearchCV(SVC(random_state=0), qbc, n_iter=40, cv=5, random_state=0).fit(B[tr], y[tr]).score(B[te], y[te])
         for tr, te in StratifiedShuffleSplit(60, test_size=0.3, random_state=1).split(X, y)]
    a = np.array(a); print(f"ql_zz PQK {lab:15s} mean {a.mean():.3f} sd {a.std():.3f}  P(acc >= 12/18) = {np.mean(a >= 12/18 - 1e-9):.2f}")
