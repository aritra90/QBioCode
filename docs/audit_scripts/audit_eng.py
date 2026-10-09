import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from audit_diag import bloch
d = pd.read_csv('/tmp/rep/x_view/eng_zz_n4_gq1_s0.csv'); X = d.iloc[:, :-1].to_numpy(float); y = d.iloc[:, -1].to_numpy(int)
qbc = {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1], "kernel": ["linear", "rbf", "poly", "sigmoid"]}
rng = np.random.default_rng(0)
B_ok = bloch(X, 2, "linear", 0, rng); B_r4 = bloch(X, 4, "linear", 0, rng)
res = {k: [] for k in ["aligned (reps2, raw x)", "pqk defaults (reps4 pairwise)", "MinMax-scaled x (scaling on)", "row-misaligned cache (permuted)"]}
def fit(Ftr, ytr, Fte, yte):
    return RandomizedSearchCV(SVC(random_state=0), qbc, n_iter=40, cv=5, random_state=0).fit(Ftr, ytr).score(Fte, yte)
for r, (tr, te) in enumerate(StratifiedShuffleSplit(100, test_size=0.3, random_state=1).split(X, y)):
    res["aligned (reps2, raw x)"].append(fit(B_ok[tr], y[tr], B_ok[te], y[te]))
    res["pqk defaults (reps4 pairwise)"].append(fit(B_r4[tr], y[tr], B_r4[te], y[te]))
    sc = MinMaxScaler().fit(X[tr]); Bs_tr = bloch(sc.transform(X[tr]), 2, "linear", 0, rng); Bs_te = bloch(sc.transform(X[te]), 2, "linear", 0, rng)
    res["MinMax-scaled x (scaling on)"].append(fit(Bs_tr, y[tr], Bs_te, y[te]))
    p = np.random.default_rng(r).permutation(len(tr))
    res["row-misaligned cache (permuted)"].append(fit(B_ok[tr][p], y[tr], B_ok[te], y[te]))
for k, v in res.items():
    v = np.array(v); print(f"{k:34s} mean {v.mean():.3f}  sd {v.std():.3f}  P(acc<=8/18) = {np.mean(v <= 8/18 + 1e-9):.2f}")
