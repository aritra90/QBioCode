import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from qdata_gen import Pauli, expvals, zz_feature_state

def bloch(X, reps, ent, shots, rng):
    n = X.shape[1]; P = [Pauli(n, {i: s}) for i in range(n) for s in "XYZ"]
    B = np.array([expvals(zz_feature_state(x, reps, ent), P)[0] for x in X])
    if shots: B = 2 * rng.binomial(shots, (1 + B) / 2) / shots - 1
    return B

def fqk(X, reps, ent):
    S = np.array([zz_feature_state(x, reps, ent) for x in X]); return np.abs(S.conj() @ S.T) ** 2

def evaluate(path, reps=2, ent="linear", R=40, seed=0):
    d = pd.read_csv(path); X = d.iloc[:, :-1].to_numpy(float); y = d.iloc[:, -1].to_numpy(int)
    rng = np.random.default_rng(seed)
    B = bloch(X, reps, ent, 0, rng); Bs = bloch(X, reps, ent, 1024, rng); K = fqk(X, reps, ent)
    arms = {k: [] for k in ["LR", "RF", "SVC-RBF(grid)", "QSVC C=0.01", "QSVC C tuned",
                            "PQK oracle (rbf,g=1,C grid)", "PQK QBC-style search", "PQK QBC-style, 1024 shots"]}
    qbc = {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1], "kernel": ["linear", "rbf", "poly", "sigmoid"]}
    for r, (tr, te) in enumerate(StratifiedShuffleSplit(R, test_size=0.3, random_state=seed).split(X, y)):
        cv = min(5, np.bincount(y[tr]).min())
        arms["LR"].append(LogisticRegression(max_iter=5000).fit(X[tr], y[tr]).score(X[te], y[te]))
        arms["RF"].append(RandomForestClassifier(200, random_state=r).fit(X[tr], y[tr]).score(X[te], y[te]))
        arms["SVC-RBF(grid)"].append(GridSearchCV(SVC(), {"C": [0.1, 1, 10, 100], "gamma": [0.1, 1, 10]}, cv=cv).fit(X[tr], y[tr]).score(X[te], y[te]))
        Ktr, Kte = K[np.ix_(tr, tr)], K[np.ix_(te, tr)]
        arms["QSVC C=0.01"].append(SVC(kernel="precomputed", C=0.01).fit(Ktr, y[tr]).score(Kte, y[te]))
        arms["QSVC C tuned"].append(GridSearchCV(SVC(kernel="precomputed"), {"C": [0.01, 0.1, 1, 10, 100]}, cv=cv).fit(Ktr, y[tr]).score(Kte, y[te]))
        arms["PQK oracle (rbf,g=1,C grid)"].append(GridSearchCV(SVC(gamma=1.0), {"C": [0.1, 1, 10, 100]}, cv=cv).fit(B[tr], y[tr]).score(B[te], y[te]))
        for key, F in (("PQK QBC-style search", B), ("PQK QBC-style, 1024 shots", Bs)):
            m = RandomizedSearchCV(SVC(random_state=0), qbc, n_iter=40, cv=cv, random_state=0).fit(F[tr], y[tr])
            arms[key].append(m.score(F[te], y[te]))
    return {k: f"{np.mean(v):.3f} ± {np.std(v):.3f}" for k, v in arms.items()}

if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p.split('/')[-1], evaluate(p), flush=True)
