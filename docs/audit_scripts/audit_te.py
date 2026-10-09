import numpy as np, warnings, types
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from qdata_gen import pauli_sum, ising_terms, level_spacing_ratio, pool_local, sparse_observable, expvals, walsh_degree_profile
def lsr(n, seed, w=0.1):
    rng = np.random.default_rng(seed)
    J = rng.uniform(1-w,1+w,n-1); h = rng.uniform(1-w,1+w,n); g = rng.uniform(0.5-w,0.5+w,n)
    return level_spacing_ratio(np.linalg.eigvalsh(pauli_sum(n, ising_terms(n,J,h,g=g,sign=1.0)).toarray()))
for n in (6, 8, 10):
    r = [lsr(n, s) for s in range(10)]
    print(f"<r> n={n:2d}: mean {np.mean(r):.3f}  min {np.min(r):.3f}  max {np.max(r):.3f}  (10 seeds)")
# effective degree and RF accuracy vs tau, several seeds (same construction as fam_te)
n, N, taus = 8, 200, [0.25, 0.5, 1, 2, 4]
rows = []; nonmono = 0
for seed in range(8):
    rng = np.random.default_rng(seed); w = 0.1
    J = rng.uniform(1-w,1+w,n-1); h = rng.uniform(1-w,1+w,n); g = rng.uniform(0.5-w,0.5+w,n)
    E, V = np.linalg.eigh(pauli_sum(n, ising_terms(n,J,h,g=g,sign=1.0)).toarray())
    pool = pool_local(n); S, a = sparse_observable(pool, 4, rng)
    xs = np.sort(rng.choice(1<<n, N, replace=False)); bits = (xs[:,None] >> np.arange(n)) & 1
    degs = []
    for tau in taus:
        U = (V*np.exp(-1j*E*tau)) @ V.T
        F_all = expvals(U, pool)[:, S] @ a
        _, d = walsh_degree_profile(F_all, n); degs.append(d)
        F = F_all[xs]; y = (F > np.median(F)).astype(int)
        acc = cross_val_score(RandomForestClassifier(200, random_state=0), bits, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
        rows.append((seed, tau, d, acc))
    nonmono += any(np.diff(degs) < 0)
rows = np.array(rows)
print(f"seeds with non-monotone effective degree in tau: {nonmono}/8")
print("Spearman(acc, tau)        = %.2f" % spearmanr(rows[:,3], rows[:,1]).correlation)
print("Spearman(acc, eff. degree)= %.2f" % spearmanr(rows[:,3], rows[:,2]).correlation)
viol = sum(any(np.diff(rows[rows[:,0]==s][:,3]) > 0.02) for s in range(8))
print(f"seeds where RF accuracy rises by >0.02 somewhere along tau: {viol}/8")
