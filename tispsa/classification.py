"""Section 4.3: medical data classification with ELM + l1 (Tables 4-5).

All budgets are counted in evaluations of the forward-backward operator T: TISPSA uses two per iteration,
the method of Bot et al. one."""
import json, os
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .algorithms import run_bot, run_tispsa
from .data import DATA, load_heart

M_HIDDEN, N_EVAL, N_LONG, LAM1, SEEDS, FOLDS, TOL_REL = 200, 50, 1000, 1e-3, 5, 10, 1e-1
beta = lambda n: 1.0 - 1.0 / (10 * (n + 1))
# methods of Tables 4-5: (algorithm, lambda_n of Bot et al.), evaluations of T per iteration
METHODS = {"tispsa": ("tispsa", None), "bot1": ("bot", 1.0), "bot14": ("bot", 1.4)}
EVALS = {"tispsa": 2, "bot1": 1, "bot14": 1}
TABLE4 = ("tispsa", "bot14")                     # methods of Table 4
LABEL = {"tispsa": "TISPSA", "bot14": r"Bo\c{t} (1.4)"}
KEYS = ["acc", "prec", "rec", "f1", "auc"]


def load_datasets():
    D = {}
    bc = load_breast_cancer(); D["WDBC"] = (bc.data.astype(float), (bc.target == 0).astype(int))      # positive = malignant
    p = pd.read_csv(os.path.join(DATA, "tabular", "pima-indians-diabetes.csv"), header=None).values
    D["Pima"] = (p[:, :-1].astype(float), p[:, -1].astype(int))
    D["Heart"] = load_heart()                                                        # positive = disease (num > 0)
    return D


def sd(v): return float(np.std(np.asarray(v, dtype=float), ddof=1))   # sample standard deviation over the 50 runs


def elm_features(X, W, b): return 1.0 / (1.0 + np.exp(-(X @ W + b)))
def soft(v, t): return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def make_T(H, Y, S, lam1, r):
    """Forward-backward operator (49): a gradient step on the weighted least-squares term followed by soft-thresholding."""
    HS = H.T * S
    T = lambda beta_: soft(beta_ - r * (HS @ (H @ beta_ - Y)), r * lam1)
    obj = lambda beta_: 0.5 * np.sum(S * (H @ beta_ - Y) ** 2) + lam1 * np.sum(np.abs(beta_))
    return T, obj


def metrics(y, score):
    pred = (score > 0).astype(int)
    return dict(acc=accuracy_score(y, pred), prec=precision_score(y, pred, zero_division=0),
                rec=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0), auc=roc_auc_score(y, score))


def run(method, T, x0, n_eval):
    """Run n_eval evaluations of T and return the iterate."""
    alg, lam_n = METHODS[method]
    if alg == "bot":
        x, _ = run_bot(T, x0, n_eval, beta, lam=lam_n)
    else:
        x, _ = run_tispsa(T, x0, n_eval // 2, beta)
    return x


def main(outdir):
    os.makedirs(os.path.join(outdir, "tables"), exist_ok=True)
    D = load_datasets(); R = {}; RUNS = {}
    for name, (X, y) in D.items():
        agg = {m: {k: [] for k in (KEYS if m in TABLE4 else []) + ["evals_tol"]} for m in METHODS}
        runs = []                                   # (seed, fold) of every one of the SEEDS x FOLDS runs
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed); d = X.shape[1]
            W = rng.uniform(-1, 1, (d, M_HIDDEN)); bvec = rng.uniform(-1, 1, M_HIDDEN)
            skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
            for fold, (tr, te) in enumerate(skf.split(X, y)):
                runs.append({"seed": int(seed), "fold": int(fold)})
                sc = StandardScaler().fit(X[tr]); Htr = elm_features(sc.transform(X[tr]), W, bvec); Hte = elm_features(sc.transform(X[te]), W, bvec)
                Ytr = 2.0 * y[tr] - 1.0; n = len(tr); npos = y[tr].sum(); nneg = n - npos
                S = np.where(y[tr] == 1, n / (2 * npos), n / (2 * nneg))            # cost-sensitive weights
                Lc = np.linalg.eigvalsh((Htr.T * S) @ Htr).max(); T, _ = make_T(Htr, Ytr, S, LAM1, 1.0 / Lc)
                x0 = np.zeros(M_HIDDEN)
                for m in TABLE4:
                    mm = metrics(y[te], Hte @ run(m, T, x0, N_EVAL))
                    for k in KEYS: agg[m][k].append(mm[k])
                # tolerance-based stopping: evaluations until the fixed point residual ||x_n - T x_n|| is reduced to
                # TOL_REL times its initial value (the residual is monitored with one extra evaluation, not counted)
                r0 = np.linalg.norm(x0 - T(x0))
                for m, (alg, lam_n) in METHODS.items():
                    hit = []
                    cb = lambda k, u: (np.linalg.norm(u - T(u)) <= TOL_REL * r0) and (hit.append(k) or True)
                    if alg == "bot": run_bot(T, x0, N_LONG, beta, lam=lam_n, callback=cb)
                    else: run_tispsa(T, x0, N_LONG // 2, beta, callback=cb)
                    agg[m]["evals_tol"].append(EVALS[m] * hit[0] if hit else N_LONG + 1)   # N_LONG + 1: not reached
        R[name] = agg; RUNS[name] = runs
        for m in METHODS:
            print(f"clf {name:5s} {m:7s} " + " ".join(f"{k} {np.mean(agg[m][k]):.4f}+-{sd(agg[m][k]):.4f}" for k in KEYS if k in agg[m]) +
                  f" evals_tol {np.mean(agg[m]['evals_tol']):.1f}", flush=True)
    # Table 4 reports mean $\pm$ sample standard deviation (ddof = 1) over the SEEDS x FOLDS = 50 runs after N_EVAL
    # evaluations of T; accuracy is a percentage, so its standard deviation is in percentage points.
    with open(os.path.join(outdir, "tables", "table4_classification.tex"), "w") as f:
        for name in D:
            for m in ["tispsa", "bot14"]:
                g = R[name][m]
                cells = [f"{100 * np.mean(g['acc']):.2f} $\\pm$ {100 * sd(g['acc']):.2f}"]
                cells += [f"{np.mean(g[k]):.4f} $\\pm$ {sd(g[k]):.4f}" for k in ["prec", "rec", "f1", "auc"]]
                f.write(f"{name} & {LABEL[m]} & " + " & ".join(cells) + " \\\\\n")
    with open(os.path.join(outdir, "tables", "table5_tolerance.tex"), "w") as f:      # mean evaluations to the tolerance and range
        for name in D:
            cells = []
            for m in METHODS:
                e = np.array(R[name][m]["evals_tol"]); cells.append(f"{e.mean():.1f} ({e.min()}--{e.max()})")
            f.write(f"{name} & " + " & ".join(cells) + " \\\\\n")
    # paired comparison: number of the 50 runs in which the evaluation count of TISPSA is below that of each other method
    wins = {name: {m: int(np.sum(np.array(R[name]["tispsa"]["evals_tol"]) < np.array(R[name][m]["evals_tol"]))) for m in ("bot1", "bot14")} for name in D}
    print("clf paired wins of TISPSA in evaluations to the tolerance:", wins, flush=True)
    def summarise(g):
        out = {k: float(np.mean(g[k])) for k in g}
        out.update({k + "_std": sd(g[k]) for k in g if k != "evals_tol"})          # Table 4: mean and standard deviation
        out.update({"evals_tol_min": int(min(g["evals_tol"])), "evals_tol_max": int(max(g["evals_tol"]))})   # Table 5
        out["n_runs"] = int(len(g["evals_tol"]))
        return out
    summary = {n: {m: summarise(g) for m, g in agg.items()} for n, agg in R.items()}
    for n in D: summary[n]["paired_wins"] = wins[n]
    json.dump(summary, open(os.path.join(outdir, "classification_results.json"), "w"), indent=1)
    # every individual run, so that the means and standard deviations can be recomputed and audited
    json.dump({n: {"runs": RUNS[n], **{m: {k: [float(v) for v in g[k]] for k in g}
                                        for m, g in agg.items()}} for n, agg in R.items()},
              open(os.path.join(outdir, "classification_per_run.json"), "w"), indent=1)
    return R


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
