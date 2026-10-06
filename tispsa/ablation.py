"""Ablation of the two inertial steps (Section 4.4, Table 6).

TISPSA is run in four versions, all with lambda_n = 1, alpha_n = 0.95 and epsilon_n = 10/(n+1)^{3/2}:
without inertia (mu_n = rho_n = 0), first inertial step only (rho_n = 0), second inertial step only (mu_n = 0),
and the setting (34) of the paper (mu_n = 0.5, rho_n = 0.1).

* Deblurring with sigma_blur = 4.0 (Section 4.2): evaluations of T needed to reach the relative objective gap (45) <= 1e-2,
  at most 300 evaluations, for the fundus, CT and MRI images.
* Classification (Section 4.3): mean over the 50 runs of the evaluations of T needed to reduce the fixed point
  residual to one tenth of its initial value (51), at most 1000 evaluations, for WDBC, Pima and Heart.

Writes ``tables/table6_ablation.tex`` and ``ablation_results.json``.  Runtime a few minutes:
``python -m tispsa.ablation results``.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import classification as C
from . import deblurring as Dm
from .algorithms import EPS, MU, RHO, run_tispsa
from .data import load_image

# version: (mu_n, rho_n); two evaluations of T per iteration in every version
VERSIONS = {
    "no_inertia": (0.0, 0.0),
    "first_only": (MU, 0.0),
    "second_only": (0.0, RHO),
    "proposed": (MU, RHO),
}
LABELS = {            # first column "Version of TISPSA" of Table 6
    "no_inertia": r"No inertia, $\mu_n=\rho_n=0$",
    "first_only": r"First inertial step only, $\rho_n=0$",
    "second_only": r"Second inertial step only, $\mu_n=0$",
    "proposed": rf"Both inertial steps, $\mu_n={MU:g}$, $\rho_n={RHO:g}$",     # the setting (34)
}
EVALS_PER_ITER = 2


def deblur_study(outdir):
    """sigma_blur = 4.0: evaluations of T to the relative objective gap TOL_REL (same protocol as Table 3)."""
    lam_sel = Dm.get_lambda(outdir); sb, sn = Dm.DEGRADATIONS["heavy"]; out = {}
    for iname in Dm.IMAGES:
        lam = lam_sel[f"heavy_{iname}"]
        x = load_image(iname); K, Kt, b = Dm.degrade(x, sb, sn); T = Dm.make_T(K, Kt, b, Dm.R_STEP, lam)
        u = b.copy()
        for _ in range(Dm.N_REF):
            u = T(u)
        Fref = Dm.objective(u, K, b, lam); F0 = Dm.objective(b, K, b, lam)
        out[iname] = {}
        for v, (mu, rho) in VERSIONS.items():
            F = {0: F0}
            run_tispsa(T, b, Dm.N_LONG // EVALS_PER_ITER, Dm.beta, mu=mu, rho=rho, eps=EPS,
                       callback=lambda k, uu: F.__setitem__(EVALS_PER_ITER * k, Dm.objective(uu, K, b, lam)) and False)
            ks = sorted(F)
            hit = [k for k in ks if (F[k] - Fref) / (F0 - Fref) <= Dm.TOL_REL]
            out[iname][v] = int(hit[0]) if hit else Dm.N_LONG + 1
            print(f"ablation deblur heavy {iname:6s} {v:12s} evaluations {out[iname][v]}", flush=True)
    return out


def classification_study():
    """Mean evaluations of T to reduce the fixed point residual by the factor TOL_REL (same protocol as Table 5)."""
    out = {}
    for dname, (X, y) in C.load_datasets().items():
        counts = {v: [] for v in VERSIONS}
        for seed in range(C.SEEDS):
            rng = np.random.default_rng(seed); d = X.shape[1]
            W = rng.uniform(-1, 1, (d, C.M_HIDDEN)); bvec = rng.uniform(-1, 1, C.M_HIDDEN)
            skf = StratifiedKFold(n_splits=C.FOLDS, shuffle=True, random_state=seed)
            for tr, _ in skf.split(X, y):
                sc = StandardScaler().fit(X[tr])
                Htr = C.elm_features(sc.transform(X[tr]), W, bvec)
                Ytr = 2.0 * y[tr] - 1.0; n = len(tr); npos = y[tr].sum(); nneg = n - npos
                S = np.where(y[tr] == 1, n / (2 * npos), n / (2 * nneg))
                Lc = np.linalg.eigvalsh((Htr.T * S) @ Htr).max(); T, _ = C.make_T(Htr, Ytr, S, C.LAM1, 1.0 / Lc)
                x0 = np.zeros(C.M_HIDDEN); r0 = np.linalg.norm(x0 - T(x0))
                for v, (mu, rho) in VERSIONS.items():
                    hit = []
                    run_tispsa(T, x0, C.N_LONG // EVALS_PER_ITER, C.beta, mu=mu, rho=rho, eps=EPS,
                               callback=lambda k, u: (np.linalg.norm(u - T(u)) <= C.TOL_REL * r0) and (hit.append(k) or True))
                    counts[v].append(EVALS_PER_ITER * hit[0] if hit else C.N_LONG + 1)   # N_LONG + 1: not reached
        out[dname] = {v: float(np.mean(c)) for v, c in counts.items()}
        print(f"ablation classification {dname:5s} " + " ".join(f"{v} {out[dname][v]:.1f}" for v in VERSIONS), flush=True)
    return out


def reductions(values):
    """Percentage reduction of each version with inertia relative to the version without inertia."""
    base = values["no_inertia"]
    return {v: 100.0 * (1.0 - values[v] / base) for v in VERSIONS if v != "no_inertia"}


def main(outdir):
    os.makedirs(os.path.join(outdir, "tables"), exist_ok=True); t0 = time.time()
    D = deblur_study(outdir); print(f"ablation: deblurring done ({time.time() - t0:.0f}s)", flush=True)
    E = classification_study(); print(f"ablation: classification done ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(outdir, "tables", "table6_ablation.tex"), "w") as f:
        for v in VERSIONS:
            cells = [str(D[i][v]) for i in Dm.IMAGES] + [f"{E[d][v]:.1f}" for d in E]
            f.write(" & ".join([LABELS[v]] + cells) + r" \\" + "\n")
    red = {"deblur_heavy": {i: reductions(D[i]) for i in D}, "classification": {d: reductions(E[d]) for d in E}}
    for app, r in red.items():
        for v in ("proposed", "first_only", "second_only"):
            vals = [r[k][v] for k in r]
            print(f"reduction {app:14s} {v:12s} {min(vals):.1f}% - {max(vals):.1f}%", flush=True)
    res = {"deblur_heavy": D, "classification": E, "reduction_percent": red}
    json.dump(res, open(os.path.join(outdir, "ablation_results.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
