"""Section 4: selection of the setting (34) of TISPSA and of the two values of lambda_n of the method of Bot et al.
(optional; about 4 min: python -m tispsa.select_setting results).

For every choice of mu_n, rho_n and epsilon_n = 10^k (n+1)^{-p} in the grid of Section 4, and for every lambda_n in
{0.4, 1, 1.4} of the method of Bot et al., the score is the sum of two numbers of evaluations of T:
* the evaluations needed to reach the relative objective gap (45) <= 1e-2 for the first deblurring problem of
  Section 4.2 (fundus image, sigma_blur = 1.5), at most 300 evaluations;
* the mean over the 50 runs of the evaluations needed to reach the residual tolerance (51) for the first dataset of
  Section 4.3 (WDBC), at most 1000 evaluations.
The setting (34) has the smallest score among the TISPSA settings, and lambda_n = 1.4 and lambda_n = 1 have the two
smallest scores of the method of Bot et al.  Writes ``setting_grid.json``.
"""
import itertools, json, os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from . import deblurring as Dm, classification as C
from .algorithms import run_bot, run_tispsa
from .data import load_image

GRID = list(itertools.product((0.5, 0.9), (0.1, 0.5), (0, 1, 2, 3), (1.1, 1.5)))   # mu_n, rho_n, k, p
SETTING_34 = (0.5, 0.1, 1, 1.5)                                                     # epsilon_n = 10/(n+1)^{3/2}
LAM_GRID = (0.4, 1.0, 1.4)


def deblur_instance(outdir):
    """Forward-backward operator, objective, reference value and initial value of the fundus image, sigma_blur = 1.5."""
    lam = Dm.get_lambda(outdir)["std_Fundus"]; x = load_image("Fundus"); sb, sn = Dm.DEGRADATIONS["std"]
    K, Kt, b = Dm.degrade(x, sb, sn); T = Dm.make_T(K, Kt, b, Dm.R_STEP, lam)
    u = b.copy()
    for _ in range(Dm.N_REF): u = T(u)
    F = lambda v: Dm.objective(v, K, b, lam)
    return T, b, F, F(u), F(b)


def wdbc_instances():
    """The 50 forward-backward operators of WDBC (same seeds and folds as tispsa.classification)."""
    X, y = C.load_datasets()["WDBC"]; out = []
    for seed in range(C.SEEDS):
        rng = np.random.default_rng(seed); d = X.shape[1]
        W = rng.uniform(-1, 1, (d, C.M_HIDDEN)); bvec = rng.uniform(-1, 1, C.M_HIDDEN)
        for tr, _ in StratifiedKFold(n_splits=C.FOLDS, shuffle=True, random_state=seed).split(X, y):
            sc = StandardScaler().fit(X[tr]); H = C.elm_features(sc.transform(X[tr]), W, bvec)
            Y = 2.0 * y[tr] - 1.0; n = len(tr); npos = y[tr].sum(); nneg = n - npos
            S = np.where(y[tr] == 1, n / (2 * npos), n / (2 * nneg))
            L = np.linalg.eigvalsh((H.T * S) @ H).max(); out.append(C.make_T(H, Y, S, C.LAM1, 1.0 / L)[0])
    return out


def score(run, per_iter, deb, insts):
    """run(T, x0, N, beta, callback) is TISPSA or the method of Bot et al.; per_iter = evaluations of T per iteration."""
    T, b, F, Fs, F0 = deb; hit = []
    run(T, b, Dm.N_LONG // per_iter, Dm.beta,
        lambda n, v: ((F(v) - Fs) / (F0 - Fs) <= Dm.TOL_REL) and (hit.append(per_iter * n) or True))
    d = hit[0] if hit else None
    ev = []
    for Tc in insts:
        x0 = np.zeros(C.M_HIDDEN); r0 = np.linalg.norm(x0 - Tc(x0)); h = []
        run(Tc, x0, C.N_LONG // per_iter, C.beta,
            lambda k, v: (np.linalg.norm(v - Tc(v)) <= C.TOL_REL * r0) and (h.append(per_iter * k) or True))
        ev.append(h[0] if h else C.N_LONG + 1)
    c = float(np.mean(ev))
    return {"deblur": d, "wdbc_mean": c, "score": None if d is None else d + c}


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    deb = deblur_instance(outdir); insts = wdbc_instances(); res = {"tispsa": [], "bot": []}
    for mu, rho, k, p in GRID:
        eps = lambda n, k=k, p=p: 10.0 ** k / (n + 1) ** p
        run = lambda T, x0, N, beta, cb, mu=mu, rho=rho, eps=eps: run_tispsa(T, x0, N, beta, mu=mu, rho=rho, eps=eps, callback=cb)
        s = {"mu": mu, "rho": rho, "k": k, "p": p, **score(run, 2, deb, insts)}; res["tispsa"].append(s)
        print(f"select TISPSA mu={mu} rho={rho} eps=10^{k}/(n+1)^{p}: deblur {s['deblur']}, WDBC {s['wdbc_mean']:.2f}", flush=True)
    for lam_n in LAM_GRID:
        run = lambda T, x0, N, beta, cb, lam_n=lam_n: run_bot(T, x0, N, beta, lam=lam_n, callback=cb)
        s = {"lambda_n": lam_n, **score(run, 1, deb, insts)}; res["bot"].append(s)
        print(f"select Bot lambda_n={lam_n}: deblur {s['deblur']}, WDBC {s['wdbc_mean']:.2f}", flush=True)
    best = min(s["score"] for s in res["tispsa"] if s["score"] is not None)
    res["tispsa_best_score"] = best
    res["tispsa_best"] = [[s["mu"], s["rho"], s["k"], s["p"]] for s in res["tispsa"] if s["score"] == best]
    res["setting_34_score"] = next(s["score"] for s in res["tispsa"] if (s["mu"], s["rho"], s["k"], s["p"]) == SETTING_34)
    res["bot_ranking"] = [s["lambda_n"] for s in sorted(res["bot"], key=lambda s: s["score"])]
    print("select: smallest TISPSA score", best, "| setting (34)", res["setting_34_score"],
          "| Bot lambda_n ranked", res["bot_ranking"], flush=True)
    json.dump(res, open(os.path.join(outdir, "setting_grid.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
