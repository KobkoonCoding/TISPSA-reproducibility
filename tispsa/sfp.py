"""Section 4.1: split feasibility problem in L^2([0, 2*pi]) (Table 1 and the numbers quoted in the text)."""
import json, os
import numpy as np
from .algorithms import run_bot, run_tispsa

TOL, NMAX, TAU, SIGMA = 1e-3, 150, 0.1, 0.01
EVAL_BUDGET = 150                # equal-cost budget of Section 4.1, in evaluations of T
N_NODES = 4001
t = np.linspace(0.0, 2 * np.pi, N_NODES)
w = np.full(N_NODES, t[1] - t[0]); w[0] *= 0.5; w[-1] *= 0.5        # trapezoidal weights
sinv = np.sin(t)


def integ(x): return float(np.dot(w, x))
def ip(x, y): return float(np.dot(w, x * y))
def nrm(x): return np.sqrt(max(ip(x, x), 0.0))


def PC(x):                       # projection onto C = {x : int x <= 1}   [Boţ et al., Sec. 6]
    s = integ(x)
    return x + (1.0 - s) / (2 * np.pi) if s > 1.0 else x.copy()


def PQ(x):                       # projection onto Q = {x : ||x - sin|| <= 4}
    d = nrm(x - sinv)
    return sinv + 4.0 * (x - sinv) / d if d ** 2 > 16.0 else x.copy()


def A(x): return integ(x) * t                       # (Ax)(t) = t * int_0^{2pi} x(s) ds
def Astar(y): return np.full(N_NODES, integ(t * y))   # adjoint of A


def E(x):                        # the function E(x) of Section 4.1
    Ax = A(x)
    return 0.5 * nrm(PC(x) - x) ** 2 + 0.5 * nrm(PQ(Ax) - Ax) ** 2


def pnorm(u): return np.sqrt(nrm(u[0]) ** 2 + nrm(u[1]) ** 2)   # norm on H x G


def T_indicator(u):              # primal-dual mapping T_P1 of problem (P1), eq. (36)
    x, v = u
    p = PC(x - TAU * Astar(v))
    r = A(2 * p - x)
    q = v + SIGMA * r - SIGMA * PQ(v / SIGMA + r)
    return np.stack([p, q])


def T_distance(u):               # primal-dual mapping T_P2 of problem (P2), eq. (37)
    x, v = u
    p = x - TAU * (Astar(v) + x - PC(x))
    r = A(2 * p - x)
    q = v + SIGMA * r - SIGMA * PQ(v / SIGMA + r)
    return np.stack([p, q])




beta_tikh = lambda n: 1.0 - 1.0 / (n + 1)

INITS = {"t^2/10": t ** 2 / 10, "(1/2)e^t": 0.5 * np.exp(t), "e^t+t^2/24": np.exp(t) + t ** 2 / 24}
TEX = {"t^2/10": r"$t^2/10$", "(1/2)e^t": r"$\tfrac12 e^t$", "e^t+t^2/24": r"$e^t+t^2/24$"}

# Methods of Table 1: (algorithm, lambda_n, Tikhonov); EVALS = evaluations of T per iteration.
# Relaxation parameters of the method of Bot et al.  By [Bot, Csetnek, Meier 2019, Corollary 17] with tau = 0.1 and
# sigma = 0.01, for which kappa = min{1/tau, 1/sigma}(1 - sqrt(tau*sigma*||A||^2)) = 2.7923, every
# lambda_n <= 2 - 1/(2*kappa) = 1.8209 is admissible for both problems, eq. (39), so the grid {0.4, 1, 1.4, 1.8}
# of GRID below is admissible throughout.  The two best values by the totals of the counts (38) are
# lambda_n = 1 and lambda_n = 1.4; they are the columns of Table 1.
METHODS = {"bot0": ("bot", 1.0, False), "bot1": ("bot", 1.0, True), "bot14": ("bot", 1.4, True), "tispsa": ("tispsa", None, True)}
EVALS = {"bot0": 1, "bot1": 1, "bot14": 1, "tispsa": 2}
GRID = (0.4, 1.0, 1.4, 1.8)      # values of lambda_n tested for the method of Bot et al. on this problem


def sustained_count(H, tol=TOL, nmax=NMAX):
    """First n such that H[k] <= tol for all n <= k <= nmax, i.e. the iterate reaches the tolerance and
    stays there until the end of the budget (nmax + 1 if this never happens).  The first-hit rule
    "first n with H[n] <= tol" is not used because an iterate can satisfy it once and then leave it again; in
    problem (P1) it can return to values near the plateau E = 17.52 (Section 4.1 of the paper)."""
    ok = np.asarray(H, dtype=float) <= tol
    return next((n for n in range(len(ok)) if ok[n:].all()), nmax + 1)


def count(method, T, u0, nmax=NMAX):
    """Sustained count (38) of E(x_n) <= TOL, the E history, and the norm of the primal-dual iterate after
    EVAL_BUDGET evaluations of T (the common budget of Section 4.1: the methods with one evaluation per
    iteration run EVAL_BUDGET iterations, TISPSA runs EVAL_BUDGET / 2)."""
    H = [E(u0[0])]
    keep = {"u": None}
    nb = EVAL_BUDGET // EVALS[method]
    def cb(n, u):
        H.append(E(u[0]))
        if n == nb:
            keep["u"] = u.copy()
        return False                              # always run the full budget, then apply the sustained rule
    alg, lam, tikh = METHODS[method]
    if alg == "bot":
        u, _ = run_bot(T, u0, nmax, beta_tikh, lam=lam, tikhonov=tikh, callback=cb)
    else:
        u, _ = run_tispsa(T, u0, nmax, beta_tikh, norm=pnorm, callback=cb)
    uB = keep["u"] if keep["u"] is not None else u
    return sustained_count(H, TOL, nmax), H, pnorm(uB)


def fmt(n): return r"$>150$" if n > NMAX else str(n)


def main(outdir):
    os.makedirs(os.path.join(outdir, "tables"), exist_ok=True)
    names = list(INITS)
    res = {}
    # ---- Table 1: problems (P1) and (P2) ----
    rows = []
    for a in names:
        for c in names:
            u0 = np.stack([INITS[a], INITS[c]])
            r = {}
            for mname, T in [("P1", T_indicator), ("P2", T_distance)]:
                r[mname] = {}
                for m in METHODS:
                    n, _, nB = count(m, T, u0)
                    r[mname][m] = n; r[mname][m + "_normB"] = nB
            rows.append((a, c, r)); res[f"{a}|{c}"] = r
            print(f"SFP {a:11s} {c:11s} P1 b=1/Bot1/Bot1.4/TISPSA = {r['P1']['bot0']}/{r['P1']['bot1']}/{r['P1']['bot14']}/{r['P1']['tispsa']}"
                  f" | P2 = {r['P2']['bot0']}/{r['P2']['bot1']}/{r['P2']['bot14']}/{r['P2']['tispsa']}"
                  f" | ||u|| at budget P1 b=1/TISPSA {r['P1']['bot0_normB']:.2f}/{r['P1']['tispsa_normB']:.3f}", flush=True)
    with open(os.path.join(outdir, "tables", "table1_sfp.tex"), "w") as f:
        for a, c, r in rows:
            cells = [fmt(r[m][k]) for m in ("P1", "P2") for k in ("tispsa", "bot1", "bot14", "bot0")]
            f.write(f"{TEX[a]} & {TEX[c]} & " + " & ".join(cells) + " \\\\\n")
    # totals over the nine pairs quoted in Section 4.1 (the column beta_n = 1 has no total: ">150" is not a count)
    tot = {m: {k: sum(r[m][k] for _, _, r in rows) for k in METHODS if k != "bot0"} for m in ("P1", "P2")}
    normsB = {m: {k: float(np.mean([r[m][k + "_normB"] for _, _, r in rows])) for k in METHODS} for m in ("P1", "P2")}
    # in how many of the nine pairs does TISPSA have the smallest norm at the equal-cost budget?
    wins = {m: sum(1 for _, _, r in rows if all(r[m]["tispsa_normB"] < r[m][k + "_normB"] for k in METHODS if k != "tispsa")) for m in ("P1", "P2")}
    # totals of the counts (38) over the whole grid of values of lambda_n, quoted in the text of Section 4.1
    grid_tot = {}
    for mlab, T in (("P1", T_indicator), ("P2", T_distance)):
        grid_tot[mlab] = {}
        for lam in GRID:
            s = 0
            for a in names:
                for c in names:
                    u0 = np.stack([INITS[a], INITS[c]])
                    H = [E(u0[0])]
                    run_bot(T, u0, NMAX, beta_tikh, lam=lam, tikhonov=True,
                            callback=lambda n, u: (H.append(E(u[0])), False)[1])
                    s += sustained_count(H, TOL, NMAX)
            grid_tot[mlab][f"lambda={lam:g}"] = s
    print("SFP grid of lambda_n, totals of (38):", grid_tot, flush=True)
    res["relaxation_grid_totals"] = grid_tot
    res["totals_iterations"] = tot
    # plateau of Section 4.1: number of the 150 iterations in which x_n lies on the boundary int x = 1 of C, for each
    # method in problem (P1), and the value of E on this boundary
    on_boundary = {}
    for a in names:
        for c in names:
            u0 = np.stack([INITS[a], INITS[c]]); on_boundary[f"{a}|{c}"] = {}
            for m, (alg, lam, tikh) in METHODS.items():
                cnt = [0]
                cb = lambda n, u: (cnt.__setitem__(0, cnt[0] + (abs(integ(u[0]) - 1.0) < 1e-8)), False)[1]
                if alg == "bot":
                    run_bot(T_indicator, u0, NMAX, beta_tikh, lam=lam, tikhonov=tikh, callback=cb)
                else:
                    run_tispsa(T_indicator, u0, NMAX // EVALS[m], beta_tikh, norm=pnorm, callback=cb)
                on_boundary[f"{a}|{c}"][m] = cnt[0]
    res["P1_iterations_on_boundary"] = on_boundary
    res["P1_plateau_value"] = 0.5 * (nrm(t - sinv) - 4.0) ** 2
    print("SFP (P1) iterations on the boundary of C:", on_boundary, "plateau value", round(res["P1_plateau_value"], 4), flush=True)
    res["eval_budget"] = EVAL_BUDGET
    res["mean_norm_at_eval_budget"] = normsB
    res["tispsa_smallest_norm_at_eval_budget"] = wins
    print("SFP totals (iterations):", tot,
          f"\nSFP mean ||u|| at {EVAL_BUDGET} evaluations:", normsB,
          "\nTISPSA has the smallest norm at that budget in", wins, "of the nine pairs", flush=True)
    json.dump(res, open(os.path.join(outdir, "sfp_results.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
