"""Section 4.5: variable step sizes under both convergence results (Table 7).

Uses the existing ELM instances and exact soft-thresholding proximal operator.
The step sizes are r_n = (1 + 0.2/(n+1))/L, n >= 1. Its total variation is
0.1/L, and both values lambda_n = 1 and lambda_n = 1.4 of Bot and Meier satisfy lambda_n <= 2-r_n*L/2, since 2-r_n*L/2 >= 1.45.
The stopping residual always uses T_ref with step 1/L, independently of n.
Update, monitoring and initial-residual evaluations are counted separately.
See Bot--Meier (2021), Theorem 2 (scheme (33) and the conditions of Remark 3.5), and Corollary 3.4 of the paper.
"""

import json
import platform
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import classification as clf
from .algorithms import ALPHA, LAM, MU, RHO, run_bot, run_tispsa

METHODS = {"tispsa": (2, None), "bm1": (1, 1.0), "bm14": (1, 1.4)}
MAX_UPDATES = 1000


def step_size(n: int, lipschitz: float) -> float:
    """Positive, decreasing step sizes with limit 1/L and first value 1.1/L."""
    if n < 1 or not np.isfinite(lipschitz) or lipschitz <= 0:
        raise ValueError("n must be positive and L must be finite and positive")
    return (1.0 + 0.2 / (n + 1)) / lipschitz


def run_instance(H: np.ndarray, Y: np.ndarray, S: np.ndarray) -> dict:
    """Run all methods on the same objective, start and stopping residual."""
    lipschitz = float(np.linalg.eigvalsh((H.T * S) @ H).max())
    monitor, _ = clf.make_T(H, Y, S, clf.LAM1, 1.0 / lipschitz)
    x0 = np.zeros(H.shape[1])
    initial_residual = float(np.linalg.norm(x0 - monitor(x0)))
    if initial_residual <= 0 or not np.isfinite(initial_residual):
        raise ValueError("The protocol requires a finite, positive initial residual")
    results = {}
    for method, (per_iteration, relaxation) in METHODS.items():
        counts = {"update": 0, "monitor": 0}
        last_residual = initial_residual

        def family(n: int, relaxation=relaxation, counts=counts):
            r = step_size(n, lipschitz)
            if relaxation is not None and relaxation > 2.0 - r * lipschitz / 2.0:
                raise AssertionError("BM relaxation bound violated")
            operator, _ = clf.make_T(H, Y, S, clf.LAM1, r)

            def counted(x: np.ndarray) -> np.ndarray:
                counts["update"] += 1
                return operator(x)

            return counted

        def stop(n: int, x: np.ndarray, counts=counts, method=method) -> bool:
            nonlocal last_residual
            counts["monitor"] += 1
            last_residual = float(np.linalg.norm(x - monitor(x)))
            if not np.isfinite(last_residual):
                raise FloatingPointError(f"Non-finite residual for {method} at {n}")
            return last_residual <= clf.TOL_REL * initial_residual

        if method == "tispsa":
            _, iterations = run_tispsa(
                monitor, x0, MAX_UPDATES // per_iteration, clf.beta,
                callback=stop, Tn=family,
            )
        else:
            _, iterations = run_bot(
                monitor, x0, MAX_UPDATES, clf.beta, lam=relaxation,
                callback=stop, Tn=family,
            )
        if counts != {"update": iterations * per_iteration, "monitor": iterations}:
            raise AssertionError("Operator-call accounting does not match iterations")
        results[method] = {
            "converged": last_residual <= clf.TOL_REL * initial_residual,
            "iterations": iterations,
            "update_evals": counts["update"],
            "relative_residual": last_residual / initial_residual,
        }
    return {"lipschitz": lipschitz, "initial_residual": initial_residual, **results}


def main(outdir: str) -> dict:
    """Use the 150 existing seeded training sets of Section 4.3, without tuning the step sizes."""
    out = Path(outdir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    runs = {}
    with threadpool_limits(limits=1):
        for name, (X, y) in clf.load_datasets().items():
            records = []
            for seed in range(clf.SEEDS):
                rng = np.random.default_rng(seed)
                W = rng.uniform(-1, 1, (X.shape[1], clf.M_HIDDEN))
                bias = rng.uniform(-1, 1, clf.M_HIDDEN)
                folds = StratifiedKFold(clf.FOLDS, shuffle=True, random_state=seed)
                for fold, (train, _) in enumerate(folds.split(X, y)):
                    scaler = StandardScaler().fit(X[train])
                    H = clf.elm_features(scaler.transform(X[train]), W, bias)
                    target = 2.0 * y[train] - 1.0
                    n = len(train)
                    positive = int(y[train].sum())
                    weights = np.where(
                        y[train] == 1, n / (2 * positive), n / (2 * (n - positive)),
                    )
                    records.append({"seed": seed, "fold": fold,
                                    **run_instance(H, target, weights)})
            runs[name] = records
            print(f"Exact variable-step: {name}, {len(records)} paired runs", flush=True)

    if not all(row[m]["converged"] for records in runs.values() for row in records for m in METHODS):
        with (out / "varstep_exact_failed_runs.json").open("w", encoding="utf-8") as handle:
            json.dump(runs, handle, indent=2, allow_nan=False)
        raise RuntimeError("Unsuccessful runs saved; no table of counts to tolerance was produced")
    summary = {}
    for name, records in runs.items():
        summary[name] = {}
        for method in METHODS:
            summary[name][method] = {
                "successes": sum(row[method]["converged"] for row in records),
                **{key: float(np.mean([row[method][key] for row in records]))
                   for key in ("update_evals",)},
            }
        summary[name]["paired_wins"] = {
            baseline: {key: sum(row["tispsa"][key] < row[baseline][key] for row in records)
                       for key in ("update_evals",)}
            for baseline in ("bm1", "bm14")
        }
    report = {
        "protocol": {
            "schedule": "r_n=(1+0.2/(n+1))/L, n>=1",
            "monitor": "||x-T_(1/L)(x)|| <= 0.1 ||x0-T_(1/L)(x0)||",
            "max_update_evals": MAX_UPDATES,
            "inertia": {"lambda": LAM, "alpha": ALPHA, "mu": MU, "rho": RHO,
                        "epsilon": "10/(n+1)^1.5"},
            "beta": "1-1/(10(n+1))",
            "threads": 1,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "summary": summary,
        "runs": runs,
    }
    with (out / "varstep_exact_results.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    lines = []
    for name, values in summary.items():
        cells = [f"{values[m]['update_evals']:.2f}" for m in METHODS]   # Table 7: evaluations of the mappings T_n
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    (out / "tables/table_varstep_exact.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
