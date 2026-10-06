"""Compare a fresh run with the results used in the paper.
* The tables (tables/*.tex) must be identical, character by character.
* Every number in the JSON files -- including all individual runs behind the means and standard deviations and the
  curves behind Figures 2 and 3 -- must agree to a relative tolerance of 1e-6 (absolute 1e-12 near zero).
* The software versions recorded in the protocol of Section 4.5 are not results and are not compared.
Usage: python verify.py results expected_results"""
import json, os, sys

if len(sys.argv) != 3:
    sys.exit("Usage: python verify.py results expected_results")
new, ref = sys.argv[1], sys.argv[2]
RTOL, ATOL = 1e-6, 1e-12


def close(x, y, path):
    if isinstance(x, dict):
        if not isinstance(y, dict) or x.keys() != y.keys():
            return [f"{path}: different keys"]
        return [e for k in x for e in close(x[k], y[k], f"{path}.{k}")]
    if isinstance(x, list):
        if not isinstance(y, list) or len(x) != len(y):
            return [f"{path}: different length"]
        return [e for i, (u, v) in enumerate(zip(x, y)) for e in close(u, v, f"{path}[{i}]")]
    if isinstance(x, float) or isinstance(y, float):
        if x is None or y is None or abs(x - y) > max(RTOL * abs(y), ATOL):
            return [f"{path}: {x} != {y}"]
        return []
    return [] if x == y else [f"{path}: {x} != {y}"]


ok = True
files = ["sfp_results.json", "deblur_results.json", "deblur_figdata.json", "classification_results.json",
         "classification_per_run.json", "ablation_results.json", "varstep_exact_results.json"]
files += [f for f in ("lambda_tv.json", "setting_grid.json") if os.path.exists(os.path.join(new, f))]   # optional steps
for f in files:
    if not os.path.exists(os.path.join(new, f)):
        print(f"{f:32s} MISSING in {new}"); ok = False; continue
    a = json.load(open(os.path.join(new, f))); b = json.load(open(os.path.join(ref, f)))
    if f == "varstep_exact_results.json":
        for d in (a, b):
            d["protocol"] = {k: v for k, v in d["protocol"].items() if k not in ("python", "numpy")}
    errors = close(a, b, f)
    ok &= not errors
    print(f"{f:32s} {'identical' if not errors else 'DIFFERENT: ' + errors[0]}")
for t in sorted(os.listdir(os.path.join(ref, "tables"))):
    p = os.path.join(new, "tables", t)
    same = os.path.exists(p) and open(p, encoding="utf-8").read() == open(os.path.join(ref, "tables", t), encoding="utf-8").read()
    ok &= same
    print(f"tables/{t:25s} {'identical' if same else 'DIFFERENT'}")
print("All results, tables and figure data match the paper." if ok else "Some results differ from the paper (see above).")
sys.exit(0 if ok else 1)
