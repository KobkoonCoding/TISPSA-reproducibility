"""Reproduce every table and figure of the paper.  Usage:  python run_all.py [results_dir]"""
import sys, time
from tispsa import sfp, deblurring, classification, ablation, varstep_exact
out = sys.argv[1] if len(sys.argv) > 1 else "results"
STEPS = [("Section 4.1: Table 1", sfp),
         ("Section 4.2: Tables 2-3, Figures 1-3", deblurring),
         ("Section 4.3: Tables 4-5", classification),
         ("Section 4.4: Table 6 (ablation)", ablation),
         ("Section 4.5: Table 7 (variable step sizes)", varstep_exact)]
for name, mod in STEPS:
    t0 = time.time(); print(f"=== {name} ==="); mod.main(out); print(f"--- done in {time.time() - t0:.0f} s\n")
print(f"All tables ({out}/tables/*.tex) and figures ({out}/figures/*.pdf) written.\n"
      "The selection of lambda_TV (python -m tispsa.select_lambda, about 15 min) and of the setting (34)\n"
      "(python -m tispsa.select_setting, about 4 min) is optional.")
