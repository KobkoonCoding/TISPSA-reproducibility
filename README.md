# Reproducibility package — TISPSA (version 3.5.0)

Code and data that reproduce **every table and figure** and every number quoted in Section 4 of the paper

> S. Suantai and K. Janngam, *A double inertial S-iteration algorithm with Tikhonov regularization for common fixed
> points and forward-backward splitting, with applications to medical image deblurring and medical data
> classification* (submitted to the Journal of Computational and Applied Mathematics).

The package contains the algorithms compared in the paper (Algorithm 1, TISPSA, and the Tikhonov-regularized
Krasnosel'skii–Mann algorithm of Boţ and Meier (2021), which reduces to that of Boţ, Csetnek and Meier (2019) for a
constant sequence T_n = T), the experiments of Sections 4.1–4.5, the data, and the exact outputs used in the paper
(`expected_results/`). Every output is a table, a figure or a number of the paper, or the data behind one (for
example the 50 individual runs behind a mean and standard deviation).

## 1. Quick start

Python 3.11–3.13 (the pinned versions of NumPy, SciPy and scikit-learn have no wheels for later versions); the
results were produced with Python 3.13.5 on Windows 11, with one CPU thread.

```bash
python -m venv venv && source venv/bin/activate      # optional; on Windows: venv\Scripts\activate
pip install -r requirements.txt                      # pinned versions used for the paper
python run_all.py results                            # about 30 min on a laptop; writes results/
python verify.py results expected_results            # compares the new results with the paper
```

or `bash run_all.sh`, which does the same. The package fixes the number of CPU threads to one when it is imported.

Each experiment can also be run alone: `python -m tispsa.sfp results`, `python -m tispsa.deblurring results`,
`python -m tispsa.classification results`, `python -m tispsa.ablation results`,
`python -m tispsa.varstep_exact results`. Figures 1–3 can be redrawn from saved data without re-running the
experiment by `python -m tispsa.deblurring results --replot` (use `expected_results` in place of `results` to redraw
them from the shipped data).

Two selection steps described in the paper are optional because their results are fixed in the code:

* `python -m tispsa.select_lambda results` (about 15 min) selects λ_TV of Section 4.2 (stored in
  `LAMBDA_TV_DEFAULT` of `tispsa/deblurring.py` and in `expected_results/lambda_tv.json`);
* `python -m tispsa.select_setting results` (about 4 min) recomputes the selection of the setting (34) of TISPSA
  and of the two values of `λ_n` of the method of Boţ et al. in Sections 4.2–4.3 (`expected_results/setting_grid.json`).

`verify.py` checks `lambda_tv.json` and `setting_grid.json` too if they exist in `results/`.

### What `verify.py` checks

* The table files `tables/*.tex` must be identical to `expected_results/tables/*.tex`, character by character. They
  contain the rows of Tables 1–7 of the paper.
* Every number of the JSON files must agree with `expected_results/` to a relative tolerance of 1e-6. This includes
  all individual runs behind the means and standard deviations of Tables 4–7 and the curves behind Figures 2 and 3.

All random elements are seeded (NumPy `default_rng`, scikit-learn `random_state`). The costs are counted as
evaluations of the operators, so they do not depend on the machine.

## 2. What produces what

| Paper item | Script | Output file (in `results/`) |
|---|---|---|
| Table 1 and the numbers of Section 4.1 (totals of the counts (38), totals over the grid of λ_n, norms after 150 evaluations, iterations on the boundary of C in problem (P1)) | `tispsa/sfp.py` | `tables/table1_sfp.tex`, `sfp_results.json` |
| Selected λ_TV (Section 4.2) | `tispsa/select_lambda.py` (optional) | `lambda_tv.json` |
| Selection of the setting (34) and of `λ_n ∈ {1, 1.4}` (Section 4) | `tispsa/select_setting.py` (optional) | `setting_grid.json` |
| Table 2 (PSNR and SSIM after 40 evaluations of T; reference solution) and its residual quoted in Section 4.2 | `tispsa/deblurring.py` | `tables/table2_deblur.tex`, `deblur_results.json` |
| Table 3 (evaluations of T needed to reach the stopping criterion (45)) | `tispsa/deblurring.py` | `tables/table3_tolerance.tex`, `deblur_results.json` |
| Figure 1 (restored images), Figure 2 (PSNR and SSIM against evaluations of T), Figure 3 (relative objective gap), σ_blur = 4.0 | `tispsa/deblurring.py` | `figures/fig1_deblur_visual.pdf`, `figures/fig2_deblur_quality.pdf`, `figures/fig3_deblur_gap.pdf`, `deblur_figdata.json`, `deblur_figimages.npz` |
| Table 4 (classification metrics after 50 evaluations, mean ± standard deviation over 50 runs) | `tispsa/classification.py` | `tables/table4_classification.tex`, `classification_results.json`, `classification_per_run.json` |
| Table 5 (evaluations of T needed to reach the stopping criterion (51), mean and range) and the run-by-run comparisons of Section 4.3 | `tispsa/classification.py` | `tables/table5_tolerance.tex`, `classification_results.json`, `classification_per_run.json` |
| Table 6 (ablation study) and the percentages of Section 4.4 | `tispsa/ablation.py` | `tables/table6_ablation.tex`, `ablation_results.json` |
| Table 7 (variable step sizes) and the run-by-run comparisons of Section 4.5 | `tispsa/varstep_exact.py` | `tables/table_varstep_exact.tex`, `varstep_exact_results.json` |

## 3. Package layout

```
tispsa/algorithms.py      run_bot (scheme (3) of the paper; (2) for T_n = T, (33) for forward-backward
                          operators with variable step sizes) and run_tispsa (Algorithm 1, eqs. (8)-(13));
                          LAM, ALPHA, MU, RHO, EPS are the setting (34) of TISPSA
tispsa/sfp.py             Section 4.1: split feasibility problems (P1) and (P2) in L^2([0,2π]), mappings (36)-(37), count (38)
tispsa/deblurring.py      Section 4.2: TV deblurring, forward-backward operator (43), Chambolle TV prox (20 inner iterations)
tispsa/classification.py  Section 4.3: ELM with ℓ1 regularization and class weights, operator (49), 5 x 10-fold cross-validation
tispsa/ablation.py        Section 4.4: Table 6
tispsa/varstep_exact.py   Section 4.5: variable step sizes (52), Table 7
tispsa/select_lambda.py   Section 4.2: selection of λ_TV (optional)
tispsa/select_setting.py  Section 4: selection of the setting (34) and of λ_n (optional)
tispsa/data.py            data loading and preprocessing; re-download and SHA-256 verification of the data
data/                     raw and preprocessed images, clinical CSV files, SHA256SUMS, SOURCES.md (provenance and licenses)
expected_results/         the tables, figures and JSON files used in the paper, and the logs of the run
run_all.py, run_all.sh, verify.py, requirements.txt, CITATION.cff, LICENSE, CHANGELOG.md
```

## 4. Experimental settings (identical to the paper)

* **Cost measure.** Sections 4.2–4.4 count evaluations of `T` and Section 4.5 evaluations of the mappings `T_n`.
  TISPSA evaluates them twice per iteration and the algorithm of Boţ and Meier once. The evaluations of `T` needed to
  monitor the stopping criterion (51) are not counted. Section 4.1 reports iteration counts and compares the methods
  after 150 evaluations of `T` (150 iterations of the methods with one evaluation, 75 of TISPSA).
* **TISPSA, setting (34).** `λ_n = 1`, `α_n = 0.95`, `μ_n = 0.5`, `ρ_n = 0.1`, `ε_n = 10/(n+1)^{3/2}` (constants
  in `tispsa/algorithms.py`), the same in all experiments; start `x_1 = x_0`, `w_0 = x_0`. The setting was chosen
  from the grid `μ_n ∈ {0.5, 0.9}`, `ρ_n ∈ {0.1, 0.5}`, `ε_n = 10^k/(n+1)^p`, `k ∈ {0, 1, 2, 3}`, `p ∈ {1.1, 1.5}`.
  Its score, the number of evaluations of `T` needed to reach (45) for the fundus image with σ_blur = 1.5 plus the
  mean number needed to reach (51) for WDBC, is 12 + 7.88, the smallest score in the grid.
* **Algorithm of Boţ and Meier.** `λ_n ∈ {1, 1.4}`, the two best values of the grid `{0.4, 1, 1.4, 1.8}` in Section
  4.1 (ranked by the sums of the counts (38)) and of `{0.4, 1, 1.4}` in Sections 4.2–4.3 (ranked by the same score as
  the setting of TISPSA). In Section 4.1 the method is also run with `β_n = 1` and `λ_n = 1`.
* **Section 4.1.** Data of Boţ, Csetnek and Meier (2019, Section 6): `(Ax)(t) = t ∫ x`, `‖A‖² = 16π⁴/3`,
  `τ = 0.1`, `σ = 0.01`, 4001 nodes on `[0, 2π]`, trapezoidal rule, `β_n = 1 − 1/(n+1)`, the nine pairs of initial
  points of their Tables 1–2. Count (38): the smallest `n` such that `E(x_k) ≤ 1e-3` for all `n ≤ k ≤ 150`.
* **Section 4.2.** Images: retinal fundus (green channel of `skimage.data.retina()`), chest CT, brain MRI, resized to
  256x256. Periodic Gaussian blur with unit sum (`‖K‖ = 1`), `σ_blur = 1.5` with `σ_noise = 0.015` and
  `σ_blur = 4.0` with `σ_noise = 0.04`; the noise is `σ_noise` times one fixed standard normal field
  (`default_rng(1)`). Step `r = 1`, `β_n = 1 − 1/(10(n+1))`, `x_0 = b`. The TV prox is computed by 20 iterations of
  Chambolle's algorithm (dual step 0.248, started at zero). The reference solution `u_ref` is computed by 1000
  forward-backward iterations. Budgets: 40 evaluations (Table 2, Figure 1) and 300 evaluations (Table 3, Figures 2–3,
  Table 6; Figure 2 shows the first 100). SSIM: scikit-image `structural_similarity` with data range 1 on the image
  clipped to [0, 1].
* **Section 4.3.** ELM with 200 sigmoidal hidden nodes (input weights and biases uniform in `[−1, 1]`, seeds 0–4,
  also used as `random_state` of `StratifiedKFold(shuffle=True)`), standardization with the training set, class
  weights `m/(2m_+)` and `m/(2m_-)`, `λ_ℓ1 = 1e-3`, `r = 1/‖HᵀSH‖`, `β_n = 1 − 1/(10(n+1))`, `ω_0 = 0`, 50
  evaluations of `T` (Table 4). Tables 5 and 6 count the evaluations until `‖ω_n − Tω_n‖ ≤ 0.1 ‖ω_0 − Tω_0‖`
  (at most 1000).
* **Section 4.4.** TISPSA without inertia (`μ_n = ρ_n = 0`), with the first inertial step only (`ρ_n = 0`), with the
  second inertial step only (`μ_n = 0`) and with the setting (34); deblurring with σ_blur = 4.0 and classification.
* **Section 4.5.** The same 150 training sets as Section 4.3, step sizes `r_n = (1 + 0.2/(n+1))/L`, `n ≥ 1`,
  `L = ‖HᵀSH‖`; stopping criterion (51) with the operator `T` of (49) and step `1/L`; at most 1000 evaluations of
  the mappings `T_n`.

## 5. Data and licenses

`data/SOURCES.md` lists for every data file its source, its original name, the package file it became, its
SHA-256 and its license; `data/SHA256SUMS` repeats the hashes. `python -m tispsa.data --verify` (or
`sha256sum -c data/SHA256SUMS`) checks the bundled files. `python -m tispsa.data` downloads the raw files again and
rebuilds the preprocessed images; the rebuilt images have the same pixels as the bundled ones, but their PNG bytes can
differ, so the hash check applies to the bundled files.

The code is released under the MIT license (`LICENSE`). The data files keep the licenses of their sources (CC0 for
the fundus image, MIT for the MRI dataset, CC BY-NC 4.0 / CC BY-NC-SA for the CT image, CC BY 4.0 and public domain
for the clinical datasets); see `data/SOURCES.md`.

## 6. Citation

Please cite the paper and this package; see `CITATION.cff`. The package is archived on Zenodo,
https://doi.org/10.5281/zenodo.23185302 (all versions), and its source is at
https://github.com/KobkoonCoding/TISPSA-reproducibility.
