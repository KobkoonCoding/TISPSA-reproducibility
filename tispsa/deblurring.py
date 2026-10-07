"""Section 4.2: medical image deblurring (Tables 2-3, Figures 1-3).

All budgets are counted in evaluations of the forward-backward operator T: TISPSA uses two per iteration,
the method of Bot et al. one."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_fn
from .algorithms import run_bot, run_tispsa
from .data import load_image

N_EVAL, N_LONG, N_REF, R_STEP, TOL_REL = 40, 300, 1000, 1.0, 1e-2      # budget, cap and reference in evaluations of T
DEGRADATIONS = {"std": (1.5, 0.015), "heavy": (4.0, 0.04)}             # (sigma_blur, sigma_noise); "std": sigma_blur = 1.5, "heavy": sigma_blur = 4.0
# lambda_TV selected per image and value of sigma_blur by tispsa/select_lambda.py
# (largest PSNR of the minimizer of the TV model on the grid {0.0005,0.001,0.002,0.004,0.008,0.016})
LAMBDA_TV = None   # filled below from results/lambda_tv.json or the built-in table
LAMBDA_TV_DEFAULT = {"std_Fundus": 0.004, "std_CT": 0.002, "std_MRI": 0.001,
                     "heavy_Fundus": 0.004, "heavy_CT": 0.004, "heavy_MRI": 0.004}
beta = lambda n: 1.0 - 1.0 / (10 * (n + 1))
IMAGES = ["Fundus", "CT", "MRI"]      # the three test images of Section 4.2
Q_FROM, Q_MAX = 10, 100               # Figure 2: evaluations shown; the y-range is fitted to evaluations >= Q_FROM
# methods of Tables 2-3: (algorithm, lambda_n of Bot et al.), evaluations of T per iteration
METHODS = {"tispsa": ("tispsa", None), "bot1": ("bot", 1.0), "bot14": ("bot", 1.4)}
EVALS = {"tispsa": 2, "bot1": 1, "bot14": 1}


def make_blur(shape, sigma):
    """Gaussian blur with periodic boundary conditions (via FFT); K is self-adjoint with ||K|| = 1."""
    n1, n2 = shape; y, x = np.mgrid[0:n1, 0:n2]; y = np.minimum(y, n1 - y); x = np.minimum(x, n2 - x)
    h = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)); h /= h.sum(); H = np.fft.fft2(h)
    K = lambda v: np.real(np.fft.ifft2(H * np.fft.fft2(v)))
    Kt = lambda v: np.real(np.fft.ifft2(np.conj(H) * np.fft.fft2(v)))
    return K, Kt


def psnr(x, ref): return 10 * np.log10(1.0 / np.mean((x - ref) ** 2))
def ssim(x, ref): return ssim_fn(np.clip(x, 0, 1), ref, data_range=1.0)


def grad(u):
    gx = np.zeros_like(u); gy = np.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]; gy[:-1, :] = u[1:, :] - u[:-1, :]; return gx, gy


def div(px, py):
    d = np.zeros_like(px)
    d[:, 0] += px[:, 0]; d[:, 1:-1] += px[:, 1:-1] - px[:, :-2]; d[:, -1] -= px[:, -2]
    d[0, :] += py[0, :]; d[1:-1, :] += py[1:-1, :] - py[:-2, :]; d[-1, :] -= py[-2, :]; return d


def tv(u): gx, gy = grad(u); return float(np.sum(np.sqrt(gx ** 2 + gy ** 2)))


def prox_tv(v, tau, iters=20, step=0.248):
    """prox of tau*TV (isotropic) at v, by Chambolle's (2004) dual projection algorithm."""
    px = np.zeros_like(v); py = np.zeros_like(v)
    for _ in range(iters):
        d = div(px, py); gx, gy = grad(d - v / tau)
        n = np.sqrt(gx ** 2 + gy ** 2); px = (px + step * gx) / (1 + step * n); py = (py + step * gy) / (1 + step * n)
    return v - tau * div(px, py)


def make_T(K, Kt, b, r, lam):
    """Forward-backward operator (43): T = prox_{r lam TV}(I - r K^*(K . - b)), a gradient step on the data term
    followed by a TV denoising step; the TV prox is approximated by 20 iterations of Chambolle's algorithm."""
    return lambda x: prox_tv(x - r * Kt(K(x) - b), r * lam)


def objective(x, K, b, lam): return 0.5 * float(np.sum((K(x) - b) ** 2)) + lam * tv(x)


def degrade(x, sigma_blur, sigma_noise, seed=1):
    K, Kt = make_blur(x.shape, sigma_blur)
    b = K(x) + sigma_noise * np.random.default_rng(seed).standard_normal(x.shape)
    return K, Kt, b


def get_lambda(outdir):
    global LAMBDA_TV
    path = os.path.join(outdir, "lambda_tv.json")
    if os.path.exists(path):
        LAMBDA_TV = {k: v["lambda_tv"] for k, v in json.load(open(path)).items()}
    else:
        LAMBDA_TV = dict(LAMBDA_TV_DEFAULT)
    return LAMBDA_TV


def run_long(method, T, b, x, K, lam, n_eval):
    """Run n_eval evaluations of T; record PSNR, SSIM and objective after every evaluation (per iteration for
    TISPSA, i.e. every two evaluations) and the iterate after N_EVAL evaluations."""
    alg, lam_n = METHODS[method]; ev = EVALS[method]
    P = {0: psnr(b, x)}; S = {0: ssim(b, x)}; F = {0: objective(b, K, b, lam)}; store = {}
    def cb(n, u):
        k = ev * n; P[k] = psnr(u, x); S[k] = ssim(u, x); F[k] = objective(u, K, b, lam)
        if k == N_EVAL: store["x"] = u.copy()
        return False
    if alg == "bot":
        run_bot(T, b, n_eval, beta, lam=lam_n, callback=cb)
    else:
        run_tispsa(T, b, n_eval // 2, beta, callback=cb)
    return P, S, F, store


def main(outdir):
    os.makedirs(os.path.join(outdir, "tables"), exist_ok=True); os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)
    lam_sel = get_lambda(outdir); res = {}; rec = {}
    lab = {"std": "1.5", "heavy": "4.0"}            # column sigma_blur of Tables 2 and 3
    for dname, (sb, sn) in DEGRADATIONS.items():
        for iname in IMAGES:
            key = f"{dname}_{iname}"; lam = lam_sel[key]
            x = load_image(iname); K, Kt, b = degrade(x, sb, sn); T = make_T(K, Kt, b, R_STEP, lam)
            # reference solution (minimizer of the TV model): N_REF plain forward-backward iterations
            u = b.copy()
            for _ in range(N_REF): u = T(u)
            Fstar = objective(u, K, b, lam); F0 = objective(b, K, b, lam)
            o = {"lambda_tv": lam, "psnr_in": psnr(b, x), "psnr_conv": psnr(u, x), "ssim_conv": ssim(u, x),
                 "F_star": Fstar, "ref_residual": float(np.linalg.norm(u - T(u)) / np.linalg.norm(b - T(b)))}
            for m in METHODS:
                P, S, F, st = run_long(m, T, b, x, K, lam, N_LONG)
                ks = sorted(F); gap = {k: (F[k] - Fstar) / (F0 - Fstar) for k in ks}
                hit = [k for k in ks if gap[k] <= TOL_REL]
                o[m] = {"psnr": P[N_EVAL], "ssim": ssim(st["x"], x),
                        "traj": [P[k] for k in ks], "ssim_traj": [S[k] for k in ks], "gap": [gap[k] for k in ks], "evals_axis": ks,
                        "evals_tol": int(hit[0]) if hit else None, "iters_tol": (int(hit[0]) // EVALS[m]) if hit else None}
                if dname == "heavy": rec[(dname, iname, m)] = st["x"]
            if dname == "heavy": rec[(dname, iname, "blur")] = b
            res[key] = o
            print(f"deblur sigma_blur={lab[dname]} {iname:5s} lam {lam} PSNR_in {o['psnr_in']:.2f} | " +
                  " | ".join(f"{m} {o[m]['psnr']:.2f}/{o[m]['ssim']:.4f} tol {o[m]['evals_tol']}" for m in METHODS) +
                  f" | ref {o['psnr_conv']:.2f}/{o['ssim_conv']:.4f} res {o['ref_residual']:.1e}", flush=True)
    def fmt(v): return str(v) if v is not None else f"$>{N_LONG}$"
    with open(os.path.join(outdir, "tables", "table2_deblur.tex"), "w") as f:      # PSNR and SSIM after N_EVAL evaluations
        for d in DEGRADATIONS:
            for im in IMAGES:
                o = res[f"{d}_{im}"]
                f.write(f"{im} & {lab[d]} & {o['psnr_in']:.2f} & " + " & ".join(f"{o[m]['psnr']:.2f}" for m in METHODS) + f" & {o['psnr_conv']:.2f} & "
                        + " & ".join(f"{o[m]['ssim']:.4f}" for m in METHODS) + f" & {o['ssim_conv']:.4f} \\\\\n")
    with open(os.path.join(outdir, "tables", "table3_tolerance.tex"), "w") as f:   # iterations and evaluations to the tolerance
        for d in DEGRADATIONS:
            for im in IMAGES:
                o = res[f"{d}_{im}"]
                f.write(f"{im} & {lab[d]} & " + " & ".join(f"{fmt(o[m]['iters_tol'])} & {fmt(o[m]['evals_tol'])}" for m in METHODS) + " \\\\\n")
    json.dump({k: {kk: ({a: b for a, b in vv.items() if a not in ("traj", "ssim_traj", "gap", "evals_axis")} if isinstance(vv, dict) else vv) for kk, vv in v.items()} for k, v in res.items()},
              open(os.path.join(outdir, "deblur_results.json"), "w"), indent=1)
    # figure data (trajectories, gaps, images for sigma_blur = 4.0) are saved so that the figures can be redrawn
    # without re-running the experiment: python -m tispsa.deblurring results --replot
    figdata = {k: {"psnr_in": o["psnr_in"], "psnr_conv": o["psnr_conv"],
                   **{m: {"psnr": o[m]["psnr"], "ssim": o[m]["ssim"], "traj": o[m]["traj"], "ssim_traj": o[m]["ssim_traj"],
                          "gap": o[m]["gap"], "evals_axis": o[m]["evals_axis"]} for m in METHODS}}
               for k, o in res.items() if k.startswith("heavy_")}
    json.dump(figdata, open(os.path.join(outdir, "deblur_figdata.json"), "w"))
    np.savez_compressed(os.path.join(outdir, "deblur_figimages.npz"), **{f"{im}_{m}": rec[("heavy", im, m)] for im in IMAGES for m in ("blur", "tispsa", "bot1", "bot14")})
    make_figures(outdir, figdata, {(im, m): rec[("heavy", im, m)] for im in IMAGES for m in ("blur", "tispsa", "bot1", "bot14")})
    return res


def make_figures(outdir, figdata, rec):
    """Figure 1 (restored images after N_EVAL evaluations, sigma_blur = 4.0), Figure 2 (PSNR and SSIM against evaluations)
    and Figure 3 (relative objective gap)."""
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)
    # ---- Figure 1: restored images, sigma_blur = 4.0 ----
    names = {"tispsa": "TISPSA", "bot1": "Boţ et al., $\\lambda_n=1$", "bot14": "Boţ et al., $\\lambda_n=1.4$"}
    fig, ax = plt.subplots(len(IMAGES), 5, figsize=(14, 11.0))
    for i, im in enumerate(IMAGES):
        o = figdata[f"heavy_{im}"]; x = load_image(im)
        panels = [("Original", x), (f"Observed image $b$\nPSNR {o['psnr_in']:.2f} dB", rec[(im, "blur")])]
        panels += [(f"{names[m]}\nPSNR {o[m]['psnr']:.2f} dB\nSSIM {o[m]['ssim']:.4f}", rec[(im, m)]) for m in ("tispsa", "bot1", "bot14")]
        for j, (t, img) in enumerate(panels):
            ax[i, j].imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax[i, j].set_title(t, fontsize=19); ax[i, j].axis("off")
        ax[i, 0].text(-0.06, 0.5, im, transform=ax[i, 0].transAxes, rotation=90, va="center", ha="right", fontsize=22)
    plt.tight_layout(rect=(0, 0, 1, 0.985), h_pad=3.0, w_pad=0.6)
    plt.savefig(os.path.join(outdir, "figures", "fig1_deblur_visual.pdf"), dpi=300, metadata={"CreationDate": None}); plt.close()
    style = {"tispsa": ("b", "-", "TISPSA"), "bot1": ("m", "-.", r"Boţ et al., $\lambda_n=1$"), "bot14": ("r", "--", r"Boţ et al., $\lambda_n=1.4$")}
    # ---- Figure 2: PSNR and SSIM against evaluations of T, sigma_blur = 4.0 ----
    fig, ax = plt.subplots(2, 3, figsize=(12, 7.0))
    for j, im in enumerate(IMAGES):
        o = figdata[f"heavy_{im}"]
        for i, (key, lab) in enumerate((("traj", "PSNR (dB)"), ("ssim_traj", "SSIM"))):
            a_ = ax[i, j]
            for m in ("tispsa", "bot1", "bot14"):
                c, ls, lb = style[m]
                a_.plot(o[m]["evals_axis"], o[m][key], color=c, ls=ls, lw=2.0, label=lb)
            a_.set_xlim(0, Q_MAX); a_.set_title(im, fontsize=16) if i == 0 else None
            a_.set_xlabel("Evaluations of $T$", fontsize=16) if i == 1 else None
            a_.set_ylabel(lab, fontsize=15); a_.tick_params(labelsize=13); a_.grid(alpha=0.3)
            vis = [v for m in ("tispsa", "bot1", "bot14") for k, v in zip(o[m]["evals_axis"], o[m][key]) if Q_FROM <= k <= Q_MAX]
            pad = 0.05 * (max(vis) - min(vis)); a_.set_ylim(min(vis) - pad, max(vis) + pad)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=13, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.0, h_pad=1.5)
    plt.savefig(os.path.join(outdir, "figures", "fig2_deblur_quality.pdf"), metadata={"CreationDate": None}); plt.close()
    # ---- Figure 3: relative objective gap against evaluations of T, sigma_blur = 4.0 ----
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.9))
    for j, im in enumerate(IMAGES):
        o = figdata[f"heavy_{im}"]; a_ = ax[j]
        for m in ("tispsa", "bot1", "bot14"):
            c, ls, lb = style[m]
            a_.semilogy(o[m]["evals_axis"], np.maximum(o[m]["gap"], 1e-6), color=c, ls=ls, lw=2.0, label=lb)
        a_.axhline(TOL_REL, color="gray", ls=(0, (5, 2)), lw=1.4, label=r"Tolerance $10^{-2}$")
        a_.set_title(im, fontsize=16); a_.set_xlabel("Evaluations of $T$", fontsize=16)
        a_.set_ylabel(r"$(F(x_n)-F_{\mathrm{ref}})/(F(x_0)-F_{\mathrm{ref}})$", fontsize=14); a_.tick_params(labelsize=13); a_.grid(alpha=0.3, which="both")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=13, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)
    plt.savefig(os.path.join(outdir, "figures", "fig3_deblur_gap.pdf"), metadata={"CreationDate": None}); plt.close()


def replot(outdir):
    """Redraw Figures 1-3 from the data saved by main()."""
    figdata = json.load(open(os.path.join(outdir, "deblur_figdata.json")))
    z = np.load(os.path.join(outdir, "deblur_figimages.npz"))
    make_figures(outdir, figdata, {(im, m): z[f"{im}_{m}"] for im in IMAGES for m in ("blur", "tispsa", "bot1", "bot14")})


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = args[0] if args else "results"
    replot(out) if "--replot" in sys.argv else main(out)
