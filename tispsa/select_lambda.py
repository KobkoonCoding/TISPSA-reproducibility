"""Selection of the TV regularization parameter (Section 4.2).

For every image and value of sigma_blur, lambda_TV is chosen on the grid below as the value
maximizing the PSNR of the reference solution u_ref (1000 forward-backward iterations).  Runtime: about
15 minutes.  The same lambda_TV is then used for all methods.  The selected values are stored in
results/lambda_tv.json and hard-coded in deblurring.LAMBDA_TV_DEFAULT for convenience.
"""
import json, os, sys
from .deblurring import DEGRADATIONS, IMAGES, degrade, make_T, psnr, ssim, R_STEP
from .data import load_image

GRID = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016]
N_REF = 1000


def converged(x, b, K, Kt, lam, n=N_REF):
    T = make_T(K, Kt, b, R_STEP, lam); u = b.copy()
    for _ in range(n): u = T(u)
    return u


def main(outdir="results"):
    os.makedirs(outdir, exist_ok=True); sel = {}
    for dname, (sb, sn) in DEGRADATIONS.items():
        for im in IMAGES:
            x = load_image(im); K, Kt, b = degrade(x, sb, sn); best = None
            for lam in GRID:
                u = converged(x, b, K, Kt, lam); p, s = psnr(u, x), ssim(u, x)
                print(f"{dname:5s} {im:5s} lambda={lam:<6} converged PSNR {p:.2f} SSIM {s:.4f}", flush=True)
                if best is None or p > best[1]: best = (lam, p, s)
            sel[f"{dname}_{im}"] = {"lambda_tv": best[0], "psnr_conv": best[1], "ssim_conv": best[2]}
            print(f"  -> {dname} {im}: lambda_TV = {best[0]}", flush=True)
    json.dump(sel, open(os.path.join(outdir, "lambda_tv.json"), "w"), indent=1)
    return sel


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
