"""The fixed point algorithms compared in the paper.

All routines are written for an arbitrary mapping ``T`` acting on NumPy arrays of any
shape (a vector, an image, or a stacked primal-dual pair), so that the *same* code is
used in all applications of Section 4.

* ``run_bot``    : the Tikhonov-regularized Krasnosel'skii-Mann iteration of Bot and Meier (2021),
                   eq. (3) of the paper,
                       x_{n+1} = beta_n x_n + lambda_n (T_n(beta_n x_n) - beta_n x_n);
                   one evaluation of T_n per iteration.  With T_n = T for all n it is the scheme
                   (2) of Bot, Csetnek and Meier (2019) used in Sections 4.1-4.3; with
                   T_n = J_{r_n A}(I - r_n B) it is the forward-backward algorithm (33) used in Section 4.5.
* ``run_tispsa`` : Algorithm 1 (TISPSA) of the paper, eqs. (8)-(13); two evaluations of T_n per iteration.

``beta`` and ``eps`` are callables n -> beta_n, n -> epsilon_n (n starts at 1).
``norm`` is the norm used inside the inertial parameters (Euclidean by default; the product
L^2 norm is passed for the split feasibility problem).
``callback(n, x)`` is called after every iteration; if it returns True the iteration stops.
``Tn`` (optional) is a callable n -> T_n; otherwise T_n = T for all n.
"""
import numpy as np

# Setting of TISPSA used in every experiment of Section 4 (eq. (34) of the paper): lambda_n = 1,
# alpha_n = 0.95, mu_n = 0.5, rho_n = 0.1 and epsilon_n = 10/(n+1)^{3/2}.  It was chosen once on one instance
# per application from the small grid described in README.md (Section 4) and is used unchanged everywhere.
LAM, ALPHA, MU, RHO = 1.0, 0.95, 0.5, 0.1
EPS = lambda n: 10.0 / (n + 1) ** 1.5


def _family(T, Tn):
    """Return n -> T_n.  ``T`` alone gives T_n = T for all n; ``Tn`` is a callable n -> mapping."""
    return Tn if Tn is not None else (lambda n: T)


def run_bot(T, x0, N, beta, lam=1.0, tikhonov=True, callback=None, Tn=None):
    """Scheme (3); ``tikhonov=False`` sets beta_n = 1 (the variant without Tikhonov regularization, Section 4.1)."""
    fam = _family(T, Tn)
    x = np.array(x0, dtype=float, copy=True)
    for n in range(1, N + 1):
        b = beta(n) if tikhonov else 1.0
        Tx = fam(n)(b * x)
        x = b * x + lam * (Tx - b * x)
        if callback is not None and callback(n, x):
            return x, n
    return x, N


def run_tispsa(T, x0, N, beta, lam=LAM, alp=ALPHA, mu=MU, rho=RHO, eps=EPS, norm=np.linalg.norm, callback=None,
               Tn=None):
    """Algorithm 1 with x_1 = x_0 and w_0 = x_0.  The upper bounds mu_n = mu and rho_n = rho must be nonnegative;
    mu = 0 gives y_n = x_n and rho = 0 gives x_{n+1} = w_n (ablation of Section 4.4)."""
    if mu < 0 or rho < 0:
        raise ValueError("Algorithm 1 requires mu_n >= 0 and rho_n >= 0")
    fam = _family(T, Tn)
    xm1 = np.array(x0, dtype=float, copy=True)   # x_0
    x = xm1.copy()                               # x_1 = x_0
    wm1 = xm1.copy()                             # w_0 = x_0
    for n in range(1, N + 1):
        b, e = beta(n), eps(n)
        # Step 1: first inertial parameter (8)
        d1 = norm(x - xm1)
        th = min(mu, e / d1) if d1 > 0 else mu
        # Step 2: (9)-(11); T_n(beta_n y_n) is evaluated once and reused
        Tcur = fam(n)
        y = x + th * (x - xm1)
        Tby = Tcur(b * y)
        z = (1 - alp) * b * y + alp * Tby
        Tz = Tcur(z)
        w = (1 - lam) * Tby + lam * Tz
        # Step 3: second inertial parameter (12) and update (13)
        d2 = norm(w - wm1)
        de = min(rho, e / d2) if d2 > 0 else rho
        xn = w + de * (w - wm1)
        xm1, x, wm1 = x, xn, w
        if callback is not None and callback(n, x):
            return x, n
    return x, N
