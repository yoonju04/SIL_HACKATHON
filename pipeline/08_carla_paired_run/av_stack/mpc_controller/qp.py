"""Dense convex QP solver (ADMM, the OSQP algorithm) in numpy/scipy.

    minimise   1/2 x' P x + q' x
    subject to l <= A x <= u

Algorithm: B. Stellato et al., "OSQP: an operator splitting solver for
quadratic programs", Mathematical Programming Computation 12, 2020
(https://osqp.org). Autoware's MPC uses the OSQP library; OSQP is not always
installable (no wheel / no network on the CARLA machine), so this is a small
dense re-implementation of the same iteration:

    (P + sigma I + rho A'A) x~ = sigma x - q + A'(rho z - y)
    z~ = A x~
    x  <- alpha x~ + (1 - alpha) x
    z+ =  Proj_[l,u]( alpha z~ + (1 - alpha) z + y / rho )
    y  <- y + rho (alpha z~ + (1 - alpha) z - z+)

with diagonal scaling (variables to unit Hessian diagonal, constraint rows to
unit norm), OSQP's termination test and adaptive rho. The iteration count is
bounded, so the worst-case time is bounded; for the 50-variable steering QP
it converges in tens to a few hundred iterations.
If the `osqp` package is installed, solve(..., backend="osqp") uses it instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QPSettings:
    rho: float = 0.1
    sigma: float = 1e-6
    alpha: float = 1.6
    eps_abs: float = 1e-6
    eps_rel: float = 1e-6
    max_iter: int = 4000
    check_every: int = 10
    adaptive_rho_every: int = 50
    polish: bool = True  # OSQP-style polishing: re-solve the KKT system on the active set


@dataclass
class QPResult:
    x: np.ndarray
    status: str  # "solved" | "max_iter" | "error:<msg>"
    iterations: int
    primal_residual: float
    dual_residual: float
    y: np.ndarray | None = None


class ADMMQP:
    def __init__(self, settings: QPSettings | None = None):
        self.st = settings or QPSettings()
        self._warm = None  # (x_bar, z_bar, y_bar) from the previous call, scaled

    def reset(self):
        self._warm = None

    def solve(self, P, q, A, l, u, warm_start: bool = True) -> QPResult:
        st = self.st
        P = np.asarray(P, float)
        q = np.asarray(q, float)
        A = np.asarray(A, float)
        l = np.asarray(l, float)
        u = np.asarray(u, float)
        n, m = P.shape[0], A.shape[0]

        # ---- scaling: D (variables), E (constraint rows), c (cost)
        D = 1.0 / np.sqrt(np.maximum(np.diag(P), 1e-12))
        Ps = D[:, None] * P * D[None, :]
        AD = A * D[None, :]
        E = 1.0 / np.maximum(np.linalg.norm(AD, ord=np.inf, axis=1), 1e-12)
        As = E[:, None] * AD
        c = 1.0 / max(1.0, np.max(np.abs(D * q)))
        Ps *= c
        qs = c * D * q
        ls, us = E * l, E * u

        x = np.zeros(n)
        z = np.clip(np.zeros(m), ls, us)
        y = np.zeros(m)
        if warm_start and self._warm is not None and self._warm[0].shape == (n,) and self._warm[1].shape == (m,):
            x, z, y = (w.copy() for w in self._warm)
            z = np.clip(z, ls, us)

        rho = st.rho
        AtA = As.T @ As
        I = np.eye(n)

        def factor(r):  # dense inverse: n is small (horizon), mat-vec is cheaper than cho_solve
            return np.linalg.inv(Ps + st.sigma * I + r * AtA)

        K = factor(rho)
        status, it = "max_iter", 0
        r_prim = r_dual = np.inf
        for it in range(1, st.max_iter + 1):
            rhs = st.sigma * x - qs + As.T @ (rho * z - y)
            xt = K @ rhs
            zt = As @ xt
            x = st.alpha * xt + (1 - st.alpha) * x
            zr = st.alpha * zt + (1 - st.alpha) * z
            z_new = np.clip(zr + y / rho, ls, us)
            y = y + rho * (zr - z_new)
            z = z_new

            if it % st.check_every == 0 or it == st.max_iter:
                Ax = As @ x
                Px = Ps @ x
                Aty = As.T @ y
                # residuals in the unscaled problem
                r_prim = np.max(np.abs((Ax - z) / E))
                r_dual = np.max(np.abs((Px + qs + Aty) / D)) / c
                e_prim = st.eps_abs + st.eps_rel * max(np.max(np.abs(Ax / E)), np.max(np.abs(z / E)))
                e_dual = st.eps_abs + st.eps_rel / c * max(np.max(np.abs(Px / D)),
                                                           np.max(np.abs(Aty / D)),
                                                           np.max(np.abs(qs / D)))
                if r_prim <= e_prim and r_dual <= e_dual:
                    status = "solved"
                    break
                if it % st.adaptive_rho_every == 0:
                    num = r_prim / max(np.max(np.abs(Ax / E)), np.max(np.abs(z / E)), 1e-12)
                    den = r_dual / max(np.max(np.abs(Px / D)), np.max(np.abs(Aty / D)),
                                       np.max(np.abs(qs / D)), 1e-12) * c
                    new_rho = float(np.clip(rho * np.sqrt(num / max(den, 1e-12)), 1e-6, 1e6))
                    if new_rho > 5 * rho or new_rho < 0.2 * rho:
                        rho = new_rho
                        K = factor(rho)
        self._warm = (x.copy(), z.copy(), y.copy())
        x_out, y_out = D * x, E * y / c
        if st.polish:
            polished = _polish(P, q, A, l, u, x_out, y_out)
            if polished is not None:
                x_out, y_out = polished
                status = status + "+polished"
        return QPResult(x=x_out, status=status, iterations=it, primal_residual=float(r_prim),
                        dual_residual=float(r_dual), y=y_out)


def _polish(P, q, A, l, u, x, y, tol=1e-7):
    """Solve the equality-constrained QP on the active set guessed from (x, y).

    Returns the polished (x, y) if it is feasible and not worse, else None.
    """
    Ax = A @ x
    lower = (Ax - l < 1e-5 * (1 + np.abs(l))) & (y < -tol)
    upper = (u - Ax < 1e-5 * (1 + np.abs(u))) & (y > tol)
    lower &= ~upper
    act = np.where(lower | upper)[0]
    b = np.where(lower, l, u)[act]
    Aa = A[act]
    n, k = P.shape[0], len(act)
    KKT = np.zeros((n + k, n + k))
    KKT[:n, :n] = P
    KKT[:n, n:] = Aa.T
    KKT[n:, :n] = Aa
    try:
        sol = np.linalg.solve(KKT + 1e-12 * np.eye(n + k), np.r_[-q, b])
    except np.linalg.LinAlgError:
        return None
    xp = sol[:n]
    Axp = A @ xp
    if np.any(Axp < l - 1e-7) or np.any(Axp > u + 1e-7):
        return None
    yp = np.zeros_like(y)
    yp[act] = sol[n:]
    f = lambda v: 0.5 * v @ P @ v + q @ v  # noqa: E731
    if f(xp) > f(x) + 1e-9 * (1 + abs(f(x))):
        return None
    return xp, yp


def solve_osqp(P, q, A, l, u, eps=1e-6, max_iter=4000) -> QPResult:  # pragma: no cover
    """Optional: the reference OSQP library (pip install osqp)."""
    import osqp
    from scipy import sparse

    prob = osqp.OSQP()
    prob.setup(sparse.csc_matrix(np.triu(P)), q, sparse.csc_matrix(A), l, u,
               eps_abs=eps, eps_rel=eps, max_iter=max_iter, verbose=False, polish=True)
    r = prob.solve()
    ok = r.info.status_val in (1, 2)
    return QPResult(x=np.asarray(r.x, float), status="solved" if ok else f"error:{r.info.status}",
                    iterations=int(r.info.iter), primal_residual=float(r.info.pri_res),
                    dual_residual=float(r.info.dua_res), y=np.asarray(r.y, float))
