import conftest  # noqa: F401  (sets sys.path)
import math

import numpy as np
from scipy.linalg import solve_discrete_are

import lqr_controller as L
from closed_loop import drive
from sim.tracks import arc_route


def test_iterative_riccati_matches_scipy():
    p = L.VehicleParams()
    for v in (3.0, 10.0, 25.0):
        A, B = L.continuous_matrices(p, v)
        Ad, Bd = L.discretize_apollo(A, B, 0.05)
        Q, R = np.diag([0.05, 0, 1, 0]), np.array([[1.0]])
        K, P, it = L.solve_lqr_iterative(Ad, Bd, Q, R, 1e-12, 5000)
        Ps = solve_discrete_are(Ad, Bd, Q, R)
        assert np.allclose(P, Ps, rtol=1e-6, atol=1e-8), (v, it)
        assert np.all(np.abs(np.linalg.eigvals(Ad - Bd @ K)) < 1.0)  # closed loop stable


def test_feedforward_gives_zero_steady_state_error_on_linear_model():
    """Constant curvature, linear model: -K x_ss + delta_ff must hold e_y = 0."""
    p = L.VehicleParams()
    ctrl = L.LQRLateralController(p, 0.05)
    v, kappa = 15.0, 1 / 80.0
    K, _, _ = ctrl.gain(v)
    A, B = L.continuous_matrices(p, v)
    E = np.array([0.0, (p.lr * p.cr - p.lf * p.cf) / (p.mass * v) - v, 0.0,
                  -(p.lf**2 * p.cf + p.lr**2 * p.cr) / (p.iz * v)]) * kappa * v
    ff = ctrl.feedforward(v, kappa, K)
    # closed loop x_dot = (A - B K) x + B ff + E = 0
    x_ss = np.linalg.solve(A - B @ K.reshape(1, 4), -(B.ravel() * ff + E))
    assert abs(x_ss[0]) < 1e-9 and abs(x_ss[1]) < 1e-9


def test_closed_loop_demo_route():
    r = drive("lqr")
    assert r["max_e_y"] < 0.10, r["max_e_y"]
    assert r["max_rate"] < math.radians(90)


def test_closed_loop_tight_arc_near_friction_limit():
    r = drive("lqr", route=arc_route(radius=45.0), a_lat=4.4)
    assert r["max_e_y"] < 0.20, r["max_e_y"]


if __name__ == "__main__":
    for fn in (test_iterative_riccati_matches_scipy, test_feedforward_gives_zero_steady_state_error_on_linear_model,
               test_closed_loop_demo_route, test_closed_loop_tight_arc_near_friction_limit):
        fn()
        print("PASS", fn.__name__)
