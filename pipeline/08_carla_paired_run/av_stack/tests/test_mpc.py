import conftest  # noqa: F401  (sets sys.path)
import math

import numpy as np

import mpc_controller as M
from closed_loop import drive
from sim.tracks import arc_route, straight_route


def _random_qp(rng, n=50):
    G = rng.normal(size=(n, n))
    P = G @ G.T * rng.uniform(1, 1e3) + np.eye(n) * rng.uniform(0.1, 10)
    q = rng.normal(size=n) * rng.uniform(1, 1e3)
    D = np.eye(n)
    D[np.arange(1, n), np.arange(n - 1)] = -1.0
    A = np.vstack([np.eye(n), D])
    lim, r = 0.5, 0.05
    l = np.r_[-lim * np.ones(n), 0.1 - r, -r * np.ones(n - 1)]
    u = np.r_[lim * np.ones(n), 0.1 + r, r * np.ones(n - 1)]
    return P, q, A, l, u


def test_qp_solution_satisfies_kkt():
    rng = np.random.default_rng(1)
    for _ in range(20):
        P, q, A, l, u = _random_qp(rng)
        res = M.ADMMQP().solve(P, q, A, l, u)
        x, y = res.x, res.y
        Ax = A @ x
        assert res.status.startswith("solved")
        assert np.all(Ax >= l - 1e-6) and np.all(Ax <= u + 1e-6)  # primal feasible
        scale = 1 + np.abs(q).max()
        assert np.abs(P @ x + q + A.T @ y).max() < 1e-4 * scale  # stationarity
        # complementarity: y < 0 only at the lower bound, y > 0 only at the upper bound
        assert np.all((y > -1e-6 * scale) | (Ax - l < 1e-5))
        assert np.all((y < 1e-6 * scale) | (u - Ax < 1e-5))


def test_qp_inactive_constraints_equals_linear_solve():
    rng = np.random.default_rng(2)
    n = 20
    G = rng.normal(size=(n, n))
    P = G @ G.T + np.eye(n)
    q = rng.normal(size=n) * 0.01
    x_ref = -np.linalg.solve(P, q)
    res = M.ADMMQP().solve(P, q, np.eye(n), -10 * np.ones(n), 10 * np.ones(n))
    assert np.abs(res.x - x_ref).max() < 1e-7


def test_butterworth_unit_dc_gain():
    f = M.Butterworth2d(0.05, 3.0)
    y = [f.filter(1.0) for _ in range(200)]
    assert abs(y[-1] - 1.0) < 1e-9


def _controller(cfg=None):
    c = M.MPCLateralController(M.VehicleParams(), 0.05, cfg)
    x, y = straight_route(200)
    c.set_reference(x, y, np.zeros_like(x), np.zeros_like(x), None, np.full(len(x), 10.0))
    return c


def test_rate_constraint_anchored_on_own_previous_raw_command():
    c = _controller()
    rate = math.radians(c.cfg.steer_rate_lim_dps) * c.dt
    prev = c.raw_prev
    for k in range(5):  # 2 m left of the path: wants a large right steer
        o = c.compute(10.0, 2.0, 0.0, 10.0, 0.0, 0.0, 0.05 * k)
        assert abs(o.raw_steer - prev) <= rate + 1e-9
        assert np.all(np.abs(np.diff(o.u_sequence)) <= math.radians(c.cfg.steer_rate_lim_dps) * c.cfg.prediction_dt + 1e-9)
        prev = o.raw_steer
    assert o.raw_steer < 0  # steers right, towards the path


def test_rate_reference_applied_option():
    c = _controller(M.MPCConfig(rate_reference="applied"))
    rate = math.radians(c.cfg.steer_rate_lim_dps) * c.dt
    o = c.compute(10.0, 2.0, 0.0, 10.0, 0.0, 0.0, 0.0, applied_prev=0.1)
    assert abs(o.raw_steer - 0.1) <= rate + 1e-9


def test_closed_loop_demo_route():
    r = drive("mpc")
    assert r["max_e_y"] < 0.15, r["max_e_y"]
    assert all(row[4].ok for row in r["rows"])


def test_closed_loop_tight_arc_near_friction_limit():
    r = drive("mpc", route=arc_route(radius=45.0), a_lat=4.4)
    assert r["max_e_y"] < 0.25, r["max_e_y"]


def test_delay_compensation_runs():
    r = drive("mpc", cfg=M.MPCConfig(input_delay_s=0.1))
    assert r["max_e_y"] < 0.20, r["max_e_y"]


def test_deterministic_outputs():
    a = drive("mpc")["rows"]
    b = drive("mpc")["rows"]
    assert [x[2] for x in a] == [x[2] for x in b]


if __name__ == "__main__":
    for fn in (test_qp_solution_satisfies_kkt, test_qp_inactive_constraints_equals_linear_solve,
               test_butterworth_unit_dc_gain, test_rate_constraint_anchored_on_own_previous_raw_command,
               test_rate_reference_applied_option, test_closed_loop_demo_route,
               test_closed_loop_tight_arc_near_friction_limit, test_delay_compensation_runs,
               test_deterministic_outputs):
        fn()
        print("PASS", fn.__name__)
