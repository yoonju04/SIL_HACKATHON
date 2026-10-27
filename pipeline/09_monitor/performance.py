"""Switch-aligned tracking, stability and comfort diagnostics; never a safety oracle.

TLC is a local constant-boundary-closing-rate approximation, not a curved-road
prediction or measured FTTI. Benchmark comfort limits do not change monitor labels.
"""
from __future__ import annotations
import numpy as np


def switch_diagnostics(tr, cfg, check, onset, switch_time, ay):
    t = np.asarray(tr.t, float)
    ey = np.asarray(tr.lane_offset_m, float)
    vx, vy, yaw = (np.asarray(x, float) for x in (tr.vx, tr.vy, tr.yaw_rate))
    heading = None if tr.lane_heading_err is None else np.asarray(tr.lane_heading_err, float)
    ay = np.asarray(ay, float)
    yaw_accel = np.gradient(yaw, t)
    jerk = np.gradient(ay, t)
    moving = vx > cfg.diagnostic_low_speed_mps
    beta = np.full(len(t), np.nan)
    beta[moving] = np.arctan2(vy[moving], vx[moving])
    tlc = np.full(len(t), np.nan)
    half_w, half_l = tr.half_width_m or cfg.vehicle_half_width_m, tr.half_length_m or cfg.vehicle_half_length_m
    lane = tr.lane_width_m
    if lane is None and cfg.lane_width_default_m is not None:
        lane = np.full(len(t), cfg.lane_width_default_m)
    tlc_available = heading is not None and lane is not None and half_w is not None and half_l is not None
    if tlc_available:
        extent = half_l * np.abs(np.sin(heading)) + half_w * np.abs(np.cos(heading))
        margins = [np.asarray(lane) / 2 - ey - extent, np.asarray(lane) / 2 + ey - extent]
        for margin in margins:
            closing = -np.gradient(margin, t)
            side = np.full(len(t), np.nan)
            outside = margin <= 0
            side[outside] = 0.0
            approaching = moving & (closing > 1e-6) & ~outside
            side[approaching] = margin[approaching] / closing[approaching]
            use = np.isfinite(side) & (~np.isfinite(tlc) | (side < tlc))
            tlc[use] = side[use]

    def peak(values, mask):
        if values is None:
            return None
        x = np.asarray(values)[mask]
        x = x[np.isfinite(x)]
        return float(np.max(np.abs(x))) if len(x) else None

    start, end = check.window
    phases = {'evaluation': (start, end)}
    if switch_time is not None:
        phases.update(pre_switch=(onset, switch_time),
                      transition=(max(start, switch_time-cfg.dt), switch_time+cfg.sr2_window_s),
                      post_switch=(switch_time+cfg.sr2_window_s, switch_time+cfg.diagnostic_post_window_s))
    result = {}
    for name, (a, b) in phases.items():
        mask = (t >= a-1e-9) & (t <= min(b,end)+1e-9) & check.win
        if name == 'pre_switch':
            mask &= t < switch_time-1e-9
        elif name == 'post_switch':
            mask &= t > a+1e-9
        samples = int(mask.sum())
        if not samples:
            result[name] = {'status': 'unavailable', 'reason': 'no samples in phase', 'samples': 0}
            continue
        measured_tlc = tlc[mask & np.isfinite(tlc)]
        tlc_status = ('unavailable_geometry' if not tlc_available else 'finite_prediction' if measured_tlc.size
                      else 'low_speed' if not np.any(moving[mask]) else 'not_closing_under_local_model')
        values = {'status': 'complete' if b <= end+1e-9 else 'truncated', 'samples': samples,
                  'window_s': [float(t[mask][0]),float(t[mask][-1])],
                  'lateral_rmse_m': float(np.sqrt(np.mean(ey[mask]**2))),
                  'lateral_mae_m': float(np.mean(np.abs(ey[mask]))), 'max_abs_lateral_error_m': peak(ey,mask),
                  'max_abs_heading_error_rad': peak(heading,mask),
                  'min_lane_margin_m': float(np.min(check.margin[mask])),
                  'max_abs_lateral_accel_mps2': peak(ay,mask),
                  'max_abs_lateral_jerk_mps3': peak(jerk,mask) if samples >= 3 else None,
                  'max_abs_yaw_rate_rad_s': peak(yaw,mask),
                  'max_abs_yaw_accel_rad_s2': peak(yaw_accel,mask) if samples >= 3 else None,
                  'max_abs_sideslip_rad': peak(beta,mask),
                  'tlc_local_min_s': float(np.min(measured_tlc)) if measured_tlc.size else None,
                  'tlc_status': tlc_status}
        result[name] = values

    observed = result.get('post_switch') or result['evaluation']
    comparisons = {}
    for metric, limit in cfg.comfort_reference_limits.items():
        value = observed.get(metric)
        comparisons[metric] = {'value': value, 'reference_limit': limit,
                               'within_reference': value <= limit if value is not None else None}
    return {'role': 'report_only', 'phases': result,
            'not_applicable_phases': ['pre_switch','transition','post_switch'] if switch_time is None else [],
            'signal_processing': {'derivative': 'numpy.gradient(actual timestamps), differentiate before slicing',
                                  'lateral_acceleration': 'logged body ay' if tr.ay_mps2 is not None else 'd(vy)/dt + vx*r',
                                  'fallback_ay_moving_average_ticks': cfg.ay_smoothing_ticks if tr.ay_mps2 is None else None,
                                  'sideslip_low_speed_exclusion_mps': cfg.diagnostic_low_speed_mps},
            'comfort_reference': {'source': 'TREEIRL_2025, Appendix B.1; benchmark-inspired subset only',
                                  'window': 'post_switch' if switch_time is not None else 'evaluation',
                                  'full_nuplan_comfort_score': False, 'comparisons': comparisons},
            'tlc_model': 'local footprint-boundary margin / positive closing rate; not FTTI or nonlinear TLC'}
