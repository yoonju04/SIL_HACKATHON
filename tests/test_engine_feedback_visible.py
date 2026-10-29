"""integration4: which runs the video explainer gets (engine.feedback.visible_failure)."""
from engine.feedback import is_failure, visible_failure


def rec(sr4=True, collision=False, sr="pass", ro="safe_recovery", cat="switched_unsafe"):
    return {"sr_checks": {"sr": {"SR4": {"pass": sr4, "collision": collision}}}, "sr_outcome": sr,
            "recovery_outcome": ro, "outcome_category": cat}


def test_jerk_only_failure_is_not_a_visible_failure():
    # result_8: switched, kept the lane, failed only on lateral jerk -> is_failure but nothing to show on video
    jerk_only = rec()
    assert is_failure(jerk_only) and not visible_failure(jerk_only)


def test_departure_collision_and_late_switch_are_visible():
    assert visible_failure(rec(sr4=False))
    assert visible_failure(rec(collision=True))
    assert visible_failure(rec(sr="F1_no_or_late_switch"))
    assert visible_failure(rec(ro="recovered_after_departure"))
    assert not visible_failure(rec(cat="no_switch", ro="no_switch"))
