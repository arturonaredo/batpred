"""Actual replay-derived charge ceilings and truthful unavailable-candidate notices."""

from tests.car_planning_fixtures import car_settings

from copy import deepcopy
import math
from types import SimpleNamespace
from unittest.mock import patch

from car_charging_solver import Slot
from car_charging_plan import _native_windows, _replay_candidate, _options, _solver_charge_run_limits, build_candidate, evaluate_car_charging
from car_charging_view import build_car_charging_plan_view, _unavailable_reason
from tests.test_car_planning_native import _setup, _controls
from tests.test_car_planning_single_table import _native_plan, _good_report


def test_charge_ceiling_is_replayed_not_cosmetic(base):
    """A guarded whole-percent target remains tied to the solver trajectory and both native replays."""
    _setup(base, minutes=60)
    base.car_charging_limit = [0] * len(base.car_charging_limit)
    trajectory_peak = max(base.soc_kw, base.soc_max * 0.73)
    result = SimpleNamespace(
        status="feasible",
        actions=tuple(SimpleNamespace(slot_index=index, battery_mode="charge", ev_action="off") for index in range(12)),
        battery_soc_trace=tuple([base.soc_kw] + [trajectory_peak] * 12),
    )
    slots = [Slot(index, index * 5, 5, 1 / 12, 0, 10, 0) for index in range(12)]
    options = dict(mode="shadow", ev_soc=0, import_limit_w=6000, terminal_min=base.reserve, initially_on=False, initial_off_minutes=10, pause=15, restart=25, arrival=0, deadline=60, off_minutes=10, load_cap=10)
    before = _controls(base)
    original = _native_windows(base, result, slots)
    replay_calls = []

    def replay_with_ceiling(_base, _options, _result, _slots, windows, scenario):
        """Capture the exact bound and mark the full paired horizon as independently checked."""
        replay_calls.append((windows, scenario))
        return {"passed": True, "reasons": [], "checked_minutes": 60}

    with patch("car_charging_plan._replay", side_effect=replay_with_ceiling):
        windows, checks = _replay_candidate(base, options, result, slots)
    target = windows[2][0] * 100 / base.soc_max
    peak_percent = trajectory_peak * 100 / base.soc_max
    expected = min(100, math.ceil(peak_percent + 1 - 1e-9))
    assert target == expected and peak_percent + 1 - 1e-9 <= target <= peak_percent + 2
    assert target < 100
    assert original[2] == [base.soc_max]
    for index, (scenario, check) in enumerate(checks.items()):
        assert check["passed"] and "_charge_peak_kwh" not in check
        assert check["checked_minutes"] == 60
        assert replay_calls[index][1] == index
        assert replay_calls[index][0][2] == windows[2]
    assert _controls(base) == before


def test_charge_ceiling_clears_restart_boundary_without_full_soc_fallback(base):
    """The native command clears a 25% restart threshold by one guarded point, not by charging to full."""
    _setup(base, minutes=5)
    peak = base.soc_max * 0.25
    result = SimpleNamespace(
        actions=(SimpleNamespace(slot_index=0, battery_mode="charge", ev_action="off"),),
        battery_soc_trace=(peak, peak),
    )
    slots = [Slot(0, 0, 5, 1 / 12, 0, 10, 0)]
    limit = _solver_charge_run_limits(base, result, slots)[0]
    assert round(limit * 100 / base.soc_max) == 26
    assert limit == base.soc_max * 0.26


def test_reduced_target_failure_cannot_reuse_full_target(base):
    """A second replay rejection stays rejected; the first passed result cannot leak."""
    actions = [SimpleNamespace(slot_index=0, battery_mode="charge", ev_action="off")]
    result = SimpleNamespace(actions=actions, battery_soc_trace=(base.soc_max * 0.5, base.soc_max * 0.5))
    slots = [Slot(0, 0, 5, 0, 0, 10, 0)]
    reference = {"passed": True, "reasons": [], "_charge_peak_kwh": [base.soc_max * 0.5]}
    rejected = {"passed": False, "reasons": ["site_import_exceeded"], "_charge_peak_kwh": [base.soc_max * 0.5]}
    with patch("car_charging_plan._replay", side_effect=[deepcopy(reference), deepcopy(rejected)]):
        windows, checks = _replay_candidate(base, {"mode": "shadow"}, result, slots)
    assert round(windows[2][0] * 100 / base.soc_max) == 51
    assert checks["pv10"]["passed"] is False
    assert all("_charge_peak_kwh" not in check for check in checks.values())


def test_charge_ceiling_uses_each_solver_charge_run_and_preserves_full_target(base):
    """Each contiguous charge run gets its solver-trace ceiling before paired replay."""
    _setup(base, minutes=20)
    base.car_charging_limit = [0] * len(base.car_charging_limit)
    actions = tuple(SimpleNamespace(slot_index=index, battery_mode=mode, ev_action="off", discharge_limit_kw=None) for index, mode in enumerate(("charge", "charge", "demand", "charge")))
    trace = (base.soc_max * 0.4, base.soc_max * 0.65, base.soc_max * 0.70, base.soc_max * 0.68, base.soc_max)
    result = SimpleNamespace(actions=actions, battery_soc_trace=trace)
    slots = [Slot(index, index * 5, 5, 0.1, 0, 10, 0) for index in range(4)]
    options = {"mode": "shadow", "grid_charge_support_w": 0}
    observed_limits = []

    def replay_with_limit(_base, _options, _result, _slots, windows, _scenario):
        """Record only the discrete ceilings presented to native replay."""
        observed_limits.append(windows[2])
        return {"passed": True, "reasons": [], "checked_minutes": 20}

    with patch("car_charging_plan._replay", side_effect=replay_with_limit) as replay:
        windows, checks = _replay_candidate(base, options, result, slots)

    assert [round(limit * 100 / base.soc_max) for limit in windows[2]] == [71, 100]
    assert observed_limits == [windows[2], windows[2]]
    assert replay.call_count == 2
    assert all(check["passed"] for check in checks.values())


def test_charge_ceiling_and_window_label_split_at_tariff_boundary(base):
    """A later cheap-band peak cannot authorize an early expensive-band target."""
    from car_charging_plan import _candidate_commands

    _setup(base, minutes=20)
    base.car_charging_limit = [0] * len(base.car_charging_limit)
    actions = tuple(SimpleNamespace(slot_index=index, battery_mode="charge", ev_action="off", discharge_limit_kw=None) for index in range(4))
    trace = (base.soc_max * 0.80, base.soc_max * 0.85, base.soc_max * 0.889, base.soc_max * 0.94, base.soc_max * 0.94)
    result = SimpleNamespace(actions=actions, battery_soc_trace=trace)
    slots = [Slot(index, index * 5, 5, 0.1, 0, 19.1 if index < 2 else 8.1, 0) for index in range(4)]
    options = {"mode": "shadow", "grid_charge_support_w": 0}

    with patch("car_charging_plan._replay", return_value={"passed": True, "reasons": [], "checked_minutes": 20}):
        windows, checks = _replay_candidate(base, options, result, slots)

    assert [(window["start"], window["end"], window["average"]) for window in windows[1]] == [
        (base.minutes_now, base.minutes_now + 10, 19.1),
        (base.minutes_now + 10, base.minutes_now + 20, 8.1),
    ]
    assert [round(limit * 100 / base.soc_max) for limit in windows[2]] == [90, 95]
    commands = _candidate_commands(base, result, slots, windows)
    assert [command["target_soc"] for command in commands] == [90, 90, 95, 95]
    assert all(check["passed"] for check in checks.values())


def test_solver_charge_ceiling_keeps_hold_window_aligned(base):
    """Interleaved Hold windows keep their native reserve limit between charge ceilings."""
    _setup(base, minutes=15)
    base.car_charging_limit = [0] * len(base.car_charging_limit)
    actions = tuple(SimpleNamespace(slot_index=index, battery_mode=mode, ev_action="off", discharge_limit_kw=None) for index, mode in enumerate(("charge", "hold", "charge")))
    trace = (base.soc_max * 0.4, base.soc_max * 0.6, base.soc_max * 0.55, base.soc_max * 0.8)
    result = SimpleNamespace(actions=actions, battery_soc_trace=trace)
    slots = [Slot(index, index * 5, 5, 0.1, 0, 10, 0) for index in range(3)]

    with patch("car_charging_plan._replay", return_value={"passed": True, "reasons": [], "checked_minutes": 15}):
        windows, checks = _replay_candidate(base, {"mode": "shadow"}, result, slots)

    assert len(windows[1]) == len(windows[2]) == 3
    assert [round(limit * 100 / base.soc_max) for limit in windows[2]] == [61, round(base.reserve * 100 / base.soc_max), 81]
    assert all(check["passed"] for check in checks.values())


def test_non_finite_solver_charge_trajectory_fails_closed(base):
    """A malformed solver SOC bound cannot be replaced by an inflated ceiling."""
    actions = (SimpleNamespace(slot_index=0, battery_mode="charge", ev_action="off", discharge_limit_kw=None),)
    result = SimpleNamespace(actions=actions, battery_soc_trace=(20.0, float("nan")))
    slots = [Slot(0, 0, 5, 0.1, 0, 10, 0)]

    try:
        _replay_candidate(base, {"mode": "shadow"}, result, slots)
    except ValueError as exc:
        assert "charge" in str(exc).lower() or "soc" in str(exc).lower()
    else:
        raise AssertionError("Non-finite solver trajectory produced an executable charge ceiling")


def test_table_limit_uses_real_candidate_target():
    """Keep the existing column but show the same limit used by native validation."""
    report = _good_report()
    for rows in report["preview"].values():
        rows[0].update(battery_modes="charge", charge_target_percent=53)
    view = build_car_charging_plan_view(_native_plan(), report)
    assert view["rows"][0]["show_limit"] == "53"
    assert view["rows"][0]["soc_percent"] == report["preview"]["nominal"][0]["battery_start_percent"]
    report["preview"]["pv10"][0]["charge_target_percent"] = 54
    assert build_car_charging_plan_view(_native_plan(), report)["plan_unavailable"] is True
    report["preview"]["pv10"][0]["charge_target_percent"] = float("nan")
    assert build_car_charging_plan_view(_native_plan(), report)["plan_unavailable"] is True


def test_rejection_notice_explains_actual_constraints():
    """Show specific failed constraints, never an unverified schedule or private text."""
    report = {"status": "conditional_native_rejected", "native_checks": {"pv10": {"reasons": ["site_import_exceeded", "ev_restart_threshold", "<private>"]}}}
    text = _unavailable_reason(report)
    assert "conservative forecast" in text and "grid import limit" in text and "restart threshold" in text
    assert "previous simulation is hidden" in text and "<private>" not in text


def test_conservative_search_miss_gets_bounded_retry(base):
    """A conservative beam miss gets the same finite retry as the nominal search."""
    _setup(base, minutes=60)
    car_settings(base)["control_model"] = "deye_solarman"
    with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
        options = _options(base)
        result, slots = build_candidate(base, options)
        assert result.actions
        empty = SimpleNamespace(actions=(), status="bounded_no_plan")
        windows = _native_windows(base, result, slots)
        rejected = {"nominal": {"passed": True}, "pv10": {"passed": False, "reasons": ["site_import_exceeded"]}}
        accepted = {"nominal": {"passed": True}, "pv10": {"passed": True}}
        with patch("car_charging_plan._collect_blockers", return_value=[]), patch("car_charging_plan.build_candidate", side_effect=[(result, slots), (empty, slots), (result, slots)]) as search:
            with patch("car_charging_plan._replay_candidate", side_effect=[(windows, rejected), (windows, accepted)]):
                report = evaluate_car_charging(base, publish=False)
        assert [call.args[1]["beam"] for call in search.call_args_list] == [128, 128, 512]
        assert report["status"] == "candidate" and len(report["search_attempts"]) == 3
        assert report["search_attempts"][-1]["forecast_basis"] == "envelope"
        assert not report["execution_enabled"] and not report["enforced"]


def test_apply_ceiling_stays_within_guarded_solver_trajectory(base):
    """Apply keeps the modest trajectory guard; it does not replace a planned ceiling with full SOC."""
    result = SimpleNamespace(actions=[SimpleNamespace(slot_index=0, battery_mode="charge", ev_action="off")], battery_soc_trace=(base.soc_max * 0.5, base.soc_max * 0.5))
    slots = [Slot(0, 0, 5, 0, 0, 10, 0)]
    reference = {"passed": True, "reasons": []}
    with patch("car_charging_plan._replay", side_effect=[deepcopy(reference), deepcopy(reference)]) as replay:
        windows, checks = _replay_candidate(base, {"mode": "apply"}, result, slots)
    assert windows[2] == [base.soc_max * 0.51] and replay.call_count == 2
    assert all("_charge_peak_kwh" not in check for check in checks.values())


def test_rejection_notice_handles_malformed_checks():
    """Malformed diagnostic data cannot crash the page or leak arbitrary values."""
    for checks in (None, [], {"pv10": None}, {"pv10": {"reasons": None}}, {"pv10": {"reasons": "private"}}):
        text = _unavailable_reason({"status": "native_rejected", "native_checks": checks})
        assert "No complete paired" in text and "private" not in text


def test_economic_rejection_recovers_only_revalidated_established_order(base):
    """An economic-order miss may recover a new valid path, never reuse a failed check."""
    _setup(base, minutes=60)
    car_settings(base)["control_model"] = "deye_solarman"
    with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
        options = _options(base)
        result, slots = build_candidate(base, options)
        windows = _native_windows(base, result, slots)
        rejected = {"nominal": {"passed": True}, "pv10": {"passed": False, "reasons": ["ev_pause_threshold"]}}
        accepted = {"nominal": {"passed": True}, "pv10": {"passed": True}}
        for last, expected in ((accepted, "candidate"), (rejected, "native_rejected")):
            with patch("car_charging_plan._best_effort_shadow", return_value=(None, [])), patch("car_charging_plan._collect_blockers", return_value=[]), patch("car_charging_plan.build_candidate", return_value=(result, slots)) as search:
                with patch("car_charging_plan._replay_candidate", side_effect=[(windows, rejected), (windows, rejected), (windows, last)]):
                    report = evaluate_car_charging(base, publish=False)
            assert search.call_count == 3 and search.call_args_list[-1].args[1]["economic_diversity"] is False
            assert report["status"] == expected and report["search_attempts"][-1]["pruning_order"] == "established"
            assert not report["execution_enabled"] and not report["enforced"]
            if expected == "native_rejected":
                assert build_car_charging_plan_view(_native_plan(), report)["plan_unavailable"] is True


def run_car_planning_targets_tests(base):
    """Run replay-target and rejection-message regressions through the native harness."""
    for test in (
        test_charge_ceiling_is_replayed_not_cosmetic,
        test_charge_ceiling_clears_restart_boundary_without_full_soc_fallback,
        test_charge_ceiling_uses_each_solver_charge_run_and_preserves_full_target,
        test_charge_ceiling_and_window_label_split_at_tariff_boundary,
        test_solver_charge_ceiling_keeps_hold_window_aligned,
        test_non_finite_solver_charge_trajectory_fails_closed,
        test_reduced_target_failure_cannot_reuse_full_target,
        test_conservative_search_miss_gets_bounded_retry,
        test_apply_ceiling_stays_within_guarded_solver_trajectory,
        test_economic_rejection_recovers_only_revalidated_established_order,
    ):
        test(base)
    test_table_limit_uses_real_candidate_target()
    test_rejection_notice_explains_actual_constraints()
    test_rejection_notice_handles_malformed_checks()
    return False
