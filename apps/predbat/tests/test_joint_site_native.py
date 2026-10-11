"""Offline acceptance of the joint planner's isolated native-prediction bridge."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from const import PREDBAT_MAX_CARS
from prediction import Prediction
from joint_site_native import _native_windows, _options, _replay, evaluate_joint_site
from tests.test_import_limit import _fixture


def _setup(base, pv=0, minutes=60):
    """Supply synthetic, fresh native forecasts and an explicit expected EV session."""
    pred = _fixture(base, load=1.0, pv=pv, minutes=minutes)
    base.soc_kw = 20.0
    base.car_charging_rate = [6.77] * PREDBAT_MAX_CARS
    base.car_charging_limit = [3.0] * PREDBAT_MAX_CARS
    base.car_charging_soc = [0.0] * PREDBAT_MAX_CARS
    base.set_discharge_during_charge = False
    base.set_charge_freeze_only = False
    base.load_minutes_step = pred.load_minutes_step
    base.load_minutes_step10 = pred.load_minutes_step10
    base.load_minutes_step90 = pred.load_minutes_step90
    base.pv_forecast_minute_step = pred.pv_forecast_minute_step
    base.pv_forecast_minute10_step = pred.pv_forecast_minute10_step
    base.pv_forecast_minute90_step = pred.pv_forecast_minute90_step
    base.args["joint_site"] = {
        "mode": "shadow",
        "load_port_limit_w": 10000,
        "session": {
            "available_from": base.now_utc.isoformat(),
            "departure": (base.now_utc + timedelta(minutes=minutes)).isoformat(),
            "soc_kwh": 0.0,
            "soc_observed_at": base.now_utc.isoformat(),
        },
    }
    base.joint_site_report = None


def _controls(base):
    """Snapshot the executable plan, not just the shadow report's claim of isolation."""
    return deepcopy({key: getattr(base, key) for key in ("car_charging_slots", "charge_window_best", "charge_limit_best", "export_window_best", "export_limits_best", "car_charging_soc", "car_charging_soc_next")})


def test_disabled_and_invalid(base):
    """Default/off never runs a solver; unsupported modes cannot accidentally enable execution."""
    _setup(base)
    for settings in (None, {}, {"mode": "off"}):
        base.args["joint_site"] = settings
        before = _controls(base)
        with patch("joint_site_native.build_candidate", side_effect=AssertionError("Disabled solver called")):
            assert evaluate_joint_site(base) is None
        assert _controls(base) == before
    for settings in (True, "shadow", {"mode": "apply"}, {"mode": "shadow", "typo": 1}):
        base.args["joint_site"] = settings
        report = evaluate_joint_site(base)
        assert report["status"] == "invalid_input", report
        assert report["execution_enabled"] is False
        assert report["enforced"] is False


def test_fresh_session_and_availability(base):
    """Expected arrival is separate from real switching; late/stale/naive sessions cannot pass."""
    _setup(base)
    settings = deepcopy(base.args["joint_site"])
    for key, value in (
        ("soc_observed_at", (base.now_utc - timedelta(hours=25)).isoformat()),
        ("soc_observed_at", (base.now_utc + timedelta(minutes=1)).isoformat()),
        ("departure", base.now_utc.isoformat()),
        ("available_from", (base.now_utc + timedelta(hours=2)).isoformat()),
        ("departure", "2026-01-01T07:30:00"),
        ("soc_kwh", float("nan")),
    ):
        base.args["joint_site"] = deepcopy(settings)
        base.args["joint_site"]["session"][key] = value
        assert evaluate_joint_site(base)["status"] == "invalid_input"
    base.args["joint_site"] = settings
    base.args["joint_site"]["session"]["available_from"] = (base.now_utc + timedelta(minutes=35)).isoformat()
    report = evaluate_joint_site(base)
    assert report["status"] == "no_plan_found", report
    assert not report["proposed_ev_windows"]


def test_shadow_candidate_and_isolation(base):
    """A real optimisation and native replay use new objects, never the active plan or relay signal."""
    _setup(base, pv=5)
    before = _controls(base)
    base.prediction = Prediction(base, base.pv_forecast_minute_step, base.pv_forecast_minute10_step, base.load_minutes_step, base.load_minutes_step10)
    active_prediction = base.prediction
    active_prediction.prediction_cache["sentinel"] = {"untouched": [1, 2]}
    active_prediction.pending_batch.append({"untouched": True})
    cache_before = deepcopy(active_prediction.prediction_cache)
    batch_before = deepcopy(active_prediction.pending_batch)
    kernel_before = active_prediction.kernel_handle
    base.prediction_kernel_enable = True
    base.publish_car_plan()
    relay_key = "binary_sensor." + base.prefix + "_car_charging_slot"
    relay_state = deepcopy(base.dashboard_values[relay_key])
    with patch("prediction.create_kernel_context", side_effect=AssertionError("Shadow must not allocate a kernel context")):
        report = evaluate_joint_site(base)
    assert report["status"] == "candidate", report
    assert report["proposed_ev_windows"], report
    assert report["execution_enabled"] is False
    assert report["enforced"] is False
    assert report["native_checks"]["nominal"]["checked_minutes"] == base.forecast_minutes
    assert report["native_checks"]["pv10"]["checked_minutes"] == base.forecast_minutes
    assert _controls(base) == before
    assert base.prediction is active_prediction
    assert active_prediction.prediction_cache == cache_before
    assert active_prediction.pending_batch == batch_before
    assert active_prediction.kernel_handle == kernel_before
    assert base.prediction_kernel_enable is True
    assert abs(report["solver_ev_ac_kwh"] - 3 / base.car_charging_loss) < 1e-8
    assert sum(w["kwh"] for w in report["proposed_ev_windows"]) > report["solver_ev_ac_kwh"]
    assert base.dashboard_values[relay_key] == relay_state
    item = base.dashboard_values[base.prefix + ".joint_site_plan"]
    assert item["state"] == report["status"]
    assert item["attributes"]["execution_enabled"] is False
    base.args.pop("joint_site")
    assert evaluate_joint_site(base) is None
    item = base.dashboard_values[base.prefix + ".joint_site_plan"]
    assert item["state"] == "disabled"
    assert item["attributes"]["proposed_ev_windows"] == []


def test_errors_clear_old_candidates(base):
    """Missing forecasts, unexpected exceptions and unsupported loads are explicit, not stale success."""
    _setup(base)
    del base.load_minutes_step[5]
    assert evaluate_joint_site(base)["status"] == "invalid_input"
    _setup(base)
    base.iboost_enable = True
    assert evaluate_joint_site(base)["status"] == "unsupported"
    _setup(base)
    base.num_cars = 2
    assert evaluate_joint_site(base)["status"] == "unsupported"
    _setup(base)
    before = _controls(base)
    with patch("joint_site_native.build_candidate", side_effect=RuntimeError("private credential must not be logged")):
        report = evaluate_joint_site(base)
    assert report["status"] == "error"
    assert "private credential" not in str(report)
    assert report["proposed_ev_windows"] == []
    assert _controls(base) == before
    with patch("joint_site_native.build_candidate", side_effect=TypeError("private input in interface failure")):
        report = evaluate_joint_site(base)
    assert report["status"] == "error"
    assert report["failure_stage"] == "solver"
    assert report["error_type"] == "TypeError"
    assert "private input" not in str(report)


def test_publication_failure_is_nonfatal(base):
    """Even a failing shadow sensor/log sink cannot abort the native control cycle."""
    _setup(base)
    before = _controls(base)
    with (
        patch("joint_site_native.build_candidate", return_value=(SimpleNamespace(actions=()), [])),
        patch.object(base, "dashboard_item", side_effect=RuntimeError("private transport details")),
        patch.object(base, "log", side_effect=RuntimeError("logger unavailable")),
    ):
        report = evaluate_joint_site(base)
    assert report["publication_failed"] is True
    assert report["status"] == "no_plan_found"
    assert "private transport" not in str(report)
    assert _controls(base) == before


def test_native_rejects_counterexamples(base):
    """Do not trust a solver's success label: replay raw power, reserve, arrival and cooldown."""
    _setup(base)
    base.soc_kw = base.reserve
    slots = [SimpleNamespace(index=i, start_min=m, duration_min=5, import_price=10) for i, m in enumerate(range(0, 60, 5))]
    actions = [SimpleNamespace(slot_index=s.index, battery_mode="demand", ev_action="on") for s in slots]
    result = SimpleNamespace(actions=actions)
    windows = _native_windows(base, result, slots)
    before = _controls(base)
    checks = _replay(base, _options(base), result, slots, windows, 0)
    assert checks["passed"] is False
    assert checks["peak_import_w"] > 7700
    assert "site_import_exceeded" in checks["reasons"]
    assert "ev_pause_threshold" in checks["reasons"]
    base.soc_kw = 20
    actions[1].ev_action = "off"
    options = _options(base)
    options["arrival"] = 10
    options["deadline"] = 45
    options["load_cap"] = 7
    checks = _replay(base, options, result, slots, _native_windows(base, result, slots), 0)
    assert "ev_availability_or_deadline" in checks["reasons"]
    assert "ev_cooldown" in checks["reasons"]
    assert "load_port_exceeded" in checks["reasons"]
    assert _controls(base) == before


def test_actual_calculate_plan_hook(base):
    """The real calculation calls shadow evaluation after forecasts, with publication disabled."""
    _setup(base)
    base.calculate_best = False
    base.calculate_savings = False
    base.charge_limit = []
    base.plan_last_updated_minutes = base.minutes_now
    base.metric_cloud_coverage = 0
    load = base.load_minutes_step.copy()
    pv = base.pv_forecast_minute_step.copy()
    original_controls = _controls(base)

    def steps(*args, **kwargs):
        """Replace only forecast acquisition, not plan calculation or prediction."""
        return pv.copy() if kwargs.get("forward") else load.copy()

    with (
        patch.object(base, "step_data_history", side_effect=steps),
        patch.object(base, "get_cloud_factor", return_value=0),
        patch.object(base, "get_load_divergence", return_value=0),
        patch.object(base, "refresh_pv_forecast_minute90"),
        patch("joint_site_native.evaluate_joint_site", wraps=evaluate_joint_site) as hook,
    ):
        base.calculate_plan(recompute=False, publish=False)
    assert hook.call_count == 1
    assert hook.call_args.kwargs == {"publish": False}
    assert base.joint_site_report["status"] == "candidate", base.joint_site_report
    assert base.joint_site_report["proposed_ev_windows"]
    after = _controls(base)
    # The normal native simulation may update its existing next-SoC projection.
    after.pop("car_charging_soc_next")
    original_controls.pop("car_charging_soc_next")
    assert after == original_controls


def test_pv10_rejection_and_failed_replay(base):
    """A real candidate can fail the conservative native case without becoming executable."""
    _setup(base, pv=5)
    base.load_minutes_step10 = {m: 12 / 12 for m in base.load_minutes_step}
    before = _controls(base)
    report = evaluate_joint_site(base)
    assert report["status"] == "native_rejected", report
    assert report["native_checks"]["nominal"]["passed"]
    assert "load_port_exceeded" in report["native_checks"]["pv10"]["reasons"]
    assert _controls(base) == before
    with patch("joint_site_native._replay", side_effect=KeyError("private replay data")), patch.object(base, "log") as log:
        report = evaluate_joint_site(base)
    assert report["status"] == "error"
    assert report["failure_stage"] == "native_replay"
    assert report["proposed_ev_windows"] == []
    assert "private replay" not in str(report) + str(log.call_args_list)
    assert _controls(base) == before


def test_timestamp_offsets_and_cost_convention(base):
    """Inward rounding uses absolute timestamps, and horizon costs exclude standing charges."""
    _setup(base)
    session = base.args["joint_site"]["session"]
    session["available_from"] = (base.now_utc + timedelta(seconds=1)).isoformat()
    session["departure"] = (base.now_utc + timedelta(minutes=59, seconds=59)).isoformat()
    options = _options(base)
    assert options["arrival"] == 5
    assert options["deadline"] == 55
    # A timezone offset that describes the same instant cannot change eligibility.
    for name in ("available_from", "departure", "soc_observed_at"):
        session[name] = datetime.fromisoformat(session[name]).astimezone(timezone(timedelta(hours=2))).isoformat()
    assert _options(base) == options
    baseline = evaluate_joint_site(base, publish=False)
    base.metric_standing_charge = 1000
    base.cost_today_sofar = 1234
    later = evaluate_joint_site(base, publish=False)
    assert baseline["status"] == later["status"] == "candidate"
    for case in ("nominal", "pv10"):
        assert abs(baseline["native_checks"][case]["horizon_energy_cost"] - later["native_checks"][case]["horizon_energy_cost"]) < 0.001


def run_joint_site_native_tests(base):
    """Run native integration cases without a live HA connection or inverter writes."""
    for test in (
        test_disabled_and_invalid,
        test_fresh_session_and_availability,
        test_shadow_candidate_and_isolation,
        test_errors_clear_old_candidates,
        test_publication_failure_is_nonfatal,
        test_native_rejects_counterexamples,
        test_actual_calculate_plan_hook,
        test_pv10_rejection_and_failed_replay,
        test_timestamp_offsets_and_cost_convention,
    ):
        test(base)
        print("PASS:", test.__name__)
    return False
