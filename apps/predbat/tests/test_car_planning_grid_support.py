"""Opt-in grid-charge sharing models actual flows without adding execution authority."""

from tests.car_planning_fixtures import car_settings, configure_car_settings

from dataclasses import replace
from copy import deepcopy
from datetime import timedelta
from unittest.mock import patch
from types import SimpleNamespace

from car_charging_solver import BatteryConfig, EVConfig, SiteConfig, PlannerConfig, Slot, _segment, plan
from car_charging_plan import _options, _continuous_grid_support_shadow, _native_windows, _priority_block_starts, _replay, evaluate_car_charging
from prediction import Prediction
from tests.test_car_planning_native import _setup, _controls, _set_entity
from tests.test_infra import reset_rates


def test_grid_support_recorded_charge_request():
    """A synthetic 7 kW load needs battery support, not an unconstrained grid purchase."""
    battery = BatteryConfig(40, 16, 4, 3 / 0.96, 8 * 0.96, 0.95 * 0.96, 0.95 * 0.96, 4)
    ev = EVConfig(40, 4, 35, rate_kw=7, efficiency=0.9)
    site = SiteConfig(6, 0, 16, charge_import_cap_kw=6)
    flows, end = _segment(5, 1, 6, 0, battery.initial_kwh, "charge", battery, ev, site)
    assert len(flows) == 1 and abs(flows[0].grid_import_kw - 6) < 1e-10
    assert flows[0].battery_charge_kw == 0 and abs(flows[0].battery_discharge_kw - 1) < 1e-10
    assert end < battery.initial_kwh and abs(end - (battery.initial_kwh - 1 / 12 / battery.discharge_efficiency)) < 1e-10
    assert _segment(5, 1, 6, 0, battery.initial_kwh, "charge", battery, ev, SiteConfig(6, 0, 16)) is None
    for mode in ("demand", "hold"):
        assert _segment(5, 1, 0, 0, battery.initial_kwh, mode, battery, ev, site) == _segment(5, 1, 0, 0, battery.initial_kwh, mode, battery, ev, SiteConfig(6, 0, 16))


def test_grid_support_headroom_and_exhaustion():
    """Charging yields first; insufficient energy/rate never becomes a fake capped import."""
    battery = BatteryConfig(10, 5, 1, 3, 8, 0.9, 0.9, 1)
    ev = EVConfig(40, 0, 35, rate_kw=7, efficiency=0.9, pause_battery_percent=0)
    site = SiteConfig(6, 0, 16, charge_import_cap_kw=6)
    for load, pv, expected_charge, expected_discharge in ((4.5, 0, 1.5, 0), (7.5, 1, 0, 0.5), (2, 4, 3, 0)):
        flows, _ = _segment(5, load, 0, pv, 5, "charge", battery, ev, site)
        for flow in flows:
            assert abs(flow.battery_charge_kw - expected_charge) < 1e-10 and abs(flow.battery_discharge_kw - expected_discharge) < 1e-10
            assert abs(flow.grid_import_kw + pv + flow.battery_discharge_kw - load - flow.battery_charge_kw - flow.curtailed_kw) < 1e-10
    assert _segment(5, 7.5, 0, 0, 5, "charge", replace(battery, max_discharge_kw=0.5), ev, site) is None
    assert _segment(5, 7.5, 0, 0, 1.01, "charge", battery, ev, site) is None
    flows, end = _segment(5, 4.5, 0, 0, 9.99, "charge", battery, ev, site)
    assert len(flows) == 2 and abs(end - 10) < 1e-10 and all(flow.grid_import_kw <= 6 for flow in flows)
    for value in (True, float("nan"), -1, 7):
        try:
            plan([Slot(0, 0, 5, 0.1, 0, 10, 0)], ev, battery, replace(site, charge_import_cap_kw=value), PlannerConfig())
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid support cap accepted")


def _support_fixture(base):
    """Explicit shadow setting and cached entities authorise modelling, never control."""
    _setup(base, minutes=60)
    base.set_read_only = base.set_discharge_during_charge = True
    car_settings(base).update(control_model="deye_solarman", grid_charge_support={"enabled_entity": "switch.test_grid_support", "limit_entity": "number.test_grid_support"})
    _set_entity(base, "switch.test_grid_support", "on")
    _set_entity(base, "number.test_grid_support", "6000")


def test_grid_support_continuous_ev_before_solar():
    """Grid and battery can supply an uninterrupted fixed-rate EV block before departure."""
    battery = BatteryConfig(32.154, 12.21852, 3.2154, 3 / 0.96, 8.192 * 0.96, 0.97 * 0.96, 0.97 * 0.96, 3.2154)
    ev = EVConfig(25, 3.25, 3.25 + 6.77 * 0.92, rate_kw=6.77, efficiency=0.92, deadline_min=60, initial_off_minutes=10)
    slots = [Slot(index, index * 5, 5, 0.5 / 12, 0 if index < 12 else 1, 10 if index < 12 else 0, 0) for index in range(24)]
    result = plan(slots, ev, battery, SiteConfig(6, 0, 16, charge_import_cap_kw=6), PlannerConfig(allowed_battery_modes=("charge",)))
    assert result.status == "feasible" and result.diagnostics.ev_stop_target_met
    assert all(action.ev_action == "on" and abs(action.ev_on_minutes - 5) < 1e-8 for action in result.actions[:12])
    assert all(action.ev_action == "off" for action in result.actions[12:])
    assert abs(sum(action.ev_delivered_kwh for action in result.actions) - 6.77) < 1e-8
    assert all(abs(segment.ev_kw - 6.77) < 1e-8 and abs(segment.battery_discharge_kw - 1.27) < 1e-8 for action in result.actions[:12] for segment in action.segments)
    # A higher target cannot be rescued by the cheap sunshine AFTER departure.
    missed = plan(slots, replace(ev, stop_soc_kwh=25), battery, SiteConfig(6, 0, 16, charge_import_cap_kw=6), PlannerConfig())
    assert missed.status == "bounded_no_plan" and not missed.actions


def test_grid_support_options_fail_closed(base):
    """Wrong models, missing apply observations or unavailable caps fail closed."""
    _support_fixture(base)
    with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
        assert _options(base)["grid_charge_support_w"] == 6000
        for change in ({"control_model": "native"}, {"grid_charge_support": True}):
            original = deepcopy(car_settings(base))
            car_settings(base).update(change)
            try:
                _options(base)
            except ValueError:
                pass
            else:
                raise AssertionError("Grid sharing accepted outside its contract")
            configure_car_settings(base, original)
        car_settings(base)["mode"] = "apply"
        assert "measured_ev_soc_required" in _options(base)["entity_blockers"]
        car_settings(base)["mode"] = "shadow"
        for entity, value in (("switch.test_grid_support", "off"), ("number.test_grid_support", "unavailable"), ("number.test_grid_support", "nan"), ("number.test_grid_support", "6500")):
            _support_fixture(base)
            _set_entity(base, entity, value)
            assert "grid_charge_support_unavailable" in _options(base)["entity_blockers"]
    _support_fixture(base)
    before = _controls(base)
    with patch("car_charging_plan._observation_clock", return_value=base.now_utc), patch("inverter_service_control.validate_template"):
        report = evaluate_car_charging(base, publish=False)
    assert report["grid_charge_support_w"] == 6000 and not report["execution_enabled"]
    assert _controls(base) == before and not getattr(base, "plan_grid_charge_support_w", 0)
    assert getattr(base, "car_charging_pending", None) is None


def test_grid_support_continuous_candidate_is_paired(base):
    """Continuity changes real reservations, keeps both gates, and falls back on replay failure."""
    _support_fixture(base)
    before = _controls(base)
    with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
        options = _options(base)
    chosen, attempts = _continuous_grid_support_shadow(base, options)
    assert chosen is not None and 1 <= len(attempts) <= 12
    result, slots, windows, checks = chosen
    assert all(check["passed"] for check in checks.values())
    assert all(a["end"] == b["start"] for a, b in zip(windows[0], windows[0][1:]))
    assert all(abs(window["kwh"] - 6.77 / 12) < 1e-8 for window in windows[0])
    assert _controls(base) == before
    for change in ({"grid_charge_support_w": 0}, {"house_only": True}):
        assert _continuous_grid_support_shadow(base, {**options, **change}) == (None, [])
    for change in ({"mode": "apply"}, {"initially_on": True, "initial_off_minutes": 0, "deadline": 30}):
        continued, runs = _continuous_grid_support_shadow(base, {**options, **change})
        assert continued is not None and all(check["passed"] for check in continued[3].values())
        if change.get("initially_on"):
            assert len(runs) == 1 and runs[0]["start_offset"] == 0
    assert _controls(base) == before
    with patch("car_charging_plan._replay_candidate", return_value=(windows, {"nominal": {"passed": False}, "pv10": {"passed": True}})):
        rejected, attempts = _continuous_grid_support_shadow(base, options)
    assert rejected is None and all(not attempt["native_passed"] for attempt in attempts)


def _priority_test_base(rates=None, solar_kwh=0):
    """Build a compact rate/PV view for priority-window shortlist tests."""
    rates = rates or {minute: 10.0 for minute in range(1000)}
    return SimpleNamespace(
        car_charging_rate=[6.0],
        inverter_hybrid=False,
        inverter_limit=20.0,
        inverter_loss=1.0,
        load_minutes_step={minute: 0.0 for minute in range(1000)},
        load_minutes_step10={minute: 0.0 for minute in range(1000)},
        minutes_now=0,
        pv_ac_limit=0.0,
        pv_forecast_minute_step={minute: solar_kwh for minute in range(1000)},
        pv_forecast_minute10_step={minute: solar_kwh for minute in range(1000)},
        rate_export={minute: 0.0 for minute in range(1000)},
        rate_import=rates,
    )


def test_cost_priority_shortlist_retains_baseline_and_samples_ties():
    """The prior twelve cost starts remain while twelve stratified ties extend coverage."""
    base = _priority_test_base()
    options = {"ev_charging_priority": "cost", "initially_on": False}

    selected = _priority_block_starts(base, options, first=0, last=775, duration=130)
    starts = [start for start, _ in selected]

    assert starts[:12] == list(range(0, 60, 5))
    assert 710 in starts
    assert len(starts) == len(set(starts))
    assert len(starts) <= 24


def test_cost_priority_shortlist_keeps_cheaper_starts_and_other_priorities():
    """Every strictly cheaper candidate survives without changing non-cost priorities."""
    rates = {minute: 10.0 for minute in range(1000)}
    cheaper_rates = {0: 1.0, 5: 2.0, 10: 3.0, 15: 4.0}
    rates.update(cheaper_rates)
    base = _priority_test_base(rates)
    cost_options = {"ev_charging_priority": "cost", "initially_on": False}
    selected = _priority_block_starts(base, cost_options, first=0, last=775, duration=5)
    starts = [start for start, _ in selected]

    assert set(cheaper_rates).issubset(starts)
    assert len(starts) == len(set(starts))
    assert len(starts) <= 24
    assert _priority_block_starts(base, {"ev_charging_priority": "now", "initially_on": False}, 0, 60, 5) == [(0, "immediate")]
    assert _priority_block_starts(base, {"ev_charging_priority": "asap", "initially_on": False}, 0, 60, 5) == [(start, "deadline_or_no_opportunity") for start in range(0, 60, 5)]
    assert _priority_block_starts(_priority_test_base(solar_kwh=1.0), {"ev_charging_priority": "solar", "initially_on": False}, 0, 60, 5) == [(start, "solar_surplus_only") for start in range(0, 60, 5)]


def test_cost_priority_tie_search_samples_late_candidate():
    """Equal EV-cost starts are spread so a later coupled-cost winner stays in the bounded search."""
    base = SimpleNamespace(
        car_charging_limit=[13.0],
        car_charging_loss=1.0,
        car_charging_rate=[6.0],
        minutes_now=0,
        rate_export={minute: 0.0 for minute in range(1000)},
        rate_import={minute: 10.0 for minute in range(1000)},
    )
    options = {
        "arrival": 0,
        "control_model": "deye_solarman",
        "deadline": 905,
        "ev_charging_priority": "cost",
        "ev_soc": 0.0,
        "grid_charge_support_w": 6000,
        "house_only": False,
        "initial_off_minutes": 10,
        "initially_on": False,
        "mode": "shadow",
        "off_minutes": 10,
        "self_consumption_first": False,
    }

    def build_candidate(_, trial):
        """Make the omitted late block cheaper while keeping every trial structurally valid."""
        block = trial.get("ev_continuous_block")
        cost = 0.0 if block and block[0] == 710 else 10.0 if block else 100.0
        result = SimpleNamespace(
            actions=[SimpleNamespace(battery_mode="demand", discharge_limit_kw=None)],
            status="feasible",
            total_cost=cost,
        )
        return result, []

    def replay_candidate(_, trial, result, slots):
        """Treat fixed blocks as paired-safe; the unrestricted fallback remains rejected."""
        passed = trial.get("ev_continuous_block") is not None
        checks = {name: {"passed": passed} for name in ("nominal", "pv10")}
        return ([], [], []), checks

    with (
        patch("car_charging_plan._solar_surplus_kwh", return_value=0),
        patch("car_charging_plan.build_candidate", side_effect=build_candidate),
        patch("car_charging_plan._replay_candidate", side_effect=replay_candidate),
    ):
        chosen, attempts = _continuous_grid_support_shadow(base, options)

    assert chosen is not None and chosen[0].total_cost == 0.0
    assert any(attempt.get("selected") and attempt.get("start_offset") == 710 for attempt in attempts)
    assert len([attempt for attempt in attempts if "start_offset" in attempt]) <= 24


def test_cost_priority_compares_safe_initial_demand_with_ordinary_candidate():
    """A feasible current Demand step must not hide cheaper, paired-safe charging now."""
    base = _priority_test_base()
    base.car_charging_limit = [13.0]
    base.car_charging_loss = 1.0
    options = {
        "arrival": 0,
        "beam": 512,
        "control_model": "deye_solarman",
        "deadline": 905,
        "ev_charging_priority": "cost",
        "ev_soc": 0.0,
        "grid_charge_support_w": 6000,
        "house_only": False,
        "initial_off_minutes": 10,
        "initially_on": False,
        "mode": "apply",
        "off_minutes": 10,
        "self_consumption_first": True,
    }

    def build(_, trial):
        constrained = trial.get("initial_demand_only", False)
        mode = "demand" if constrained else "charge"
        cost = 2.0 if constrained else ordinary_cost
        if "ev_continuous_block" not in trial:
            cost = 100.0
        return SimpleNamespace(actions=[SimpleNamespace(battery_mode=mode, discharge_limit_kw=None)], status="feasible", total_cost=cost), []

    for ordinary_safe, ordinary_cost, expected_mode, expected_cost in ((True, 1.0, "charge", 1.0), (True, 2.0, "demand", 2.0), (False, 1.0, "demand", 2.0)):

        def replay(_, trial, result, slots):
            safe = trial.get("initial_demand_only", False) or ordinary_safe
            return ([], [], []), {"nominal": {"passed": True}, "pv10": {"passed": safe}}

        with (
            patch("car_charging_plan._priority_block_starts", return_value=[(0, "coupled_cost")]),
            patch("car_charging_plan.build_candidate", side_effect=build),
            patch("car_charging_plan._replay_candidate", side_effect=replay),
        ):
            chosen, attempts = _continuous_grid_support_shadow(base, options)

        assert chosen is not None and chosen[0].total_cost == expected_cost
        assert chosen[0].actions[0].battery_mode == expected_mode
        assert len([attempt for attempt in attempts if "start_offset" in attempt]) == 2


def test_apply_cost_refines_representative_starts_even_after_a_feasible_result():
    """Feasibility at a narrow beam must not prevent bounded economic refinement."""
    base = _priority_test_base()
    base.car_charging_limit = [13.0]
    base.car_charging_loss = 1.0
    options = {
        "arrival": 0,
        "beam": 128,
        "control_model": "deye_solarman",
        "deadline": 905,
        "ev_charging_priority": "cost",
        "ev_soc": 0.0,
        "grid_charge_support_w": 6000,
        "house_only": False,
        "initial_off_minutes": 10,
        "initially_on": False,
        "mode": "apply",
        "off_minutes": 10,
        "self_consumption_first": False,
    }
    ranked = [(index * 5, "coupled_cost") for index in range(24)]

    for refined_safe in (True, False):
        calls = []

        def build(_, trial):
            block = trial.get("ev_continuous_block")
            calls.append((trial["beam"], block))
            cost = 1.0 if trial["beam"] == 512 and block and block[0] == 115 else 10.0
            if block is None:
                cost = 100.0
            result = SimpleNamespace(actions=[SimpleNamespace(battery_mode="demand", discharge_limit_kw=None)], status="feasible", total_cost=cost)
            return result, []

        def replay(_, trial, result, slots):
            safe = trial["beam"] != 512 or refined_safe
            return ([], [], []), {"nominal": {"passed": True}, "pv10": {"passed": safe}}

        with (
            patch("car_charging_plan._priority_block_starts", return_value=ranked),
            patch("car_charging_plan.build_candidate", side_effect=build),
            patch("car_charging_plan._replay_candidate", side_effect=replay),
        ):
            chosen, attempts = _continuous_grid_support_shadow(base, options)

        assert chosen is not None and chosen[0].total_cost == (1.0 if refined_safe else 10.0)
        assert {block[0] for beam, block in calls if beam == 128 and block} == {start for start, _ in ranked}
        assert [(beam, block[0]) for beam, block in calls if beam == 512] == [(512, start) for start in [0, 35, 75, 115]]
        selected = next(attempt for attempt in attempts if attempt.get("selected"))
        assert selected["beam_width"] == (512 if refined_safe else 128)
        assert len(attempts) == 29


def test_grid_support_replay_rejects_hidden_reserve_tail(base):
    """The independent bridge uses exact intra-step peaks even without companion import diagnostics."""
    _support_fixture(base)
    base.forecast_minutes = base.end_record = 5
    base.soc_kw = base.reserve + 0.1
    base.car_charging_limit = [0] * len(base.car_charging_limit)
    base.load_minutes_step = base.load_minutes_step10 = {0: 6.5 / 12}
    base.pv_forecast_minute_step = base.pv_forecast_minute10_step = {0: 0}
    options = dict(
        mode="shadow", control_model="deye_solarman", grid_charge_support_w=5000, ev_soc=0, import_limit_w=6000, terminal_min=base.reserve, initially_on=False, initial_off_minutes=10, pause=15, restart=25, arrival=0, deadline=5, off_minutes=10, load_cap=16
    )
    result = SimpleNamespace(actions=[SimpleNamespace(slot_index=0, battery_mode="charge", ev_action="off")])
    slots = [Slot(0, 0, 5, 6.5 / 12, 0, 10, 0)]
    original = Prediction.run_prediction

    def without_diagnostic(pred, *args, **kwargs):
        """Simulate independent installation of the combined PR without the diagnostic PR."""
        returned = original(pred, *args, **kwargs)
        pred.import_limit_diagnostics = None
        return returned

    with patch.object(Prediction, "run_prediction", without_diagnostic):
        check = _replay(base, options, result, slots, _native_windows(base, result, slots), 0)
    assert not check["passed"] and "site_import_exceeded" in check["reasons"]
    assert abs(check["peak_import_w"] - 6500) < 1e-6 and check["checked_minutes"] == 5


def test_grid_support_capped_plan_fails_closed_without_pv_and_accepts_solar(base):
    """A solver-derived ceiling cannot bypass the cap; PV can make the same EV block safe."""
    safe_solar_plan = None
    for sunny in (False, True):
        _support_fixture(base)
        base.forecast_minutes = base.end_record = 360
        reset_rates(base, 10, 0)
        base.rate_import.update({base.minutes_now + minute: 40 for minute in range(180, 360)})
        base.soc_kw = base.soc_max * 0.40
        # Hold the same terminal floor without rewarding extra end-of-horizon stock.
        base.metric_battery_value_scaling = 0
        base.load_minutes_step = base.load_minutes_step10 = {minute: 2 / 12 for minute in range(0, 360, 5)}
        base.pv_forecast_minute_step = base.pv_forecast_minute10_step = {minute: 4 / 12 if sunny and minute >= 180 else 0 for minute in range(0, 360, 5)}
        with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
            options = _options(base)
        chosen, attempts = _continuous_grid_support_shadow(base, options)
        if not sunny:
            assert chosen is None
            assert attempts and all(not attempt["native_passed"] for attempt in attempts)
            assert any("site_import_exceeded" in reasons for attempt in attempts for reasons in attempt["native_reasons"].values())
            continue

        assert chosen is not None
        result, slots, windows, checks = chosen
        assert all(check["passed"] and check["final_ev_kwh"] >= 3 for check in checks.values())
        assert all(window["end"] <= base.minutes_now + 60 for window in windows[0])
        assert all(limit < base.soc_max for limit in windows[2])
        assert any(action.pv_kwh > 0 for action in result.actions)
        safe_solar_plan = result
    assert safe_solar_plan is not None


def test_initial_demand_is_a_scoped_search_constraint():
    """Consistent demand-first actions can still use later grid preparation for the same floor."""
    slots = [Slot(index, index * 5, 5, 0, 0, 10, 0) for index in range(3)]
    battery = BatteryConfig(10, 2, 1, 3, 8, 1, 1, 2.4, terminal_value_per_kwh=100)
    ev = EVConfig(40, 0, 0, rate_kw=7, efficiency=0.9, deadline_min=15)
    site = SiteConfig(6, 0, 16)
    config = PlannerConfig(beam_width=None, allowed_battery_modes=("demand", "charge"))
    default = plan(slots, ev, battery, site, config)
    preferred = plan(slots, ev, battery, site, replace(config, initial_battery_mode="demand"))
    assert default.actions[0].battery_mode == "charge"
    assert preferred.status == "feasible" and preferred.actions[0].battery_mode == "demand"
    assert preferred.actions[0].grid_import_kwh == 0 and preferred.actions[0].battery_soc_kwh_end == 2
    assert any(action.battery_mode == "charge" for action in preferred.actions[1:])
    assert preferred.diagnostics.final_battery_soc_kwh >= 2.4
    for mode in (True, "unsupported", "hold"):
        try:
            plan(slots, ev, battery, site, replace(config, initial_battery_mode=mode))
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported initial mode accepted")


def test_self_consumption_preference_is_paired_and_explicit(base):
    """Opt-in sources preference changes the search, not source settings or replay guards."""
    _support_fixture(base)
    car_settings(base).update(self_consumption_first=True, ev_charging_priority="asap")
    before = deepcopy(base.args)
    with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
        options = _options(base)
    chosen, attempts = _continuous_grid_support_shadow(base, options)
    assert chosen is not None and chosen[0].actions[0].battery_mode == "demand"
    assert attempts[0]["initial_demand_only"] is True and attempts[0]["native_passed"]
    assert all(check["passed"] for check in chosen[3].values())
    assert base.args == before and not getattr(base, "car_charging_pending", None)
    # Force the initial-demand trial to fail independent replay. The fallback
    # must be separately checked, rather than hiding the rejection or dropping a gate.
    from car_charging_plan import _replay_candidate as original

    def reject_demand(base, trial, result, slots):
        """Inject a paired-check failure only in the preferred trial."""
        windows, checks = original(base, trial, result, slots)
        if trial.get("initial_demand_only"):
            checks["pv10"]["passed"] = False
        return windows, checks

    with patch("car_charging_plan._replay_candidate", side_effect=reject_demand):
        fallback, attempts = _continuous_grid_support_shadow(base, options)
    assert fallback is not None and all(check["passed"] for check in fallback[3].values())
    assert attempts[0]["initial_demand_only"] and not attempts[0]["native_passed"]
    assert not attempts[1]["initial_demand_only"] and attempts[1]["native_passed"]
    assert base.args == before
    for value in ("yes", 1, None):
        car_settings(base)["self_consumption_first"] = value
        with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
            try:
                _options(base)
            except ValueError:
                pass
            else:
                raise AssertionError("Non-boolean source preference accepted")


def test_joint_native_reference_reuses_identical_ev_pattern_and_native_battery_windows(base):
    """The cost baseline is the current native battery schedule plus the candidate EV schedule."""
    from car_charging_household import _replay_native_joint_plan

    _setup(base, minutes=10)
    slots = [Slot(index, index * 5, 5, 0.1, 0, 10, 0) for index in range(2)]
    candidate = SimpleNamespace(
        actions=(
            SimpleNamespace(slot_index=0, battery_mode="demand", ev_action="on", discharge_limit_kw=None),
            SimpleNamespace(slot_index=1, battery_mode="demand", ev_action="off", discharge_limit_kw=None),
        )
    )
    ev_windows = _native_windows(base, candidate, slots)[0]
    snapshot = {
        "plan_valid": True,
        "charge_windows": [{"start": base.minutes_now + 5, "end": base.minutes_now + 10}],
        "charge_limits": [base.soc_max],
    }
    replayed = []

    def capture_replay(_base, _options, result, observed_slots, windows, scenario):
        """Record the replay contract without replacing its schedule construction."""
        replayed.append((result, observed_slots, windows, scenario))
        return {"passed": True, "reasons": [], "checked_minutes": 10}

    with patch("car_charging_plan._replay", side_effect=capture_replay):
        result, windows, checks = _replay_native_joint_plan(base, {}, snapshot, candidate, slots, ev_windows)

    assert windows[0] == ev_windows
    assert [(action.battery_mode, action.ev_action) for action in result.actions] == [("demand", "on"), ("charge", "off")]
    assert [item[2][0] for item in replayed] == [ev_windows, ev_windows]
    assert [item[3] for item in replayed] == [0, 1]
    assert all(check["passed"] for check in checks.values())


def test_joint_cost_improvement_uses_raw_paired_vectors_and_switch_floor():
    """Numerical noise, one-scenario savings, and extra switching cannot qualify COST."""
    from car_charging_household import _joint_cost_improvement

    reference = {"nominal": 100.0, "pv10": 110.0}
    assert _joint_cost_improvement({"nominal": 98.5, "pv10": 109.5}, reference, 2, 0) is False
    assert _joint_cost_improvement({"nominal": 98.5, "pv10": 110.0}, reference, 0, 0) is True
    assert _joint_cost_improvement({"nominal": 99.5, "pv10": 109.5}, reference, 0, 0) is False
    assert _joint_cost_improvement({"nominal": 101.0, "pv10": 90.0}, reference, 0, 0) is False
    assert _joint_cost_improvement({"nominal": 96.9, "pv10": 110.0}, reference, 2, 0) is True


def test_incident_shaped_early_100_at_19_1_is_rejected_for_native_88_at_8_1(base):
    """Synthetic paired replay rejects early full charging against a cheaper 88% native band."""
    from car_charging_plan import _search_joint_cost_candidates

    _setup(base, minutes=10)
    base.metric_battery_cycle = 0
    base.metric_battery_value_scaling = 0
    base.battery_value_rate = lambda _minute: 0
    base.rate_import.update({base.minutes_now: 19.1, base.minutes_now + 5: 8.1})
    options = {"beam": 512, "mode": "apply", "ev_charging_priority": "cost", "control_model": "deye_solarman"}
    slots = [Slot(0, 0, 5, 0, 0, 19.1, 0), Slot(1, 5, 5, 0, 0, 8.1, 0)]
    candidate = SimpleNamespace(
        status="feasible",
        total_cost=0,
        actions=(
            SimpleNamespace(slot_index=0, battery_mode="charge", ev_action="off", discharge_limit_kw=None),
            SimpleNamespace(slot_index=1, battery_mode="demand", ev_action="on", discharge_limit_kw=None),
        ),
    )
    ev_windows = _native_windows(base, candidate, slots)[0]
    candidate_windows = (
        ev_windows,
        [{"start": base.minutes_now, "end": base.minutes_now + 5, "average": 19.1}],
        [base.soc_max],
    )
    candidate_checks = {name: {"passed": True, "reasons": [], "checked_minutes": 10, "horizon_energy_cost": 19.1, "battery_cycle_kwh": 0, "final_battery_kwh": 10, "final_ev_kwh": 1} for name in ("nominal", "pv10")}
    reference_checks = {name: {"passed": True, "reasons": [], "checked_minutes": 10, "horizon_energy_cost": 8.1, "battery_cycle_kwh": 0, "final_battery_kwh": 10, "final_ev_kwh": 1} for name in ("nominal", "pv10")}
    native_snapshot = {
        "plan_valid": True,
        "charge_windows": [{"start": base.minutes_now + 5, "end": base.minutes_now + 10, "average": 8.1, "target": base.soc_max * 0.88}],
        "charge_limits": [base.soc_max * 0.88],
    }
    reference_actions = []

    def replay(_base, _options, result, _slots, windows, scenario):
        """Return complete generic-horizon evidence, keeping the candidate EV reservation fixed."""
        assert windows[0] == ev_windows
        if result.status == "ordinary_native_joint_reference":
            reference_actions.append([(action.battery_mode, action.ev_action) for action in result.actions])
            return deepcopy(reference_checks["nominal" if scenario == 0 else "pv10"])
        raise AssertionError("Unexpected replay source in synthetic reference test")

    with (
        patch("car_charging_plan._replay_candidate", return_value=(candidate_windows, candidate_checks)),
        patch("car_charging_plan._replay", side_effect=replay),
    ):
        selected, attempts = _search_joint_cost_candidates(base, options, native_snapshot, (candidate, slots), [])

    assert selected is not None and selected["source"] == "native_reference_alternative"
    assert native_snapshot["charge_limits"] == [base.soc_max * 0.88]
    assert candidate_windows[2] == [base.soc_max]  # early proposed full target, not incident replay evidence
    assert selected["commands"][0]["target_soc"] == 0
    assert selected["commands"][1]["target_soc"] == 88
    assert selected["economic_comparison"]["proposed_candidate_costs"] == {"nominal": 19.1, "pv10": 19.1}
    assert selected["economic_comparison"]["candidate_costs"] == {"nominal": 8.1, "pv10": 8.1}
    assert selected["economic_comparison"]["battery_plan_improvement"] is False
    assert selected["economic_comparison"]["same_ev_delivery"] is True
    assert reference_actions == [[("demand", "off"), ("charge", "on")]] * 2
    assert attempts[0]["economic_rejection"] == "no_material_battery_improvement"


def test_joint_cost_search_keeps_configured_and_continuous_candidates_when_512_fails(base):
    """A failed unrestricted refinement cannot erase an earlier continuous winner or its reasons."""
    from car_charging_plan import _search_joint_cost_candidates

    _setup(base, minutes=10)
    base.metric_battery_cycle = 0
    base.metric_battery_value_scaling = 0
    base.battery_value_rate = lambda _minute: 0
    options = {"beam": 128, "mode": "apply", "ev_charging_priority": "cost", "control_model": "deye_solarman"}
    slots = [Slot(index, index * 5, 5, 0.1, 0, 10, 0) for index in range(2)]

    def make_result(label, mode="demand"):
        """Build a distinguishable result without depending on solver rank."""
        actions = tuple(SimpleNamespace(slot_index=index, battery_mode=mode, ev_action="off", discharge_limit_kw=None) for index in range(2))
        return SimpleNamespace(status="feasible", actions=actions, total_cost=label, diagnostics=SimpleNamespace(ev_delivered_total_kwh=0, pv_curtailed_total_kwh=0), label=label)

    configured = make_result("configured")
    continuous = make_result("continuous", "charge")
    refined = make_result("refined")
    safe_checks = {name: {"passed": True, "reasons": [], "checked_minutes": 10, "horizon_energy_cost": 98.0, "battery_cycle_kwh": 0, "final_battery_kwh": 0, "final_ev_kwh": 3} for name in ("nominal", "pv10")}
    baseline_checks = {name: {"passed": True, "reasons": [], "checked_minutes": 10, "horizon_energy_cost": 100.0, "battery_cycle_kwh": 0, "final_battery_kwh": 0, "final_ev_kwh": 3} for name in ("nominal", "pv10")}
    refined_failure = {name: {"passed": False, "reasons": ["site_import_exceeded"], "checked_minutes": 10} for name in ("nominal", "pv10")}
    windows = ([], [], [])
    continuous_attempt = {
        "beam_width": 128,
        "native_passed": True,
        "_candidate": (continuous, slots, windows, safe_checks, {**options, "beam": 128}, "continuous_block"),
    }
    baseline_result = SimpleNamespace(status="ordinary_native_joint_reference", actions=tuple(SimpleNamespace(slot_index=i, battery_mode="demand", ev_action="off", discharge_limit_kw=None) for i in range(2)))
    build_calls = []

    def build(_base, trial):
        """Return only the unrestricted 512 refinement requested by the search."""
        build_calls.append(trial["beam"])
        return refined, slots

    def replay(_base, _trial, result, _slots):
        """Keep configured evidence and make only the wider replay unsafe."""
        return windows, baseline_checks if result is configured else refined_failure

    with (
        patch("car_charging_plan.build_candidate", side_effect=build),
        patch("car_charging_plan._replay_candidate", side_effect=replay),
        patch("car_charging_household._replay_native_joint_plan", return_value=(baseline_result, ([], [], []), baseline_checks)),
    ):
        selected, attempts = _search_joint_cost_candidates(
            base,
            options,
            {"plan_valid": True, "charge_windows": [], "charge_limits": []},
            (configured, slots),
            [continuous_attempt],
        )

    assert build_calls == [512]
    assert selected["source"] == "continuous_block"
    assert [attempt["beam_width"] for attempt in attempts if "beam_width" in attempt] == [128, 128, 512]
    assert attempts[-1]["native_reasons"] == {"nominal": ["site_import_exceeded"], "pv10": ["site_import_exceeded"]}


def test_joint_cost_search_retains_equal_native_battery_plan_for_ev_delivery(base):
    """An equal-cost, paired-safe native battery schedule remains an executable EV alternative."""
    from car_charging_plan import _search_joint_cost_candidates

    _setup(base, minutes=10)
    base.metric_battery_cycle = 0
    base.metric_battery_value_scaling = 0
    base.battery_value_rate = lambda _minute: 0
    options = {"beam": 512, "mode": "apply", "ev_charging_priority": "cost", "control_model": "deye_solarman"}
    slots = [Slot(index, index * 5, 5, 0.1, 0, 10, 0) for index in range(2)]
    result = SimpleNamespace(
        status="feasible",
        actions=tuple(SimpleNamespace(slot_index=index, battery_mode="demand", ev_action="on" if index == 0 else "off", discharge_limit_kw=None) for index in range(2)),
        total_cost=-1000,
        diagnostics=SimpleNamespace(ev_delivered_total_kwh=1, pv_curtailed_total_kwh=0),
    )
    checks = {name: {"passed": True, "reasons": [], "checked_minutes": 10, "horizon_energy_cost": 100, "battery_cycle_kwh": 0, "final_battery_kwh": 5, "final_ev_kwh": 3, "validated_ev_target_kwh": 3} for name in ("nominal", "pv10")}
    windows = ([{"start": base.minutes_now, "end": base.minutes_now + 5, "kwh": 1}], [], [])
    native_result = SimpleNamespace(
        status="ordinary_native_joint_reference",
        actions=tuple(SimpleNamespace(slot_index=index, battery_mode="demand", ev_action="on" if index == 0 else "off", discharge_limit_kw=None) for index in range(2)),
        total_cost=None,
        diagnostics=SimpleNamespace(ev_delivered_total_kwh=1, pv_curtailed_total_kwh=0),
    )

    with (
        patch("car_charging_plan._replay_candidate", return_value=(windows, checks)),
        patch("car_charging_household._replay_native_joint_plan", return_value=(native_result, windows, checks)),
    ):
        selected, attempts = _search_joint_cost_candidates(
            base,
            options,
            {"plan_valid": True, "charge_windows": [], "charge_limits": []},
            (result, slots),
            [],
        )

    assert selected["source"] == "native_reference_alternative"
    assert selected["commands"] == [
        {"start": base.minutes_now, "end": base.minutes_now + 5, "target_soc": 0, "freeze": False, "power_w": None, "ev_on": True},
        {"start": base.minutes_now + 5, "end": base.minutes_now + 10, "target_soc": 0, "freeze": False, "power_w": None, "ev_on": False},
    ]
    assert selected["economic_comparison"]["battery_plan_improvement"] is False
    assert selected["economic_comparison"]["status"] == "native_reference_alternative"
    assert any(attempt.get("economic_rejection") == "no_material_battery_improvement" for attempt in attempts)


def _household_template_fixture(base, minutes=360, economic_fallback=True):
    """Build the actual verified template with no configured EV permission or SOC source."""
    from tests.test_car_planning_combined import _template_inputs

    inverter, writes, _, _ = _template_inputs(base, minutes=minutes, pv=0)
    settings = car_settings(base)
    for name in ("session", "soc_entity", "soc_observed_at_entity", "permission_entity", "connection_entity", "connection_observed_at_entity"):
        settings.pop(name, None)
    settings.update(
        mode="apply",
        control_model="deye_solarman",
        apply_verified=True,
        household_fallback=True,
        ev_executor_lease_verified=False,
        grid_charge_support={"enabled_entity": "switch.combined_test_grid_support", "limit_entity": "number.combined_test_grid_support"},
        import_limit_w=6000,
    )
    base.args["car_charging_household_economic_fallback"] = economic_fallback
    base.ha_interface.is_alive = lambda: True

    def live_state_api(endpoint, **_kwargs):
        """Mirror HA's GET /api/states/<entity_id> response for configured controls."""
        if not isinstance(endpoint, str) or not endpoint.startswith("/api/states/"):
            return None
        entity = endpoint.removeprefix("/api/states/")
        payload = base.ha_interface.get_state(entity_id=entity, raw=True)
        if not isinstance(payload, dict):
            return None
        last_changed = payload.get("last_changed")
        last_updated = payload.get("last_updated")
        return {
            "entity_id": entity,
            "state": payload.get("state"),
            "last_changed": last_changed.isoformat() if hasattr(last_changed, "isoformat") else last_changed,
            "last_updated": last_updated.isoformat() if hasattr(last_updated, "isoformat") else last_updated,
            "attributes": payload.get("attributes", {}),
        }

    base.ha_interface.api_call = live_state_api
    _set_entity(base, "switch.combined_test_grid_support", "on")
    _set_entity(base, "number.combined_test_grid_support", "6000")
    _set_entity(base, "binary_sensor.combined_test_relay", "off", last_changed=base.now_utc - timedelta(minutes=10))
    base.soc_kw = 4.0
    base.metric_battery_value_scaling = 0
    base.plan_valid = True
    target = min(base.soc_max, base.soc_kw + 3)
    start = base.minutes_now + (120 if minutes >= 360 else 0)
    base.charge_window_best = [{"start": start, "end": start + min(60, minutes), "average": 8.1, "target": target}]
    base.charge_limit_best = [target]
    base.export_window_best = []
    base.export_limits_best = []
    return inverter, writes


def test_household_authorized_metered_route_does_not_require_ev_reads(base):
    """The existing metered control-authorisation route is not a commissioning attestation."""
    from car_charging_household import evaluate_household_fallback, household_options

    _household_template_fixture(base)
    settings = car_settings(base)
    settings.update(
        apply_verified=False,
        execution_authorised=True,
    )
    reset_rates(base, 10, 0)
    base.rate_import.update({base.minutes_now + minute: 40 for minute in range(120, 360)})
    reads = []
    original = base.ha_interface.api_call

    def record_read(endpoint, **kwargs):
        """Record exact live HA state reads so this proves EV evidence is not consulted."""
        reads.append(endpoint.removeprefix("/api/states/"))
        return original(endpoint, **kwargs)

    with patch.object(base.ha_interface, "api_call", side_effect=record_read), patch("car_charging_plan._observation_clock", return_value=base.now_utc):
        options = household_options(base, settings)
        report = evaluate_household_fallback(base, settings, "connection_not_confirmed")

    assert options["domestic_only"] is True
    assert options["settings"]["apply_verified"] is False
    assert report["status"] == "candidate", report
    assert "execution_authorised_or_apply_commissioning" in report["activation_requirements"]
    assert "apply_commissioning" not in report["activation_requirements"]
    assert set(reads) == {
        "binary_sensor.combined_test_relay",
        "switch.combined_test_grid_support",
        "number.combined_test_grid_support",
    }
    assert "sensor.combined_test_ev_soc" not in reads
    assert "switch.combined_test_permission" not in reads


def test_household_fallback_refuses_to_stage_without_a_valid_native_plan(base):
    """An empty or invalid ordinary plan cannot be replaced by a fabricated Demand baseline."""
    from car_charging_household import evaluate_household_fallback

    _household_template_fixture(base, minutes=60)
    base.charge_window_best = []
    base.charge_limit_best = []
    searched = []

    def record_search(*_args):
        """Record whether fallback improperly searches without a native reference."""
        searched.append(True)
        return None

    with patch("car_charging_plan._observation_clock", return_value=base.now_utc), patch("car_charging_household._solve_and_replay", side_effect=record_search):
        report = evaluate_household_fallback(base, car_settings(base), "connection_not_confirmed")

    assert report["execution_reason"] == "native_baseline_unavailable", report
    assert report.get("native_comparison", {}).get("status") == "baseline_unavailable"
    assert searched == []
    assert base.car_charging_pending is None


def test_household_native_baseline_replays_the_pre_fallback_plan_snapshot(base):
    """The reported native reference is the saved ordinary plan, not candidate or all-Demand windows."""
    import car_charging_plan
    from car_charging_household import evaluate_household_fallback

    _household_template_fixture(base, minutes=360)
    original_windows = deepcopy(base.charge_window_best)
    original_limits = deepcopy(base.charge_limit_best)
    replayed_native = []
    original_replay = car_charging_plan._replay

    def record_replay(observed_base, options, result, slots, windows, scenario):
        """Capture the actual native schedule passed into both independent replays."""
        if getattr(result, "status", None) == "ordinary_native_household_plan":
            replayed_native.append((deepcopy(windows[1]), deepcopy(windows[2])))
        return original_replay(observed_base, options, result, slots, windows, scenario)

    def search_mutates_native_state(*_args):
        """Model a fallback candidate builder replacing shared plan fields after the snapshot."""
        base.charge_window_best = [{"start": base.minutes_now, "end": base.minutes_now + 5, "target": base.soc_max}]
        base.charge_limit_best = [base.soc_max]
        return None

    with (
        patch("car_charging_plan._observation_clock", return_value=base.now_utc),
        patch("car_charging_plan._replay", side_effect=record_replay),
        patch("car_charging_household._solve_and_replay", side_effect=search_mutates_native_state),
    ):
        report = evaluate_household_fallback(base, car_settings(base), "connection_not_confirmed")

    comparison = report.get("native_comparison", {})
    assert comparison.get("baseline_kind") == "ordinary_native_household_plan"
    assert comparison.get("baseline_charge_windows") == original_windows
    assert comparison.get("baseline_charge_limits") == original_limits
    assert replayed_native == [(original_windows, original_limits), (original_windows, original_limits)]
    assert report["demand_comparison"].get("baseline_kind") == "direct_all_demand_counterfactual"
    assert report["demand_comparison"].get("baseline_executable") is False


def test_household_baseline_uses_native_schedule_after_current_cycle_recompute(base):
    """Fallback economics bind to calculate_plan's new schedule, not the previous incumbent or test pollution."""
    from unit_test import create_predbat

    base = create_predbat()
    _household_template_fixture(base, minutes=1440)
    base.args["threads"] = 0
    now = base.minutes_now
    previous_windows = [{"start": now + 35, "end": now + 55, "average": 8.1, "target": base.soc_max}]
    current_window = {"start": now + 5, "end": now + 25, "average": 5.0, "target": base.soc_max}
    base.plan_valid = True
    base.plan_made_by_version = "previous-cycle-plan"
    base.update_pending = False
    base.current_charge_limit = 100.0
    base.charge_window_best = deepcopy(previous_windows)
    base.charge_limit_best = [base.soc_max]
    base.export_window_best = []
    base.export_limits_best = []
    base.calculate_best_charge = False
    base.calculate_best_export = False
    base.charge_window = [deepcopy(current_window)]
    base.charge_limit = [base.soc_max]
    base.export_window = []
    base.export_limits = []
    native_snapshots = []
    from car_charging_household import _snapshot_native_plan

    def capture_native_snapshot(observed_base):
        """Capture the exact snapshot helper used at calculate_plan's fallback boundary."""
        snapshot = _snapshot_native_plan(observed_base)
        native_snapshots.append(deepcopy(snapshot))
        return snapshot

    with (
        patch("car_charging_plan._observation_clock", return_value=base.now_utc),
        patch("car_charging_household._snapshot_native_plan", side_effect=capture_native_snapshot),
        patch("car_charging_household._solve_and_replay", return_value=None),
    ):
        base.calculate_plan(recompute=True, publish=True)

    current_windows = deepcopy(base.charge_window_best)
    current_limits = deepcopy(base.charge_limit_best)
    assert current_windows != previous_windows, (previous_windows, current_windows)
    expected_snapshot = {"plan_valid": True, "charge_windows": current_windows, "charge_limits": current_limits}
    assert native_snapshots == [expected_snapshot], (
        native_snapshots,
        base.car_charging_report,
        {name: getattr(base, name) for name in ("plan_valid", "charge_window_best", "charge_limit_best", "export_window_best", "export_limits_best", "forecast_minutes")},
    )
    comparison = base.car_charging_report["native_comparison"]
    assert comparison["baseline_charge_windows"] == current_windows
    assert comparison["baseline_charge_limits"] == current_limits
    assert comparison["baseline_initial_battery_soc_kwh"] == base.soc_kw


def test_household_economic_gate_rejects_nominal_regression_hidden_by_pv10_savings(base):
    """A PV10 improvement cannot authorize a candidate that is costlier than native nominal."""
    from car_charging_household import evaluate_household_fallback

    _household_template_fixture(base, minutes=60)
    base.metric_battery_cycle = 0
    base.metric_battery_value_scaling = 0
    options = {"mode": "apply", "grid_charge_support_w": 6000, "import_limit_w": 6000}
    action = SimpleNamespace(slot_index=0, battery_mode="charge", ev_action="off", discharge_limit_kw=None)
    result = SimpleNamespace(actions=(action,), status="candidate", total_cost=None)
    slots = [SimpleNamespace(index=0, start_min=0, duration_min=5)]
    windows = ([], [{"start": base.minutes_now, "end": base.minutes_now + 5}], [base.soc_max])
    candidate_checks = {
        "nominal": {"passed": True, "checked_minutes": 60, "horizon_energy_cost": 150, "battery_cycle_kwh": 0, "final_battery_kwh": 0},
        "pv10": {"passed": True, "checked_minutes": 60, "horizon_energy_cost": 100, "battery_cycle_kwh": 0, "final_battery_kwh": 0},
    }
    baseline_checks = {
        "nominal": {"passed": True, "checked_minutes": 60, "horizon_energy_cost": 100, "battery_cycle_kwh": 0, "final_battery_kwh": 0},
        "pv10": {"passed": True, "checked_minutes": 60, "horizon_energy_cost": 200, "battery_cycle_kwh": 0, "final_battery_kwh": 0},
    }
    demand_checks = {name: {**check, "horizon_energy_cost": 300} for name, check in candidate_checks.items()}
    baseline_result = SimpleNamespace(status="ordinary_native_household_plan", actions=())
    demand_result = SimpleNamespace(status="direct_demand_counterfactual", total_cost=None, actions=())
    fake_candidate = (result, slots, windows, candidate_checks, options)

    with (
        patch("car_charging_plan._observation_clock", return_value=base.now_utc),
        patch("car_charging_household._replay_native_plan", return_value=(baseline_result, ([], base.charge_window_best, base.charge_limit_best), baseline_checks), create=True),
        patch("car_charging_household._solve_and_replay", return_value=fake_candidate),
        patch("car_charging_household._replay_demand_counterfactual", return_value=(demand_result, ([], [], []), demand_checks)),
    ):
        report = evaluate_household_fallback(base, car_settings(base), "connection_not_confirmed")

    assert report["status"] == "household_fallback", report
    assert report["execution_reason"] == "no_componentwise_savings_over_native", report
    assert base.car_charging_pending is None


def test_household_fallback_requires_live_ha_transport_for_cached_controls(base):
    """An OFF/on cache snapshot cannot grant a domestic command without the existing HA transport."""
    from car_charging_household import HouseholdFallbackBlocked, household_options

    _household_template_fixture(base)
    base.ha_interface.is_alive = lambda: False
    try:
        with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
            household_options(base, car_settings(base))
    except HouseholdFallbackBlocked as exc:
        assert exc.reason == "household_control_transport_unavailable"
    else:
        raise AssertionError("Disconnected HA transport accepted cached relay/support values")


def test_household_fallback_rechecks_controls_from_live_ha_state_api(base):
    """A cached relay-OFF value cannot override a live HA response that says relay ON."""
    from car_charging_household import HouseholdFallbackBlocked, household_options

    _household_template_fixture(base)
    live_states = {
        "/api/states/binary_sensor.combined_test_relay": {
            "entity_id": "binary_sensor.combined_test_relay",
            "state": "on",
            "last_changed": base.now_utc.isoformat(),
            "last_updated": base.now_utc.isoformat(),
            "attributes": {},
        },
        "/api/states/switch.combined_test_grid_support": {
            "entity_id": "switch.combined_test_grid_support",
            "state": "on",
            "last_changed": base.now_utc.isoformat(),
            "last_updated": base.now_utc.isoformat(),
            "attributes": {},
        },
        "/api/states/number.combined_test_grid_support": {
            "entity_id": "number.combined_test_grid_support",
            "state": "6000",
            "last_changed": base.now_utc.isoformat(),
            "last_updated": base.now_utc.isoformat(),
            "attributes": {},
        },
    }
    base.ha_interface.api_call = lambda endpoint, **_kwargs: live_states.get(endpoint)

    try:
        with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
            household_options(base, car_settings(base))
    except HouseholdFallbackBlocked as exc:
        assert exc.reason == "waiting_for_ev_relay_off"
        assert exc.waiting is True
    else:
        raise AssertionError("Cached relay OFF overrode a fresh HA state response showing relay ON")


def test_household_fallback_blocks_when_live_control_request_raises(base):
    """A failed current-state request becomes a bounded wait instead of escaping execution."""
    from car_charging_household import HouseholdFallbackBlocked, household_options

    _household_template_fixture(base)

    def fail_live_read(_endpoint, **_kwargs):
        """Represent a transport exception from the HA current-state request."""
        raise OSError("test transport failure")

    base.ha_interface.api_call = fail_live_read
    try:
        with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
            household_options(base, car_settings(base))
    except HouseholdFallbackBlocked as exc:
        assert exc.reason == "relay_observation_unavailable"
        assert exc.waiting is True
    else:
        raise AssertionError("A failed current-state request was treated as valid cached evidence")


def test_joint_search_miss_attempts_opted_in_household_fallback(base):
    """A missing joint candidate triggers domestic fallback without replacing its diagnostics."""
    from tests.test_car_planning_combined import _template_inputs

    _template_inputs(base, minutes=60, pv=0)
    settings = car_settings(base)
    settings.update(
        mode="apply",
        control_model="deye_solarman",
        apply_verified=True,
        household_fallback=True,
        household_economic_fallback=True,
        grid_charge_support={"enabled_entity": "switch.combined_test_grid_support", "limit_entity": "number.combined_test_grid_support"},
        import_limit_w=6000,
    )
    _set_entity(base, "switch.combined_test_grid_support", "on")
    _set_entity(base, "number.combined_test_grid_support", "6000")
    _set_entity(base, "binary_sensor.combined_test_relay", "off", last_changed=base.now_utc - timedelta(minutes=10))
    fallback_report = {"status": "candidate", "domestic_only": True, "native_checks": {}}
    no_candidate = SimpleNamespace(actions=(), status="bounded_no_plan", total_cost=None)
    base.car_charging_pending = {"report": {"status": "candidate"}, "commands": [{"ev_on": True}]}

    with (
        patch("car_charging_plan._observation_clock", return_value=base.now_utc),
        patch("car_charging_policy.household_reason", return_value=None),
        patch("car_charging_plan.build_candidate", return_value=(no_candidate, [])),
        patch("car_charging_household.evaluate_household_fallback", return_value=fallback_report) as fallback,
    ):
        report = evaluate_car_charging(base, publish=False)

    assert fallback.call_count == 1
    assert fallback.call_args.args[2] == "joint_candidate_economic_rejected"
    assert report is fallback_report
    assert report["joint_candidate_rejection"]["status"] == "economic_rejected"
    assert report["joint_candidate_rejection"]["search_beam_attempts"]
    assert report["joint_candidate_rejection"]["joint_economic_search_attempts"]
    assert all(attempt["native_reasons"] == {"nominal": ["solver_no_plan"], "pv10": ["solver_no_plan"]} for attempt in report["joint_candidate_rejection"]["joint_economic_search_attempts"])
    assert base.car_charging_pending is None


def test_household_native_cost_refinement_runs_both_beams_and_includes_cycle_and_terminal_value(base):
    """Refinement ranks paired native costs, including each replay's cycle and terminal terms."""
    from car_charging_household import _solve_and_replay, household_options

    _household_template_fixture(base, minutes=60)
    base.metric_battery_cycle = 2
    base.metric_battery_value_scaling = 1
    base.battery_value_rate = lambda _minute: 5
    with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
        options = household_options(base, car_settings(base))
    options["beam"] = 128
    options["native_reference_costs"] = {"nominal": 100, "pv10": 100}
    calls = []

    def fake_build(_base, trial):
        """Return a feasible, one-slot Demand shape with deliberately misleading solver ranks."""
        calls.append(trial["beam"])
        action = SimpleNamespace(slot_index=0, battery_mode="demand", ev_action="off", discharge_limit_kw=None)
        result = SimpleNamespace(actions=(action,), total_cost=1 if trial["beam"] == 128 else 1000, beam=trial["beam"])
        return result, [SimpleNamespace(index=0)]

    def fake_replay(_base, _trial, result, _slots):
        """Make the wider search cheaper only after native cycle and terminal adjustments."""
        horizon_cost, cycle_kwh, final_soc = (90, 10, 10) if result.beam == 128 else (95, 0, 8)
        checks = {
            name: {
                "passed": True,
                "checked_minutes": 60,
                "horizon_energy_cost": horizon_cost,
                "battery_cycle_kwh": cycle_kwh,
                "final_battery_kwh": final_soc,
            }
            for name in ("nominal", "pv10")
        }
        return ([], [], []), checks

    with patch("car_charging_plan.build_candidate", side_effect=fake_build), patch("car_charging_plan._replay_candidate", side_effect=fake_replay):
        selected = _solve_and_replay(base, options)

    assert calls == [128, 512], calls
    assert selected is not None
    assert selected[0].beam == 512, "the wider plan should win on replay cost, not the solver objective"
    assert selected[4]["beam"] == 512


def test_household_native_cost_refinement_discards_scenario_regression_before_ranking(base):
    """A lower worst-case scalar cannot hide nominal regression against the native reference."""
    from car_charging_household import _solve_and_replay, household_options

    _household_template_fixture(base, minutes=60)
    base.metric_battery_cycle = 0
    base.metric_battery_value_scaling = 0
    with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
        options = household_options(base, car_settings(base))
    options["beam"] = 128
    options["native_reference_costs"] = {"nominal": 100, "pv10": 200}

    def fake_build(_base, trial):
        """Return two paired-safe schedules: the scalar winner regresses nominal cost."""
        action = SimpleNamespace(slot_index=0, battery_mode="charge", ev_action="off", discharge_limit_kw=None)
        return SimpleNamespace(actions=(action,), beam=trial["beam"]), [SimpleNamespace(index=0)]

    def fake_replay(_base, _trial, result, _slots):
        """Expose literal costs where beam 512 improves PV10 but regresses nominal."""
        costs = (99, 199) if result.beam == 128 else (101, 150)
        checks = {
            name: {
                "passed": True,
                "checked_minutes": 60,
                "horizon_energy_cost": cost,
                "battery_cycle_kwh": 0,
                "final_battery_kwh": 0,
            }
            for name, cost in zip(("nominal", "pv10"), costs)
        }
        return ([], [{"start": base.minutes_now, "end": base.minutes_now + 5}], [base.soc_max]), checks

    with patch("car_charging_plan.build_candidate", side_effect=fake_build), patch("car_charging_plan._replay_candidate", side_effect=fake_replay):
        selected = _solve_and_replay(base, options)

    assert selected is not None
    assert selected[0].beam == 128, "the nominal-regressing scalar winner must be excluded before ranking"


def test_household_direct_demand_counterfactual_can_fail_cap_without_blocking_candidate(base):
    """The non-executable direct-Demand replay is diagnostic, not a solver-feasibility gate."""
    import car_charging_plan

    _household_template_fixture(base)
    reset_rates(base, 10, 0)
    base.rate_import.update({base.minutes_now + minute: 40 for minute in range(120, 360)})
    original_build = car_charging_plan.build_candidate
    original_replay = car_charging_plan._replay
    demand_searches = []

    def build_without_feasible_demand(observed_base, options):
        """Retain ordinary candidates while making the Demand solver baseline infeasible."""
        result, slots = original_build(observed_base, options)
        if options.get("domestic_demand_baseline"):
            demand_searches.append(options["beam"])
            return SimpleNamespace(actions=(), status="bounded_no_plan", total_cost=None), slots
        return result, slots

    def replay_with_counterfactual_cap_failure(observed_base, options, result, slots, windows, scenario):
        """Keep real replay accounting, but model a Demand-only site-cap violation."""
        check = original_replay(observed_base, options, result, slots, windows, scenario)
        if result.actions and all(action.battery_mode == "demand" for action in result.actions):
            check["passed"] = False
            check["reasons"] = ["site_import_exceeded"]
            check["peak_import_w"] = options["import_limit_w"] + 100
        return check

    with (
        patch("car_charging_plan._observation_clock", return_value=base.now_utc),
        patch("car_charging_plan.build_candidate", side_effect=build_without_feasible_demand),
        patch("car_charging_plan._replay", side_effect=replay_with_counterfactual_cap_failure),
    ):
        report = evaluate_car_charging(base, publish=False)

    assert report["status"] == "candidate", report
    assert report["native_checks"]["nominal"]["passed"] and report["native_checks"]["pv10"]["passed"]
    assert demand_searches == [], "the direct Demand counterfactual must not require a feasible solver result"
    assert report["demand_comparison"]["baseline_executable"] is False
    assert report["demand_comparison"]["baseline_checks"]["nominal"]["reasons"] == ["site_import_exceeded"]


def test_household_direct_demand_replay_runs_when_solver_finds_no_candidate(base):
    """A bounded search miss still records the full non-executable all-Demand replay, including cap failure."""
    import car_charging_plan
    from car_charging_household import evaluate_household_fallback

    _household_template_fixture(base, minutes=60)
    reset_rates(base, 10, 0)
    original_build = car_charging_plan.build_candidate
    original_replay = car_charging_plan._replay
    replayed_modes = []

    def build_without_plan(observed_base, options):
        """Return complete forecast slots with a truthful bounded search miss."""
        _, slots = original_build(observed_base, options)
        return SimpleNamespace(actions=(), status="bounded_no_plan", total_cost=None), slots

    def record_demand_replay(observed_base, options, result, slots, windows, scenario):
        """Keep native replay behavior but expose that both direct-Demand scenarios ran."""
        if result.status == "direct_demand_counterfactual":
            replayed_modes.append((scenario, {action.battery_mode for action in result.actions}))
        check = original_replay(observed_base, options, result, slots, windows, scenario)
        if result.status == "direct_demand_counterfactual":
            check["passed"] = False
            check["reasons"] = ["site_import_exceeded"]
        return check

    with (
        patch("car_charging_plan._observation_clock", return_value=base.now_utc),
        patch("car_charging_plan.build_candidate", side_effect=build_without_plan),
        patch("car_charging_plan._replay", side_effect=record_demand_replay),
    ):
        report = evaluate_household_fallback(base, car_settings(base), "joint_candidate_no_plan_found")

    assert report["execution_reason"] == "household_plan_not_found", report
    assert replayed_modes == [(0, {"demand"}), (1, {"demand"})]
    comparison = report["demand_comparison"]
    assert comparison["baseline_kind"] == "direct_all_demand_counterfactual"
    assert comparison["baseline_executable"] is False
    assert comparison["baseline_checks"]["nominal"]["checked_minutes"] == 60
    assert comparison["baseline_checks"]["pv10"]["checked_minutes"] == 60
    assert comparison["baseline_checks"]["nominal"]["reasons"] == ["site_import_exceeded"]
    assert comparison["baseline_checks"]["pv10"]["reasons"] == ["site_import_exceeded"]


def test_household_failed_economic_gate_resumes_paired_safe_native_plan(base):
    """With no prior domestic authority, a rejected candidate leaves the native executor in control."""
    from inverter_service_control import dispatch_template

    _household_template_fixture(base)
    reset_rates(base, 40, 0)
    base.soc_kw = 20
    with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
        report = evaluate_car_charging(base, publish=False)
    assert report["status"] == "household_fallback"
    assert report["native_comparison"]["baseline_passed"] is True, report
    assert base.car_charging_pending is None

    inverter = base.inverters[0]
    with (
        patch("car_charging_plan._observation_clock", return_value=base.now_utc),
        patch("inverter_service_control.dispatch_template", wraps=dispatch_template) as shared_dispatch,
        patch.object(inverter, "adjust_charge_window", wraps=inverter.adjust_charge_window) as native_window,
        patch.object(base, "set_window_minutes", 240),
    ):
        base.execute_plan()

    assert shared_dispatch.call_count == 0
    assert native_window.called
    assert base.car_charging_execution is None
    assert base.car_charging_report["household_native_handoff"] == "paired_safe"


def test_household_rejected_candidate_reports_conflict_when_native_replay_fails(base):
    """A cap-failing native schedule is neither silently executed nor replaced with fresh Demand."""
    from car_charging_execute import execute_car_charging_plan
    from inverter_service_control import dispatch_template

    _household_template_fixture(base)
    reset_rates(base, 40, 0)
    base.soc_kw = 20
    with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
        report = evaluate_car_charging(base, publish=False)
    assert report["status"] == "household_fallback" and base.car_charging_pending is None, report
    comparison = report["native_comparison"]
    comparison["baseline_passed"] = False
    comparison["baseline_checks"]["nominal"]["passed"] = False
    comparison["baseline_checks"]["nominal"]["reasons"] = ["site_import_exceeded"]

    with patch("car_charging_plan._observation_clock", return_value=base.now_utc), patch("inverter_service_control.dispatch_template", wraps=dispatch_template) as shared_dispatch:
        outcome = execute_car_charging_plan(base)

    assert outcome[0] == "Household conflict", outcome
    assert shared_dispatch.call_count == 0
    assert base.car_charging_report["status"] == "safety_economics_conflict"
    assert base.car_charging_report["household_native_handoff"] == "blocked"


def test_household_prior_temporary_authority_is_cancelled_before_native_handoff(base):
    """Only a genuine earlier receipt gets checked target-zero cancellation before native execution."""
    from car_charging_execute import execute_car_charging_plan
    from inverter_service_control import dispatch_template

    _household_template_fixture(base)
    reset_rates(base, 40, 0)
    base.soc_kw = 20
    with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
        report = evaluate_car_charging(base, publish=False)
    assert report["native_comparison"]["baseline_passed"] is True, report
    assert report["status"] == "household_fallback" and base.car_charging_pending is None, report
    base.car_charging_execution = {"domestic_only": True}

    with patch("car_charging_plan._observation_clock", return_value=base.now_utc), patch("inverter_service_control.dispatch_template", wraps=dispatch_template) as shared_dispatch:
        outcome = execute_car_charging_plan(base)

    assert outcome is None
    assert [call.args[1] for call in shared_dispatch.call_args_list] == [0]
    assert base.car_charging_execution is None
    assert base.car_charging_report["household_native_handoff"] == "paired_safe_after_cancellation"


def test_household_fallback_validates_48h_and_displays_only_24h(base):
    """The accepted domestic route keeps its full paired replay while the view stops at 24 hours."""
    from datetime import datetime

    from car_charging_view import build_car_charging_plan_view

    _household_template_fixture(base, minutes=2880)
    base.forecast_plan_hours = 24
    reset_rates(base, 10, 0)
    base.rate_import.update({base.minutes_now + minute: 40 for minute in range(120, 360)})
    with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
        report = evaluate_car_charging(base, publish=False)

    assert report["status"] == "candidate", report
    assert set(report["native_checks"]) == {"nominal", "pv10"}
    assert all(check["passed"] and check["checked_minutes"] == 2880 for check in report["native_checks"].values())
    assert all(sum(row["minutes"] for row in rows) == 2880 for rows in report["preview"].values())
    view = build_car_charging_plan_view({}, report, now=base.now_utc)
    assert view["plan_simulation"] is True and view["rows"]
    visible_minutes = sum((datetime.fromisoformat(row["plan_interval_end"]) - datetime.fromisoformat(row["time"])).total_seconds() / 60 for row in view["rows"])
    cutoff = datetime.fromisoformat(report["preview"]["nominal"][0]["start"]) + timedelta(hours=24)
    assert visible_minutes == 1440
    assert datetime.fromisoformat(view["rows"][-1]["plan_interval_end"]) == cutoff
    assert any(datetime.fromisoformat(row["start"]) >= cutoff for row in report["preview"]["nominal"])


def test_household_demand_comparison_uses_an_all_demand_baseline(base):
    """The candidate comparison uses an independently replayed all-Demand diagnostic."""
    from car_charging_household import _replay_demand_counterfactual, _solve_and_replay, household_options

    _household_template_fixture(base)
    reset_rates(base, 10, 0)
    base.rate_import.update({base.minutes_now + minute: 40 for minute in range(120, 360)})
    with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
        options = household_options(base, car_settings(base))
    candidate = _solve_and_replay(base, options)

    assert candidate is not None
    candidate_result, slots, _, candidate_checks, _ = candidate
    demand_result, _, demand_checks = _replay_demand_counterfactual(base, options, candidate_result, slots)
    assert any(action.battery_mode == "charge" for action in candidate_result.actions)
    assert all(action.battery_mode == "demand" for action in demand_result.actions)
    assert demand_result.status == "direct_demand_counterfactual" and demand_result.total_cost is None
    assert all(check["passed"] and check["checked_minutes"] == 360 for check in (*candidate_checks.values(), *demand_checks.values()))
    assert all(action.ev_action == "off" for action in (*candidate_result.actions, *demand_result.actions))
    assert all("battery_cycle_kwh" in check for check in (*candidate_checks.values(), *demand_checks.values()))


def test_household_fallback_plans_cheaper_than_demand_without_ev_evidence(base):
    """The opt-in household schedule beats native operation, replays both cases, and reads no EV consent or SOC."""
    import car_charging_plan
    from car_charging_policy import household_reason

    _, writes = _household_template_fixture(base)
    reset_rates(base, 10, 0)
    base.rate_import.update({base.minutes_now + minute: 40 for minute in range(120, 360)})
    original = car_charging_plan._read_entity_state
    read_entities = []

    def guarded_read(observed_base, entity):
        """Keep real HA cache reads but reject an EV permission or SOC lookup."""
        read_entities.append(entity)
        assert entity not in ("sensor.combined_test_ev_soc", "switch.combined_test_permission")
        return original(observed_base, entity)

    with patch.object(car_charging_plan, "_read_entity_state", side_effect=guarded_read), patch.object(car_charging_plan, "_observation_clock", return_value=base.now_utc):
        report = evaluate_car_charging(base, publish=False)

    assert report["status"] == "candidate", report
    assert report["domestic_only"] is True and report["ev_soc_not_required"] is True
    assert "ev_initial_state" not in report and not report.get("proposed_ev_windows")
    assert all(check["passed"] and check["checked_minutes"] == 360 for check in report["native_checks"].values())
    assert {"nominal", "pv10"} == set(report["native_checks"])
    assert report["native_comparison"]["status"] == "componentwise_cheaper_than_native"
    assert report["native_comparison"]["componentwise_non_regression"] is True
    assert report["native_comparison"]["positive_improvement"] is True
    assert report["native_comparison"]["baseline_kind"] == "ordinary_native_household_plan"
    assert all(check["final_battery_kwh"] >= base.reserve - 0.002 for check in report["native_comparison"]["baseline_checks"].values())
    assert len(base.car_charging_pending["commands"]) == 72
    assert all(command["ev_on"] is False for command in base.car_charging_pending["commands"])
    assert any(command["target_soc"] > 0 for command in base.car_charging_pending["commands"])
    assert "sensor.combined_test_ev_soc" not in read_entities
    assert "switch.combined_test_permission" not in read_entities
    assert not writes

    pending = deepcopy(base.car_charging_pending)
    from inverter_service_control import dispatch_template

    _set_entity(base, "select.combined_test_program_1_charging", "Grid", attributes={"options": ["Disabled", "Grid", "Generator", "Both"]})
    with (
        patch("car_charging_plan._observation_clock", return_value=base.now_utc),
        patch("car_charging_policy.household_reason", wraps=household_reason) as fallback_reason,
        patch("inverter_service_control.dispatch_template", wraps=dispatch_template) as shared_dispatch,
    ):
        fallback_reason.reset_mock()
        base.publish_car_plan()
        assert fallback_reason.call_count == 0, "Domestic candidate publication must not inspect EV availability or consent"
        assert base.car_charging_pending is not None
        empty_slot = base.dashboard_values["binary_sensor." + base.prefix + "_car_charging_slot"]
        assert empty_slot["state"] == "off" and empty_slot["attributes"]["planned"] == []

        outcome = base.execute_plan()
        assert fallback_reason.call_count == 0, "Domestic dispatch must not inspect EV availability or consent"
        assert outcome[0] == ("Charging" if pending["commands"][0]["target_soc"] else "Demand"), (outcome, base.car_charging_report)
        assert shared_dispatch.call_count == 1
        assert writes
        assert base.car_charging_execution["domestic_only"] is True
        assert base.car_charging_report["hardware_confirmed"] is False
        base.publish_car_plan()
        assert fallback_reason.call_count == 0, "Domestic receipt publication must not inspect EV availability or consent"
    signal = base.dashboard_values["binary_sensor." + base.prefix + "_car_charging_slot"]
    assert signal["state"] == "off" and signal["attributes"]["planned"] == []
    assert base.car_charging_report["status"] == "applied"

    base.car_charging_execution = {"domestic_only": False, "observations_match": True}
    with patch("car_charging_plan._observation_clock", return_value=base.now_utc), patch("car_charging_policy.household_reason", wraps=household_reason) as fallback_reason:
        fallback_reason.reset_mock()
        base.publish_car_plan()
        assert fallback_reason.call_count == 0, "Domestic handback must revoke stale EV authority without checking consent"
    assert base.car_charging_execution is None and base.car_charging_pending is None
    assert base.car_charging_report["status"] == "household_fallback"


def test_household_execution_rechecks_relay_and_support_before_dispatch(base):
    """Support loss cancels safely; an ON relay waits before any new inverter call."""
    from car_charging_policy import household_reason
    from car_charging_execute import execute_car_charging_plan
    from inverter_service_control import dispatch_template

    for entity, state, expected_status in (
        ("switch.combined_test_grid_support", "off", "household_fallback"),
        ("binary_sensor.combined_test_relay", "on", "household_waiting"),
    ):
        _, writes = _household_template_fixture(base)
        reset_rates(base, 10, 0)
        base.rate_import.update({base.minutes_now + minute: 40 for minute in range(120, 360)})
        with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
            report = evaluate_car_charging(base, publish=False)
        assert report["status"] == "candidate", report
        _set_entity(base, entity, state)

        with (
            patch("car_charging_plan._observation_clock", return_value=base.now_utc),
            patch("car_charging_policy.household_reason", wraps=household_reason) as fallback_reason,
            patch("inverter_service_control.dispatch_template", wraps=dispatch_template) as shared_dispatch,
        ):
            fallback_reason.reset_mock()
            outcome = execute_car_charging_plan(base)
            if entity == "switch.combined_test_grid_support":
                assert shared_dispatch.call_count == 0
            else:
                assert shared_dispatch.call_count == 0
            assert fallback_reason.call_count == 0
            base.publish_car_plan()
            assert fallback_reason.call_count == 0

        if entity == "switch.combined_test_grid_support":
            assert not writes
            assert outcome[0] == "Household conflict", outcome
            assert base.car_charging_report["status"] == "safety_economics_conflict"
            assert base.car_charging_report["household_demand_handback"] is False
        else:
            assert not writes
        assert base.car_charging_pending is None and base.car_charging_execution is None
        assert base.car_charging_report["status"] == ("safety_economics_conflict" if entity == "switch.combined_test_grid_support" else expected_status)
        if entity == "switch.combined_test_grid_support":
            assert outcome[0] == "Household conflict", outcome
        else:
            assert outcome[0] == "EV waiting", outcome
        signal = base.dashboard_values["binary_sensor." + base.prefix + "_car_charging_slot"]
        assert signal["state"] == "off" and signal["attributes"]["planned"] == []


def test_household_execution_rechecks_relay_and_support_after_dispatch(base):
    """Evidence changing while shared writes settle revokes the receipt and returns safe handback."""
    from car_charging_policy import household_reason
    from car_charging_execute import execute_car_charging_plan
    from inverter_service_control import dispatch_template

    for changed_entity, changed_state, expected_status in (
        ("switch.combined_test_grid_support", "off", "household_fallback"),
        ("binary_sensor.combined_test_relay", "on", "household_waiting"),
    ):
        _, writes = _household_template_fixture(base)
        reset_rates(base, 10, 0)
        base.rate_import.update({base.minutes_now + minute: 40 for minute in range(120, 360)})
        with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
            report = evaluate_car_charging(base, publish=False)
        assert report["status"] == "candidate", report
        candidate_target = base.car_charging_pending["commands"][0]["target_soc"]
        _set_entity(base, "select.combined_test_program_1_charging", "Grid", attributes={"options": ["Disabled", "Grid", "Generator", "Both"]})

        def dispatch_then_drift(inverter, target_soc, **kwargs):
            """Perform the real template dispatch, then model a concurrent HA observation change."""
            observed = dispatch_template(inverter, target_soc, **kwargs)
            _set_entity(base, changed_entity, changed_state)
            return observed

        with (
            patch("car_charging_plan._observation_clock", return_value=base.now_utc),
            patch("car_charging_policy.household_reason", wraps=household_reason) as fallback_reason,
            patch("inverter_service_control.dispatch_template", side_effect=dispatch_then_drift) as shared_dispatch,
        ):
            fallback_reason.reset_mock()
            outcome = execute_car_charging_plan(base)
            if changed_entity == "switch.combined_test_grid_support":
                assert [call.args[1] for call in shared_dispatch.call_args_list] == [candidate_target, 0]
            else:
                assert shared_dispatch.call_count == 1
            assert fallback_reason.call_count == 0
            base.publish_car_plan()
            assert fallback_reason.call_count == 0

        assert writes
        assert base.car_charging_pending is None and base.car_charging_execution is None
        assert base.car_charging_report["status"] == ("safety_economics_conflict" if changed_entity == "switch.combined_test_grid_support" else expected_status)
        if changed_entity == "switch.combined_test_grid_support":
            assert outcome[0] == "Household conflict", outcome
            assert base.car_charging_report["household_demand_handback"] is True
            assert base.car_charging_report["status"] == "safety_economics_conflict"
        else:
            assert outcome[0] == "EV waiting", outcome
        signal = base.dashboard_values["binary_sensor." + base.prefix + "_car_charging_slot"]
        assert signal["state"] == "off" and signal["attributes"]["planned"] == []


def test_household_cancellation_failure_reports_unobserved_hardware_stop(base):
    """A failed zero-target handback blocks and cannot resume native charging."""
    from car_charging_execute import execute_car_charging_plan
    from inverter_service_control import dispatch_template

    _household_template_fixture(base)
    reset_rates(base, 10, 0)
    base.rate_import.update({base.minutes_now + minute: 40 for minute in range(120, 360)})
    with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
        report = evaluate_car_charging(base, publish=False)
    assert report["status"] == "candidate", report
    _set_entity(base, "select.combined_test_program_1_charging", "Grid", attributes={"options": ["Disabled", "Grid", "Generator", "Both"]})
    targets = []

    def fail_zero_target(inverter, target, **kwargs):
        """Allow the initial intent, then fail the explicit zero-target cancellation."""
        targets.append(target)
        if len(targets) > 1 and target == 0:
            return False
        observed = dispatch_template(inverter, target, **kwargs)
        _set_entity(base, "switch.combined_test_grid_support", "off")
        return observed

    with (
        patch("car_charging_plan._observation_clock", return_value=base.now_utc),
        patch("inverter_service_control.dispatch_template", side_effect=fail_zero_target),
    ):
        outcome = execute_car_charging_plan(base)

    assert len(targets) == 2 and targets[-1] == 0, f"expected one intent and one failed cancellation, got {targets}"
    assert outcome[0] == "EV blocked", outcome
    assert base.car_charging_report["execution_reason"] == "hardware_stop_not_observed"
    assert base.car_charging_execution is None and base.car_charging_pending is None
    assert base.isCharging is False and base.isExporting is False


def test_domestic_economic_fallback_remains_default_false(base):
    """The existing native household handback does not become an arbitrage plan by default."""
    _household_template_fixture(base, economic_fallback=False)

    with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
        report = evaluate_car_charging(base, publish=False)

    assert report["status"] == "household_fallback", report
    assert "demand_comparison" not in report
    assert base.car_charging_pending is None


def test_household_fallback_keeps_demand_when_arbitrage_does_not_save(base):
    """A household candidate must improve componentwise on the ordinary native baseline."""
    _household_template_fixture(base)
    reset_rates(base, 40, 0)
    base.soc_kw = 20
    with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
        report = evaluate_car_charging(base, publish=False)

    assert report["status"] == "household_fallback", report
    assert report["native_comparison"]["status"] == "not_componentwise_cheaper"
    assert report["native_comparison"]["componentwise_rejected_candidates"]
    assert base.car_charging_pending is None


def test_household_fallback_waits_for_relay_off_and_enabled_grid_support(base):
    """Support loss blocks new charge, while only live relay OFF permits a safe Demand handback."""
    from car_charging_execute import execute_car_charging_plan
    from inverter_service_control import dispatch_template

    for support_state, limit_state in (("off", "6000"), ("unknown", "6000"), ("on", "unknown")):
        _, writes = _household_template_fixture(base, minutes=60)
        _set_entity(base, "switch.combined_test_grid_support", support_state)
        _set_entity(base, "number.combined_test_grid_support", limit_state)
        with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
            report = evaluate_car_charging(base, publish=False)
        assert report["status"] == "household_fallback", (support_state, limit_state, report)
        assert base.car_charging_pending is None
        with patch("car_charging_plan._observation_clock", return_value=base.now_utc), patch("inverter_service_control.dispatch_template", wraps=dispatch_template) as handback:
            outcome = execute_car_charging_plan(base)
        assert outcome[0] == "Household conflict", (support_state, limit_state, outcome)
        assert handback.call_count == 0
        assert base.car_charging_report["status"] == "safety_economics_conflict"

    for relay_state in ("on", "unknown"):
        _, writes = _household_template_fixture(base, minutes=60)
        _set_entity(base, "binary_sensor.combined_test_relay", relay_state)
        with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
            report = evaluate_car_charging(base, publish=False)
        assert report["status"] == "household_waiting", (relay_state, report)
        assert base.car_charging_pending is None
        with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
            outcome = execute_car_charging_plan(base)
        assert outcome[0] == "EV waiting", (relay_state, outcome)
        assert not writes


def run_car_planning_grid_support_tests(base):
    """Run deterministic dispatch and authority-boundary regressions."""
    for test in (test_grid_support_recorded_charge_request, test_grid_support_headroom_and_exhaustion, test_grid_support_continuous_ev_before_solar):
        test()
    test_grid_support_options_fail_closed(base)
    test_grid_support_continuous_candidate_is_paired(base)
    test_joint_native_reference_reuses_identical_ev_pattern_and_native_battery_windows(base)
    test_joint_cost_improvement_uses_raw_paired_vectors_and_switch_floor()
    test_joint_cost_search_keeps_configured_and_continuous_candidates_when_512_fails(base)
    test_joint_cost_search_retains_equal_native_battery_plan_for_ev_delivery(base)
    test_cost_priority_shortlist_retains_baseline_and_samples_ties()
    test_cost_priority_shortlist_keeps_cheaper_starts_and_other_priorities()
    test_cost_priority_tie_search_samples_late_candidate()
    test_cost_priority_compares_safe_initial_demand_with_ordinary_candidate()
    test_apply_cost_refines_representative_starts_even_after_a_feasible_result()
    test_grid_support_replay_rejects_hidden_reserve_tail(base)
    test_grid_support_capped_plan_fails_closed_without_pv_and_accepts_solar(base)
    test_incident_shaped_early_100_at_19_1_is_rejected_for_native_88_at_8_1(base)
    test_initial_demand_is_a_scoped_search_constraint()
    test_self_consumption_preference_is_paired_and_explicit(base)
    test_household_demand_comparison_uses_an_all_demand_baseline(base)
    test_household_direct_demand_replay_runs_when_solver_finds_no_candidate(base)
    test_household_authorized_metered_route_does_not_require_ev_reads(base)
    test_household_fallback_refuses_to_stage_without_a_valid_native_plan(base)
    test_household_native_baseline_replays_the_pre_fallback_plan_snapshot(base)
    test_household_baseline_uses_native_schedule_after_current_cycle_recompute(base)
    test_household_economic_gate_rejects_nominal_regression_hidden_by_pv10_savings(base)
    test_household_fallback_requires_live_ha_transport_for_cached_controls(base)
    test_household_fallback_rechecks_controls_from_live_ha_state_api(base)
    test_household_fallback_blocks_when_live_control_request_raises(base)
    test_joint_search_miss_attempts_opted_in_household_fallback(base)
    test_household_native_cost_refinement_runs_both_beams_and_includes_cycle_and_terminal_value(base)
    test_household_native_cost_refinement_discards_scenario_regression_before_ranking(base)
    test_household_direct_demand_counterfactual_can_fail_cap_without_blocking_candidate(base)
    test_household_failed_economic_gate_resumes_paired_safe_native_plan(base)
    test_household_rejected_candidate_reports_conflict_when_native_replay_fails(base)
    test_household_prior_temporary_authority_is_cancelled_before_native_handoff(base)
    test_household_fallback_validates_48h_and_displays_only_24h(base)
    test_household_fallback_plans_cheaper_than_demand_without_ev_evidence(base)
    test_household_execution_rechecks_relay_and_support_before_dispatch(base)
    test_household_execution_rechecks_relay_and_support_after_dispatch(base)
    test_household_cancellation_failure_reports_unobserved_hardware_stop(base)
    test_domestic_economic_fallback_remains_default_false(base)
    test_household_fallback_keeps_demand_when_arbitrage_does_not_save(base)
    test_household_fallback_waits_for_relay_off_and_enabled_grid_support(base)
    return False
