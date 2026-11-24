"""Power-readiness policy regressions using real native shared-support prediction."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch


def _inputs(base, minutes=720):
    """Prepare a six-hour cheap band and a usable ten-percent safety reserve."""
    from tests.test_car_planning_grid_support import _generated_reference_inputs

    options, slots, candidate, checks = _generated_reference_inputs(base, minutes=minutes)
    base.args["battery_power_readiness"] = {"enabled": True, "cheap_price_ceiling": 12.0}
    base.soc_max = 32.154
    base.reserve = base.soc_max * 0.1
    base.soc_kw = base.reserve
    base.car_charging_limit[0] = 0
    options.update(grid_charge_support_w=6000, import_limit_w=6000, house_only=True, terminal_min=base.reserve)
    for minute in range(base.minutes_now - 360, base.minutes_now + minutes):
        base.rate_import[minute] = 8.1 if minute < base.minutes_now + 360 else 20.0
    from car_charging_plan import build_native_slots

    slots = build_native_slots(base, options)
    actions = tuple(SimpleNamespace(slot_index=slot.index, battery_mode="demand", ev_action="off", discharge_limit_kw=None) for slot in slots)
    result = SimpleNamespace(status="feasible", total_cost=0, actions=actions, battery_soc_trace=tuple(base.reserve for _ in range(len(slots) + 1)))
    return options, slots, result


def test_cheap_band_excludes_the_ten_cent_boundary(base):
    """A strict ten-cent ceiling excludes equality, dearer tails and dear prior history."""
    from power_readiness import cheap_readiness_periods, readiness_settings

    local = SimpleNamespace(args={"battery_power_readiness": {"enabled": True, "cheap_price_ceiling": 10.0}}, minutes_now=360, rate_import={})
    assert readiness_settings(SimpleNamespace(args={"battery_power_readiness": {"enabled": True}}))["cheap_price_ceiling"] == 10.0
    slots = [SimpleNamespace(start_min=minute, duration_min=5) for minute in range(0, 480, 5)]
    for price in (-1.0, 0.0, 8.1, 9.999999, 10.0, 11.2):
        local.rate_import = {minute: price if 360 <= minute < 720 else 11.2 for minute in range(0, 840)}
        periods = cheap_readiness_periods(local, slots)
        assert periods == ([{"first": 0, "stop": 72, "start": 360, "end": 720}] if price < 10 else []), (price, periods)
    short_slots = slots[:24]
    local.rate_import = {minute: 8.1 if minute >= 360 else 10.0 for minute in range(0, 840)}
    assert cheap_readiness_periods(local, short_slots) == [], "Exactly ten cents cannot extend a short cheap band through prior history"


def test_readiness_does_not_enable_economic_continuation_for_late_peak(base):
    """A failed cheap buffer must not silently authorise a full household economic continuation."""
    from car_charging_plan import _replay_candidate
    from power_readiness import prepare_candidate, stage_household_readiness

    options, slots, seed = _inputs(base)
    for minute in range(0, base.forecast_minutes + 30, 5):
        base.load_minutes_step[minute] = (2.0 if minute < 360 else 4.0) / 12
        base.load_minutes_step10[minute] = (2.0 if minute < 360 else 4.0) / 12
        base.pv_forecast_minute_step[minute] = 0
        base.pv_forecast_minute10_step[minute] = 0
    base.load_minutes_step10[650] = 8.0 / 12
    from car_charging_plan import build_native_slots

    slots = build_native_slots(base, options)
    fixed = prepare_candidate(base, options, seed, slots, ([], [], []))
    _, fixed_checks = _replay_candidate(base, options, fixed, slots)
    assert not fixed_checks["pv10"]["passed"], fixed_checks
    assert "site_import_exceeded" in fixed_checks["pv10"]["reasons"]
    base.car_charging_pending = None
    with patch("car_charging_plan.build_candidate") as economic_solver:
        report = stage_household_readiness(base, options, slots, "test_future_peak")
    economic_solver.assert_not_called()
    assert report["status"] != "candidate", report
    assert report["execution_reason"] == "power_readiness_native_replay_failed", report
    assert base.car_charging_pending is None
    base.load_minutes_step10[0] = 100.0 / 12
    base.car_charging_pending = None
    slots = build_native_slots(base, options)
    rejected = stage_household_readiness(base, options, slots, "test_impossible_peak")
    assert rejected["status"] != "candidate", rejected
    assert base.car_charging_pending is None


def test_cheap_only_peak_cover_uses_authorised_cap_without_economic_solver(base):
    """Raise only the cheap-band target to cover a later peak, never schedule non-cheap charging."""
    from car_charging_plan import build_native_slots
    from power_readiness import stage_household_readiness

    options, _slots, _seed = _inputs(base)
    base.args["battery_power_readiness"]["maximum_soc_percent"] = 70
    for minute in range(0, base.forecast_minutes + 30, 5):
        base.load_minutes_step[minute] = (2.0 if minute < 360 else 2.2) / 12
        base.load_minutes_step10[minute] = base.load_minutes_step[minute]
        base.pv_forecast_minute_step[minute] = 0
        base.pv_forecast_minute10_step[minute] = 0
    base.load_minutes_step[650] = base.load_minutes_step10[650] = 8.0 / 12
    before = deepcopy(base.args)
    slots = build_native_slots(base, options)
    with patch("car_charging_plan.build_candidate") as economic_solver:
        report = stage_household_readiness(base, options, slots, "test_later_peak")
    economic_solver.assert_not_called()
    assert report["status"] == "candidate", report
    assert 35 < report["power_readiness"]["target_soc_percent"] <= 70
    assert all(check["passed"] for check in report["native_checks"].values()), report
    for command in base.car_charging_pending["commands"]:
        assert command["ev_on"] is False
        if command["start"] >= base.minutes_now + 360:
            assert command["target_soc"] == 0, command
        else:
            assert 35 <= command["target_soc"] <= 70, command
    assert base.args == before


def test_cheap_peak_cover_rejects_unapproved_targets_and_ev_authority(base):
    """A service replay target cannot exceed policy bounds or grant any EV-on schedule."""
    from power_readiness import prepare_candidate

    options, slots, seed = _inputs(base)
    base.args["battery_power_readiness"]["maximum_soc_percent"] = 70
    for target in (71, 34, True, 35.0):
        try:
            prepare_candidate(base, options, seed, slots, ([], [], []), service_target_percent=target)
        except ValueError:
            continue
        raise AssertionError("Unapproved service replay target accepted")
    for fault in ("not_household", "ev_on"):
        changed_options = {**options, "house_only": fault != "not_household"}
        changed_seed = deepcopy(seed)
        if fault == "ev_on":
            changed_seed.actions[0].ev_action = "on"
        try:
            prepare_candidate(base, changed_options, changed_seed, slots, ([], [], []), service_target_percent=70)
        except ValueError:
            continue
        raise AssertionError("Non-domestic replay target accepted")


def test_readiness_defaults_off_and_rejects_invalid_settings(base):
    """Explicit opt-in is required and invalid settings cannot silently become an enabled policy."""
    import power_readiness

    options, slots, result = _inputs(base)
    base.args.pop("battery_power_readiness")
    assert power_readiness.prepare_candidate(base, options, result, slots) is result
    for invalid in (True, {"enabled": "true"}, {"enabled": True, "target_soc_percent": 100}, {"enabled": True, "minimum_cheap_minutes": 0}, {"enabled": True, "price_spread": float("nan")}):
        base.args["battery_power_readiness"] = invalid
        try:
            power_readiness.prepare_candidate(base, options, result, slots)
        except ValueError:
            continue
        raise AssertionError("Invalid readiness configuration accepted")


def test_flat_cheap_weekend_and_holiday_buffer_without_changing_reserve(base):
    """The same long cheap tariffs activate readiness on any weekday, including holidays."""
    import power_readiness
    from car_charging_plan import _replay_candidate, _replay

    options, slots, original = _inputs(base)
    before = deepcopy(base.args)
    for weekday in (0, 5, 6):
        base.midnight_utc = datetime(2026, 11, 9, tzinfo=timezone.utc) + timedelta(days=weekday)
        base.now_utc = base.midnight_utc + timedelta(minutes=base.minutes_now)
        prepared = power_readiness.prepare_candidate(base, options, original, slots)
        assert prepared is not original
        assert prepared.power_readiness["active"] is True
        assert prepared.power_readiness["target_soc_percent"] == 35
        assert prepared.actions[0].battery_mode == "charge"
        windows, checks = _replay_candidate(base, options, prepared, slots)
        assert all(check["passed"] and check["checked_minutes"] == base.forecast_minutes for check in checks.values()), checks
        assert all(limit >= base.soc_max * 0.35 for window, limit in zip(windows[1], windows[2]) if window["start"] == base.minutes_now)
        full = _replay(base, options, prepared, slots, windows, 0)
        assert max(full["_charge_peak_kwh"]) >= base.soc_max * 0.35 - 0.01
        assert base.reserve == base.soc_max * 0.1
        assert base.args == before
        assert all(action.battery_mode == "demand" for action in original.actions)


def test_expensive_flat_or_short_cheap_period_does_not_activate(base):
    """Flatness alone is not cheapness and a brief dip cannot authorise a standing buffer."""
    import power_readiness

    for cheap_minutes in (0, 60):
        options, slots, result = _inputs(base)
        for minute in base.rate_import:
            base.rate_import[minute] = 8.1 if base.minutes_now <= minute < base.minutes_now + cheap_minutes else 30.0
        assert power_readiness.prepare_candidate(base, options, result, slots) is result


def test_readiness_continues_near_end_of_long_cheap_period(base):
    """Earlier observed cheap hours prevent readiness switching off merely as the future interval shrinks."""
    import power_readiness

    options, slots, result = _inputs(base)
    for minute in base.rate_import:
        base.rate_import[minute] = 8.1 if base.minutes_now - 360 <= minute < base.minutes_now + 60 else 30.0
    prepared = power_readiness.prepare_candidate(base, options, result, slots)
    assert prepared.power_readiness["active"] is True
    assert prepared.actions[0].battery_mode == "charge"


def test_long_cheap_band_is_not_removed_by_a_brief_even_cheaper_future_dip(base):
    """A cheap weekend buffer cannot disappear just because one later short interval is cheaper still."""
    import power_readiness

    options, slots, result = _inputs(base)
    for minute in range(base.minutes_now + 400, base.minutes_now + 430):
        base.rate_import[minute] = 0.0
    prepared = power_readiness.prepare_candidate(base, options, result, slots)
    assert getattr(prepared, "power_readiness", {}).get("active") is True
    assert prepared.actions[0].battery_mode == "charge"


def test_real_solver_and_generated_reference_keep_the_same_readiness_service(base):
    """The optimizer and its independently generated native alternative cannot shed readiness to save money."""
    from car_charging_plan import build_candidate, _replay_candidate
    from car_charging_household import _generate_low_rates_native_reference

    options, _slots, _result = _inputs(base)
    result, slots = build_candidate(base, options)
    assert getattr(result, "power_readiness", {}).get("active") is True, "real solver output omitted the requested service"
    windows, checks = _replay_candidate(base, options, result, slots)
    assert all(check["passed"] for check in checks.values()), checks
    schedule = tuple((action.slot_index, action.ev_action) for action in result.actions)
    reference = _generate_low_rates_native_reference(base, options, schedule, checks, slots)
    assert reference["status"] == "paired_safe", reference
    assert reference["provenance"]["power_readiness"]["active"] is True
    for candidate_windows in (windows, reference["windows"]):
        first = next(limit for window, limit in zip(candidate_windows[1], candidate_windows[2]) if window["start"] <= base.minutes_now < window["end"])
        assert first >= base.soc_max * 0.35
    assert all(action.battery_mode == "charge" for action in reference["result"].actions[:72])


def test_buffer_is_spendable_for_ten_kw_burst_and_recovers(base):
    """A 35% target supplies a 10 kW burst above 6 kW and replenishes without raising reserve."""
    import power_readiness
    from car_charging_plan import _replay

    options, slots, result = _inputs(base)
    base.soc_kw = base.soc_max * 0.35
    for minute in range(60, 120, 5):
        base.load_minutes_step[minute] = base.load_minutes_step10[minute] = 10 * 5 / 60
    prepared = power_readiness.prepare_candidate(base, options, result, slots)
    check = _replay(base, options, prepared, slots, prepared.native_policy_windows, 0)
    assert check["passed"], check
    assert check["minimum_battery_kwh"] < base.soc_max * 0.35 - 1
    assert check["minimum_battery_kwh"] >= base.reserve - 0.002
    assert max(check["_charge_peak_kwh"]) >= base.soc_max * 0.35 - 0.01
    assert check["peak_import_w"] <= 6000 + 1e-6


def test_dynamic_buffer_and_impossible_power_never_hide_import_violation(base):
    """Long predicted deficits can increase readiness, but an impossible instantaneous deficit remains rejected."""
    import power_readiness
    from car_charging_plan import _replay

    options, slots, result = _inputs(base)
    base.soc_kw = base.soc_max * 0.6
    for minute in range(60, 240, 5):
        base.load_minutes_step[minute] = base.load_minutes_step10[minute] = 10 * 5 / 60
    prepared = power_readiness.prepare_candidate(base, options, result, slots)
    assert 35 < prepared.power_readiness["target_soc_percent"] <= 60
    for minute in range(60, 65, 5):
        base.load_minutes_step[minute] = base.load_minutes_step10[minute] = 20 * 5 / 60
    prepared = power_readiness.prepare_candidate(base, options, result, slots)
    check = _replay(base, options, prepared, slots, prepared.native_policy_windows, 0)
    assert check["passed"] is False and "site_import_exceeded" in check["reasons"]
    assert check["peak_import_w"] > 6000


def test_no_ev_readiness_stages_and_dispatches_without_economic_fallback_or_vehicle_reads(base):
    """Explicit readiness charges the house from reserve with the EV unavailable and ordinary economic fallback off."""
    from tests.test_car_planning_grid_support import _household_template_fixture
    from tests.test_infra import reset_rates
    from car_charging_plan import evaluate_car_charging
    from car_charging_execute import execute_car_charging_plan

    inverter, writes = _household_template_fixture(base, minutes=360, economic_fallback=False)
    reset_rates(base, 8.1, 0)
    base.args["battery_power_readiness"] = {"enabled": True}
    base.soc_kw = inverter.soc_kw = base.reserve
    inverter.soc_percent = 10
    reads = []
    original = base.ha_interface.api_call

    def observe_read(endpoint, **kwargs):
        """Record the live control reads; no vehicle energy or consent should be necessary."""
        reads.append(endpoint)
        return original(endpoint, **kwargs)

    with patch("car_charging_plan._observation_clock", return_value=base.now_utc), patch("car_charging_execute._clock", return_value=base.now_utc), patch.object(base.ha_interface, "api_call", side_effect=observe_read):
        report = evaluate_car_charging(base, publish=False)
        assert report["status"] == "candidate", report
        assert report["power_readiness"]["active"] is True
        assert report["power_readiness_service"] is True
        assert report["battery_plan_improvement"] is False
        assert report["proposed_ev_windows"] == []
        assert base.car_charging_pending["commands"][0]["target_soc"] == 35
        assert execute_car_charging_plan(base)[0] == "Charging", base.car_charging_report
    assert base.car_charging_report["status"] == "applied"
    assert base.args["car_charging_household_economic_fallback"] is False
    assert all("ev_soc" not in item and "permission" not in item for item in reads)
    assert writes


def test_mode_commitment_solar_ev_and_missing_forecast_are_not_overridden(base):
    """Readiness cannot override an existing Demand commitment, solar-only EV or incomplete physical inputs."""
    import power_readiness

    options, slots, result = _inputs(base)
    options.update(initial_window_mode="demand", initial_window_remaining_minutes=10, battery_min_window_minutes=5)
    prepared = power_readiness.prepare_candidate(base, options, result, slots)
    assert all(action.battery_mode == "demand" for action in prepared.actions[:2])
    assert prepared.actions[2].battery_mode == "charge"
    options.update(initial_window_remaining_minutes=0, ev_charging_priority="solar")
    result.actions[0].ev_action = "on"
    prepared = power_readiness.prepare_candidate(base, options, result, slots)
    assert prepared.actions[0].battery_mode == "demand" and prepared.actions[0].ev_action == "on"
    base.pv_forecast_minute10_step.pop(0)
    try:
        power_readiness.prepare_candidate(base, options, result, slots)
    except (ValueError, KeyError):
        return
    raise AssertionError("Missing PV10 physical evidence did not fail closed")


def test_readiness_charge_intent_is_retained_above_target_on_the_real_shared_template(base):
    """A ready battery above 35% retains the bounded Grid intent rather than a software overshoot stop."""
    from tests.test_car_planning_grid_support import _household_template_fixture
    from inverter_service_control import dispatch_template

    inverter, writes = _household_template_fixture(base)
    inverter.soc_percent = 60
    assert inverter.inv_has_target_soc is False
    assert dispatch_template(inverter, 35, charge_ceiling=True) is True
    assert writes
    for row in range(1, 7):
        assert base.get_state_wrapper(entity_id="select.combined_test_program_{}_charging".format(row)) == "Grid"
        assert float(base.get_state_wrapper(entity_id="number.combined_test_program_{}_soc".format(row))) == 35
    # The generic software-controlled overshoot stop remains unchanged elsewhere.
    inverter.inverter_type = "OtherServiceProfile"
    inverter.adjust_charge_immediate(35, freeze=False)
    assert all(base.get_state_wrapper(entity_id="select.combined_test_program_{}_charging".format(row)) == "Disabled" for row in range(1, 7))


def test_domestic_service_revokes_changed_tariff_or_disabled_policy_before_dispatch(base):
    """A stale price proof or removed service opt-in cannot retain domestic authority."""
    from tests.test_car_planning_grid_support import _household_template_fixture
    from tests.car_planning_fixtures import car_settings
    from tests.test_infra import reset_rates
    from car_charging_plan import evaluate_car_charging
    from car_charging_household import execution_evidence, HouseholdFallbackBlocked

    for fault in ("tariff", "disabled"):
        _inverter, writes = _household_template_fixture(base, minutes=360, economic_fallback=False)
        reset_rates(base, 8.1, 0)
        base.args["battery_power_readiness"] = {"enabled": True}
        with patch("car_charging_plan._observation_clock", return_value=base.now_utc):
            report = evaluate_car_charging(base, publish=False)
            assert report["status"] == "candidate", report
            pending = base.car_charging_pending
            if fault == "tariff":
                base.rate_import[base.minutes_now] += 0.25
            else:
                base.args["battery_power_readiness"]["enabled"] = False
            try:
                execution_evidence(base, car_settings(base), pending)
            except HouseholdFallbackBlocked as exc:
                assert exc.reason == ("power_readiness_inputs_changed" if fault == "tariff" else "household_fallback_not_authorised")
            else:
                raise AssertionError("Stale readiness service retained authority")
        assert not writes


def test_ev_waits_when_house_peak_uses_the_available_battery_power(base):
    """The real solver delays the flexible EV rather than shedding the house or hiding a power shortage."""
    from car_charging_plan import build_candidate, _replay_candidate

    options, _slots, _result = _inputs(base, minutes=360)
    base.soc_kw = base.soc_max * 0.35
    base.car_charging_limit[0] = 1.0
    options.update(house_only=False, pause=20, restart=25, forecast_basis="envelope")
    for minute in range(0, 360, 5):
        base.load_minutes_step[minute] = base.load_minutes_step10[minute] = (8 if minute < 30 else 1) * 5 / 60
    result, slots = build_candidate(base, options)
    assert result.actions, result
    assert all(action.ev_action == "off" for action in result.actions[:6])
    assert any(action.ev_action == "on" for action in result.actions[6:])
    _windows, checks = _replay_candidate(base, options, result, slots)
    assert all(check["passed"] for check in checks.values()), checks


def test_inactive_domestic_readiness_leaves_ordinary_native_control_unchanged(base):
    """Enabling the policy must not block ordinary household control when no cheap interval qualifies."""
    from tests.test_car_planning_grid_support import _household_template_fixture
    from tests.test_infra import reset_rates
    from car_charging_plan import evaluate_car_charging
    from car_charging_execute import execute_car_charging_plan

    _inverter, writes = _household_template_fixture(base, minutes=360, economic_fallback=False)
    reset_rates(base, 30.0, 0)
    base.args["battery_power_readiness"] = {"enabled": True}
    with patch("car_charging_plan._observation_clock", return_value=base.now_utc), patch("car_charging_execute._clock", return_value=base.now_utc):
        report = evaluate_car_charging(base, publish=False)
        assert report["status"] == "household_fallback", report
        assert report["power_readiness"]["active"] is False
        assert execute_car_charging_plan(base) is None, base.car_charging_report
    assert base.car_charging_pending is None and not writes


def test_readiness_setting_is_admitted_by_full_ev_configuration(base):
    """The real configuration entry point must accept readiness before any candidate overlay runs."""
    from tests.test_car_planning_grid_support import _support_fixture
    from tests.test_infra import reset_rates
    from car_charging_plan import _options, evaluate_car_charging

    _support_fixture(base)
    base.args["battery_power_readiness"] = {"enabled": True, "minimum_cheap_minutes": 30}
    base.export_limit = 0
    reset_rates(base, 8.1, 0)
    with patch("car_charging_plan._observation_clock", return_value=base.now_utc), patch("inverter_service_control.validate_template"):
        assert _options(base)["grid_charge_support_w"] == 6000
        report = evaluate_car_charging(base, publish=False)
    assert report["status"] not in ("invalid_input", "error"), report
    assert report["control_model"] == "deye_solarman"
    assert report["power_readiness"]["active"] is True


def test_household_readiness_survives_native_memory_cleanup(base):
    """Ordinary debug-off cleanup cannot erase evidence needed by the next real lease refresh."""
    from tests.test_car_planning_grid_support import _household_template_fixture
    from tests.test_infra import reset_rates
    from car_charging_plan import evaluate_car_charging
    from car_charging_execute import execute_car_charging_plan, refresh_car_charging_lease

    for invalidation in ("tariff", "expired", "logging_failure"):
        inverter, writes = _household_template_fixture(base, minutes=360, economic_fallback=False)
        reset_rates(base, 8.1, 0)
        base.args.update(battery_power_readiness={"enabled": True}, car_charging_recurring_session={"enabled": True})
        base.soc_kw = inverter.soc_kw = base.reserve
        inverter.soc_percent = 10
        base.debug_enable = False
        started = base.now_utc
        with patch("car_charging_plan._observation_clock", return_value=started), patch("car_charging_execute._clock", return_value=started):
            assert evaluate_car_charging(base, publish=False)["status"] == "candidate"
            assert execute_car_charging_plan(base)[0] == "Charging"
        before = len(writes)
        base.release_plan_forecasts()
        for seconds in (3, 15, 60, 240):
            observed = started + timedelta(seconds=seconds)
            with patch("car_charging_plan._observation_clock", return_value=observed), patch("car_charging_execute._clock", return_value=observed):
                refresh_car_charging_lease(base)
            assert base.car_charging_execution is not None, base.car_charging_report
            assert base.car_charging_report["status"] == "applied", base.car_charging_report
            assert len(writes) == before, "A cache-only refresh rewrote the inverter"
        if invalidation in ("tariff", "logging_failure"):
            base.rate_import[base.minutes_now] += 0.25
        observed = started + timedelta(seconds=300 if invalidation == "expired" else 241)
        with (
            patch("car_charging_plan._observation_clock", return_value=observed),
            patch("car_charging_execute._clock", return_value=observed),
            patch.object(base, "log", side_effect=OSError("synthetic logging failure") if invalidation == "logging_failure" else None) as log,
        ):
            refresh_car_charging_lease(base)
        reason = "receipt_expired" if invalidation == "expired" else "power_readiness_inputs_changed"
        assert any(call.args == ("Info: EV/household authority withdrawn: " + reason,) for call in log.call_args_list), log.call_args_list
        assert base.car_charging_execution is None
        assert base.car_charging_report["status"] != "applied"
        base.release_plan_forecasts()
        assert all(not getattr(base, name) for name in ("load_minutes_step", "load_minutes_step10", "pv_forecast_minute_step", "pv_forecast_minute10_step"))


def test_automatic_now_is_current_band_only_and_restores_requested_priority(base):
    """Automatic NOW uses real long cheap bands, never writes the selector or overrides solar."""
    from power_readiness import readiness_ev_priority

    _inputs(base)
    base.args["battery_power_readiness"] = {"enabled": True, "auto_ev_now": True}
    before = deepcopy(base.args)
    for minutes in (359, 360):
        for minute in base.rate_import:
            base.rate_import[minute] = 8.1 if base.minutes_now <= minute < base.minutes_now + minutes else 30.0
        assert readiness_ev_priority(base, "cost", "apply", base.now_utc) == ("now" if minutes == 360 else "cost")
    assert readiness_ev_priority(base, "solar", "apply", base.now_utc) == "solar"
    assert readiness_ev_priority(base, "cost", "shadow", base.now_utc) == "cost"
    assert readiness_ev_priority(base, "cost", "apply", base.now_utc + timedelta(minutes=360)) == "cost"
    for minute in base.rate_import:
        base.rate_import[minute] = 8.1 if base.minutes_now + 60 <= minute < base.minutes_now + 420 else 30.0
    assert readiness_ev_priority(base, "cost", "apply", base.now_utc) == "cost"
    for minute in base.rate_import:
        base.rate_import[minute] = 8.1 if base.minutes_now - 300 <= minute < base.minutes_now + 60 else 30.0
    assert readiness_ev_priority(base, "asap", "apply", base.now_utc) == "now"
    base.rate_import.pop(base.minutes_now)
    assert readiness_ev_priority(base, "cost", "apply", base.now_utc) == "cost"
    assert base.args == before


def _readiness_ev_fixture(base):
    """Keep real session/Storage checks and mirror only mocked live HA control-state reads."""
    from tests.test_car_planning_session import session_inputs

    result = session_inputs(base)
    base.ha_interface.is_alive = lambda: True

    def live_state(endpoint, **_kwargs):
        """Return the synthetic current-state REST contract without consulting any real service."""
        entity = endpoint.removeprefix("/api/states/")
        payload = base.ha_interface.get_state(entity_id=entity, raw=True)
        if not isinstance(payload, dict):
            return None
        return {**payload, "entity_id": entity, **{key: value.isoformat() if hasattr(value, "isoformat") else value for key, value in payload.items() if key in ("last_changed", "last_updated")}}

    base.ha_interface.api_call = live_state
    return result


def test_full_ev_automatic_now_is_revalidated_before_dispatch(base):
    """A new EV start cannot retain an automatic NOW proof after its current cheap band disappears."""
    from tests.car_planning_fixtures import car_settings
    from tests.test_infra import reset_rates
    from car_charging_plan import evaluate_car_charging
    from car_charging_execute import execute_car_charging_plan

    for changed in (False, True):
        _inverter, writes, _sources, _targets = _readiness_ev_fixture(base)
        settings = car_settings(base)
        settings.update(ev_charging_priority="cost", household_fallback=True, household_economic_fallback=False)
        base.args["battery_power_readiness"] = {"enabled": True, "auto_ev_now": True}
        reset_rates(base, 8.1, 0)
        for minute in range(base.minutes_now - 360, base.minutes_now):
            base.rate_import[minute] = 8.1
        with patch("car_charging_plan._observation_clock", return_value=base.now_utc), patch("car_charging_execute._clock", return_value=base.now_utc):
            report = evaluate_car_charging(base, publish=False)
            assert report["status"] == "candidate", report
            assert report["ev_charging_priority"] == "cost"
            assert report["effective_ev_charging_priority"] == "now"
            assert base.car_charging_pending["commands"][0]["ev_on"] is True
            if changed:
                base.rate_import[base.minutes_now] = 30.0
            executed = execute_car_charging_plan(base)
        if changed:
            assert not writes and base.car_charging_execution is None, base.car_charging_report
        else:
            assert executed[0] == "Charging", base.car_charging_report
            assert base.car_charging_report["status"] == "applied"
        assert car_settings(base)["ev_charging_priority"] == "cost"


def test_now_with_empty_home_battery_prepares_house_instead_of_deadlocking(base):
    """A fully authorised EV session below restart SOC can prepare domestic energy without starting the EV."""
    from tests.car_planning_fixtures import car_settings
    from tests.test_infra import reset_rates
    from car_charging_plan import evaluate_car_charging
    from car_charging_execute import execute_car_charging_plan

    for requested in ("now", "cost"):
        inverter, writes, _sources, _targets = _readiness_ev_fixture(base)
        settings = car_settings(base)
        settings.update(ev_charging_priority=requested, household_fallback=True, household_economic_fallback=False)
        base.args["battery_power_readiness"] = {"enabled": True, "auto_ev_now": True}
        base.all_active_keep = {}
        base.soc_kw = inverter.soc_kw = base.reserve
        inverter.soc_percent = 10
        reset_rates(base, 8.1, 0)
        for minute in range(base.minutes_now - 360, base.minutes_now):
            base.rate_import[minute] = 8.1
        with patch("car_charging_plan._observation_clock", return_value=base.now_utc), patch("car_charging_execute._clock", return_value=base.now_utc):
            report = evaluate_car_charging(base, publish=False)
            assert report["status"] == "candidate" and report["power_readiness_service"] is True, report
            assert report["domestic_only"] is True and report["proposed_ev_windows"] == []
            assert all(not command["ev_on"] for command in base.car_charging_pending["commands"])
            assert execute_car_charging_plan(base)[0] == "Charging", base.car_charging_report
        assert writes and not any(service.startswith("switch") for service, _data in writes)
        assert base.car_charging_execution["domestic_only"] is True
        assert settings["household_economic_fallback"] is False


def test_empty_battery_readiness_keeps_live_safety_checks(base):
    """Preparation after a failed NOW start cannot bypass authority, relay OFF or grid-support checks."""
    from tests.car_planning_fixtures import car_settings
    from tests.test_car_planning_actual import _set_entity
    from tests.test_infra import reset_rates
    from car_charging_plan import evaluate_car_charging
    from car_charging_execute import execute_car_charging_plan

    for fault in ("authority", "relay_on", "support_off", "read_only"):
        inverter, writes, _sources, _targets = _readiness_ev_fixture(base)
        settings = car_settings(base)
        settings.update(ev_charging_priority="now", household_economic_fallback=False)
        base.args["battery_power_readiness"] = {"enabled": True, "auto_ev_now": True}
        base.all_active_keep = {}
        base.soc_kw = inverter.soc_kw = base.reserve
        inverter.soc_percent = 10
        reset_rates(base, 8.1, 0)
        for minute in range(base.minutes_now - 360, base.minutes_now):
            base.rate_import[minute] = 8.1
        if fault == "authority":
            settings.update(apply_verified=False, execution_authorised=False)
        elif fault == "relay_on":
            _set_entity(base, settings["relay_entity"], "on")
        elif fault == "support_off":
            _set_entity(base, settings["grid_charge_support"]["enabled_entity"], "off")
        else:
            base.set_read_only = True
        with patch("car_charging_plan._observation_clock", return_value=base.now_utc), patch("car_charging_execute._clock", return_value=base.now_utc):
            evaluate_car_charging(base, publish=False)
            execute_car_charging_plan(base)
        assert not writes, (fault, writes)
        assert base.car_charging_execution is None, (fault, base.car_charging_report)


def run_power_readiness_tests(base):
    """Run service-policy and complete native numerical regressions."""
    test_cheap_peak_cover_rejects_unapproved_targets_and_ev_authority(base)
    test_cheap_only_peak_cover_uses_authorised_cap_without_economic_solver(base)
    test_readiness_does_not_enable_economic_continuation_for_late_peak(base)
    test_readiness_defaults_off_and_rejects_invalid_settings(base)
    test_flat_cheap_weekend_and_holiday_buffer_without_changing_reserve(base)
    test_expensive_flat_or_short_cheap_period_does_not_activate(base)
    test_readiness_continues_near_end_of_long_cheap_period(base)
    test_long_cheap_band_is_not_removed_by_a_brief_even_cheaper_future_dip(base)
    test_real_solver_and_generated_reference_keep_the_same_readiness_service(base)
    test_buffer_is_spendable_for_ten_kw_burst_and_recovers(base)
    test_dynamic_buffer_and_impossible_power_never_hide_import_violation(base)
    test_no_ev_readiness_stages_and_dispatches_without_economic_fallback_or_vehicle_reads(base)
    test_mode_commitment_solar_ev_and_missing_forecast_are_not_overridden(base)
    test_readiness_charge_intent_is_retained_above_target_on_the_real_shared_template(base)
    test_domestic_service_revokes_changed_tariff_or_disabled_policy_before_dispatch(base)
    test_ev_waits_when_house_peak_uses_the_available_battery_power(base)
    test_inactive_domestic_readiness_leaves_ordinary_native_control_unchanged(base)
    test_readiness_setting_is_admitted_by_full_ev_configuration(base)
    test_household_readiness_survives_native_memory_cleanup(base)
    test_automatic_now_is_current_band_only_and_restores_requested_priority(base)
    test_full_ev_automatic_now_is_revalidated_before_dispatch(base)
    test_now_with_empty_home_battery_prepares_house_instead_of_deadlocking(base)
    test_empty_battery_readiness_keeps_live_safety_checks(base)
    return False
