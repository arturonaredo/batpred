"""Domestic-only economic fallback using the existing shared Deye controls."""

from copy import deepcopy
from datetime import timedelta
from itertools import product
import math
import re
from types import SimpleNamespace

HOUSEHOLD_COST_TOLERANCE = 0.002
JOINT_COST_IMPROVEMENT_FLOOR = 1.0
JOINT_COST_EXTRA_SWITCH_FLOOR = 1.0
GENERATED_REFERENCE_MAX_WINDOWS = 3
GENERATED_REFERENCE_MAX_SEEDS = 4
GENERATED_REFERENCE_MAX_PAIRED_EVALUATIONS = 768
GENERATED_REFERENCE_REFINEMENT_PASSES = 2


class HouseholdFallbackBlocked(ValueError):
    """A stable fail-closed reason for domestic-only evidence or execution."""

    def __init__(self, reason, waiting=False, attempted=False):
        """Keep a bounded reason code plus relay-wait and dispatch-attempt state."""
        super().__init__(reason)
        self.reason = reason
        self.waiting = waiting
        self.attempted = attempted


def _future_observation(payload, now):
    """Return true when either HA timestamp is future-dated or malformed."""
    return any(not hasattr(payload.get(name), "tzinfo") or payload[name] > now for name in ("last_changed", "last_updated"))


def _read_live_control_state(base, entity_id):
    """Read one control state from HA's current-state REST endpoint, never its cache."""
    if not isinstance(entity_id, str) or not re.fullmatch(r"[a-z0-9_]+\.[a-z0-9_]+", entity_id):
        return None
    ha_interface = getattr(base, "ha_interface", None)
    api_call = getattr(ha_interface, "api_call", None)
    if not callable(api_call):
        return None
    try:
        payload = api_call("/api/states/" + entity_id)
        if not isinstance(payload, dict) or payload.get("entity_id") != entity_id or payload.get("state") is None:
            return None
        from car_charging_plan import _parse_entity_timestamp

        last_changed = _parse_entity_timestamp(payload.get("last_changed"), "control_last_changed")
        last_updated = _parse_entity_timestamp(payload.get("last_updated"), "control_last_updated")
    except Exception:
        return None
    return {
        "state": payload["state"],
        "last_changed": last_changed,
        "last_updated": last_updated,
        "value_last_changed": last_changed,
        "value_history": None,
        "attributes": deepcopy(payload.get("attributes", {})),
    }


def household_options(base, settings):
    """Build solver-only OFF-EV inputs from household, battery and support evidence."""
    from car_charging_plan import _aware, _number, _observation_clock

    from power_readiness import readiness_enabled

    if not isinstance(settings, dict) or settings.get("mode") != "apply" or settings.get("household_fallback") is not True or not (settings.get("household_economic_fallback") is True or readiness_enabled(base)):
        raise HouseholdFallbackBlocked("household_fallback_not_authorised")
    if settings.get("control_model") != "deye_solarman":
        raise HouseholdFallbackBlocked("shared_template_required")
    if settings.get("apply_verified") is not True and settings.get("execution_authorised") is not True:
        raise HouseholdFallbackBlocked("control_authorisation_required")

    now = _aware(_observation_clock(), "household_observation_clock")
    ha_interface = getattr(base, "ha_interface", None)
    is_alive = getattr(ha_interface, "is_alive", None)
    try:
        transport_alive = callable(is_alive) and is_alive() is True
    except Exception:
        transport_alive = False
    if not transport_alive:
        raise HouseholdFallbackBlocked("household_control_transport_unavailable", waiting=True)
    relay_entity = settings.get("relay_entity")
    if not isinstance(relay_entity, str) or not relay_entity:
        raise HouseholdFallbackBlocked("relay_entity_required", waiting=True)
    relay = _read_live_control_state(base, relay_entity)
    if relay is None or _future_observation(relay, now):
        raise HouseholdFallbackBlocked("relay_observation_unavailable", waiting=True)
    if relay["state"] != "off":
        raise HouseholdFallbackBlocked("waiting_for_ev_relay_off", waiting=True)

    support = settings.get("grid_charge_support")
    if not isinstance(support, dict) or set(support) != {"enabled_entity", "limit_entity"} or any(not isinstance(value, str) or not value for value in support.values()):
        raise HouseholdFallbackBlocked("grid_charge_support_unavailable")
    enabled = _read_live_control_state(base, support["enabled_entity"])
    limit = _read_live_control_state(base, support["limit_entity"])
    if enabled is None or enabled["state"] != "on" or limit is None or isinstance(limit["state"], bool):
        raise HouseholdFallbackBlocked("grid_charge_support_unavailable")
    if _future_observation(enabled, now) or _future_observation(limit, now):
        raise HouseholdFallbackBlocked("grid_charge_support_unavailable")
    import_limit = _number(settings.get("import_limit_w", getattr(base, "import_limit", 0)), "import_limit_w", 1)
    if abs(import_limit - base.import_limit) > 0.001:
        raise HouseholdFallbackBlocked("import_limit_mismatch")
    try:
        support_w = _number(float(limit["state"]), "grid_charge_support_w", 1, import_limit)
    except (TypeError, ValueError):
        raise HouseholdFallbackBlocked("grid_charge_support_unavailable") from None

    from battery_window import planning_window, selected_window_minutes

    window_minutes = selected_window_minutes(base)
    initial_mode, initial_remaining = planning_window(base, base.now_utc, now, window_minutes)
    pause = base.reserve / base.soc_max * 100
    terminal_percent = _number(settings.get("terminal_min_percent", pause), "terminal_min_percent", 0, 100)
    beam = _number(settings.get("beam_width", 128), "beam_width", 8, 512)
    if beam != int(beam):
        raise HouseholdFallbackBlocked("household_planner_settings_invalid")
    return {
        "mode": "apply",
        "control_model": "deye_solarman",
        "grid_charge_support_w": support_w,
        "import_limit_w": import_limit,
        "terminal_min": max(terminal_percent * base.soc_max / 100, base.reserve),
        "forecast_basis": "envelope",
        "beam": int(beam),
        "battery_min_window_minutes": window_minutes,
        "initial_window_mode": initial_mode,
        "initial_window_remaining_minutes": initial_remaining,
        "ev_charging_priority": "cost",
        # These zero values are solver sentinels only: availability is empty,
        # no physical EV state is read, and no EV window can be emitted.
        "ev_soc": 0.0,
        "house_only": True,
        "arrival": 0,
        "deadline": base.forecast_minutes,
        "pause": pause,
        "restart": pause,
        "off_minutes": 10,
        "initially_on": False,
        "initial_off_minutes": 10,
        "load_cap": None,
        "settings": deepcopy(settings),
        "entity_config": {"relay_entity": relay_entity},
        "entity_observations": {"relay_state": "off"},
        "entity_blockers": [],
        "planning_reference_at": base.now_utc.isoformat(),
        "observations_checked_at": now.isoformat(),
        "departure_passed": False,
        "domestic_only": True,
    }


def _paired_passed(checks, horizon):
    """Require exact full-horizon nominal and PV10 native replay evidence."""
    return _paired_complete(checks, horizon) and all(check.get("passed") is True for check in checks.values())


def _paired_complete(checks, horizon):
    """Require complete nominal and PV10 replay traces, regardless of cap outcome."""
    return isinstance(checks, dict) and set(checks) == {"nominal", "pv10"} and all(isinstance(check, dict) and check.get("checked_minutes") == horizon for check in checks.values())


def _snapshot_native_plan(base):
    """Copy the valid native plan clipped to the replay horizon before fallback search mutates state."""
    from car_charging_plan import _number
    from const import PREDICT_STEP

    if getattr(base, "plan_valid", None) is not True:
        return None
    windows = getattr(base, "charge_window_best", None)
    limits = getattr(base, "charge_limit_best", None)
    export_windows = getattr(base, "export_window_best", None)
    export_limits = getattr(base, "export_limits_best", None)
    if not isinstance(windows, list) or not isinstance(limits, list) or len(windows) != len(limits):
        return None
    if export_windows not in (None, []) or export_limits not in (None, []):
        return None
    now = getattr(base, "minutes_now", None)
    if type(now) is not int or type(base.forecast_minutes) is not int or base.forecast_minutes <= 0 or base.forecast_minutes % PREDICT_STEP:
        return None

    copied_windows = deepcopy(windows)
    copied_limits = deepcopy(limits)
    horizon_end = now + base.forecast_minutes
    previous_end = None
    replay_windows = []
    replay_limits = []
    for window, limit in zip(copied_windows, copied_limits):
        if not isinstance(window, dict) or type(window.get("start")) is not int or type(window.get("end")) is not int:
            return None
        start = window["start"]
        end = window["end"]
        if end <= start or (previous_end is not None and start < previous_end):
            return None
        if (start - now) % PREDICT_STEP or (end - now) % PREDICT_STEP:
            return None
        try:
            target = _number(limit, "native_charge_limit", 0, base.soc_max)
            if "target" in window and abs(_number(window["target"], "native_charge_target", 0, base.soc_max) - target) > 0.002:
                return None
        except (ValueError, TypeError, AttributeError):
            return None
        previous_end = end
        clipped_start = max(start, now)
        clipped_end = min(end, horizon_end)
        if clipped_start < clipped_end:
            window["start"] = clipped_start
            window["end"] = clipped_end
            replay_windows.append(window)
            replay_limits.append(target)
    return {
        "plan_valid": True,
        "plan_provenance": {
            "plan_made_by_version": deepcopy(getattr(base, "plan_made_by_version", None)),
            "plan_last_updated": deepcopy(getattr(base, "plan_last_updated", None)),
            "plan_last_updated_minutes": deepcopy(getattr(base, "plan_last_updated_minutes", None)),
        },
        "charge_windows": replay_windows,
        "charge_limits": replay_limits,
    }


def _componentwise_improvement(candidate, reference):
    """Require positive improvement in one scenario and no material regression in either."""
    scenarios = ("nominal", "pv10")
    if not isinstance(candidate, dict) or not isinstance(reference, dict) or set(candidate) != set(scenarios) or set(reference) != set(scenarios):
        return False
    try:
        return all(candidate[name] <= reference[name] + HOUSEHOLD_COST_TOLERANCE for name in scenarios) and any(candidate[name] < reference[name] - HOUSEHOLD_COST_TOLERANCE for name in scenarios)
    except TypeError:
        return False


def _joint_cost_improvement(candidate, reference, candidate_switches, reference_switches):
    """Require raw per-scenario non-regression and material savings net of extra mode changes."""
    from car_charging_plan import _number

    scenarios = ("nominal", "pv10")
    if not isinstance(candidate, dict) or not isinstance(reference, dict) or set(candidate) != set(scenarios) or set(reference) != set(scenarios):
        return False
    if type(candidate_switches) is not int or type(reference_switches) is not int or min(candidate_switches, reference_switches) < 0:
        return False
    try:
        values = {name: (_number(candidate[name], "joint_candidate_cost", -math.inf), _number(reference[name], "joint_reference_cost", -math.inf)) for name in scenarios}
    except (ValueError, TypeError):
        return False
    threshold = JOINT_COST_IMPROVEMENT_FLOOR + max(0, candidate_switches - reference_switches) * JOINT_COST_EXTRA_SWITCH_FLOOR
    return all(candidate_cost <= reference_cost + HOUSEHOLD_COST_TOLERANCE for candidate_cost, reference_cost in values.values()) and any(candidate_cost < reference_cost - threshold for candidate_cost, reference_cost in values.values())


def _native_adjusted_cost(base, checks):
    """Return separate nominal/PV10 costs with cycle wear and shared terminal-SOC value."""
    if not _paired_complete(checks, base.forecast_minutes):
        return None
    from car_charging_plan import _number

    try:
        terminal_rate = _number(base.battery_value_rate(base.minutes_now + base.forecast_minutes), "household_terminal_value", -math.inf)
        terminal_rate *= _number(base.metric_battery_value_scaling, "household_terminal_value_scaling")
        cycle_cost = _number(base.metric_battery_cycle, "household_cycle_cost")
        costs = {}
        for name in ("nominal", "pv10"):
            check = checks[name]
            horizon_cost = _number(check.get("horizon_energy_cost"), "household_horizon_cost", -math.inf)
            cycle_kwh = _number(check.get("battery_cycle_kwh"), "household_battery_cycle_kwh")
            final_soc = _number(check.get("final_battery_kwh"), "household_final_battery_kwh")
            costs[name] = horizon_cost + cycle_kwh * cycle_cost - final_soc * terminal_rate
        return costs
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def _plan_report(base, reason, status="household_fallback", **extra):
    """Return a fresh non-executable household report with no fabricated EV state."""
    from car_charging_plan import _empty_report

    report = _empty_report(status)
    report.update(
        {
            "requested_mode": "apply",
            "execution_reason": reason,
            "ev_available": False,
            "ev_soc_not_required": True,
            "domestic_only": True,
            "household_executor": "waiting_for_ev_off" if status == "household_waiting" else "native",
            "forecast_plan_hours": getattr(base, "forecast_plan_hours", 48),
            **extra,
        }
    )
    return report


def _solve_and_replay(base, options, diagnostics=None):
    """Refine both beam widths and rank only paired-safe, componentwise-cheaper plans."""
    from car_charging_plan import _replay_candidate, _validate_candidate_modes, build_candidate

    beam_widths = [options["beam"]]
    if options["beam"] < 512:
        beam_widths.append(512)
    candidates = []
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    diagnostics.update(paired_candidates=0, componentwise_rejections=[])
    for beam in beam_widths:
        trial = {**options, "beam": beam}
        result, slots = build_candidate(base, trial)
        if not result.actions:
            continue
        _validate_candidate_modes(trial, result)
        windows, checks = _replay_candidate(base, trial, result, slots)
        if not _paired_passed(checks, base.forecast_minutes):
            continue
        adjusted_cost = _native_adjusted_cost(base, checks)
        if adjusted_cost is not None:
            diagnostics["paired_candidates"] += 1
            reference_costs = options.get("native_reference_costs")
            if reference_costs is not None and not _componentwise_improvement(adjusted_cost, reference_costs):
                diagnostics["componentwise_rejections"].append(adjusted_cost)
                continue
            candidates.append(((max(adjusted_cost.values()), sum(adjusted_cost.values())), result, slots, windows, checks, trial))
    if not candidates:
        return None
    _, result, slots, windows, checks, selected_options = min(candidates, key=lambda item: item[0])
    return result, slots, windows, checks, selected_options


def _replay_demand_counterfactual(base, options, result, slots):
    """Replay a separately labeled all-Demand counterfactual; it is never executable."""
    from car_charging_plan import _replay

    actions = tuple(SimpleNamespace(slot_index=slot.index, battery_mode="demand", ev_action="off", discharge_limit_kw=None) for slot in slots)
    demand_result = SimpleNamespace(status="direct_demand_counterfactual", total_cost=None, actions=actions)
    windows = ([], [], [])
    checks = {name: _replay(base, options, demand_result, slots, windows, scenario) for name, scenario in (("nominal", 0), ("pv10", 1))}
    for check in checks.values():
        check.pop("_charge_peak_kwh", None)
    return demand_result, windows, checks


def _replay_native_plan(base, options, snapshot, slots=None):
    """Replay the snapshotted ordinary charge schedule against the same full horizon."""
    from car_charging_plan import _replay, build_native_slots

    slots = build_native_slots(base, options) if slots is None else slots
    actions = []
    covered = [False] * len(snapshot["charge_windows"])
    for slot in slots:
        start = base.minutes_now + slot.start_min
        end = start + slot.duration_min
        active = []
        for index, window in enumerate(snapshot["charge_windows"]):
            overlaps = window["start"] < end and window["end"] > start
            if overlaps and not (window["start"] <= start and end <= window["end"]):
                raise ValueError("Native plan window is not aligned to the forecast replay")
            if overlaps:
                active.append(index)
                covered[index] = True
        if len(active) > 1:
            raise ValueError("Native plan has overlapping charge windows")
        actions.append(
            SimpleNamespace(
                slot_index=slot.index,
                battery_mode="charge" if active else "demand",
                ev_action="off",
                discharge_limit_kw=None,
            )
        )
    if not all(covered):
        raise ValueError("Native plan charge windows are outside the replay horizon")
    result = SimpleNamespace(status="ordinary_native_household_plan", total_cost=None, actions=tuple(actions))
    windows = ([], deepcopy(snapshot["charge_windows"]), deepcopy(snapshot["charge_limits"]))
    from power_readiness import prepare_candidate

    result = prepare_candidate(base, options, result, slots, windows)
    windows = getattr(result, "native_policy_windows", windows)
    checks = {name: _replay(base, options, result, slots, windows, scenario) for name, scenario in (("nominal", 0), ("pv10", 1))}
    for check in checks.values():
        check.pop("_charge_peak_kwh", None)
    return result, windows, checks


def _replay_native_joint_plan(base, options, snapshot, candidate, slots, ev_windows):
    """Pair the ordinary native battery schedule with this exact candidate EV reservation."""
    from car_charging_plan import _native_windows, _replay

    if not isinstance(snapshot, dict) or snapshot.get("plan_valid") is not True:
        raise ValueError("Current-cycle native battery reference is unavailable")
    actions = getattr(candidate, "actions", None)
    if not isinstance(actions, (tuple, list)) or not actions or len(actions) != len(slots):
        raise ValueError("Joint native reference requires a complete candidate action horizon")
    expected_ev_windows = _native_windows(base, candidate, slots)[0]
    if expected_ev_windows != ev_windows:
        raise ValueError("Native reference EV reservation differs from the selected candidate")

    native_actions = []
    covered = [False] * len(snapshot["charge_windows"])
    for slot, ev_action in zip(slots, actions):
        if ev_action.slot_index != slot.index or ev_action.ev_action not in ("on", "off"):
            raise ValueError("Native reference EV action timeline is invalid")
        start = base.minutes_now + slot.start_min
        end = start + slot.duration_min
        active = []
        for index, window in enumerate(snapshot["charge_windows"]):
            overlaps = window["start"] < end and window["end"] > start
            if overlaps and not (window["start"] <= start and end <= window["end"]):
                raise ValueError("Native reference window is not aligned to the replay")
            if overlaps:
                active.append(index)
                covered[index] = True
        if len(active) > 1:
            raise ValueError("Native reference has overlapping charge windows")
        native_actions.append(
            SimpleNamespace(
                slot_index=slot.index,
                battery_mode="charge" if active else "demand",
                ev_action=ev_action.ev_action,
                discharge_limit_kw=None,
            )
        )
    if not all(covered):
        raise ValueError("Native reference charge window is outside the replay horizon")
    result = SimpleNamespace(status="ordinary_native_joint_reference", total_cost=None, actions=tuple(native_actions))
    windows = (deepcopy(ev_windows), deepcopy(snapshot["charge_windows"]), deepcopy(snapshot["charge_limits"]))
    from power_readiness import prepare_candidate

    result = prepare_candidate(base, options, result, slots, windows)
    windows = getattr(result, "native_policy_windows", windows)
    checks = {name: _replay(base, options, result, slots, windows, scenario) for name, scenario in (("nominal", 0), ("pv10", 1))}
    for check in checks.values():
        check.pop("_charge_peak_kwh", None)
    return result, windows, checks


def _fixed_ev_action_schedule(result, slots):
    """Copy only EV slot decisions from a candidate, never its battery actions."""
    actions = getattr(result, "actions", None)
    if not isinstance(actions, (tuple, list)) or not actions or len(actions) != len(slots):
        raise ValueError("Fixed EV reference requires a complete action horizon")
    schedule = []
    for action, slot in zip(actions, slots):
        slot_index = getattr(action, "slot_index", None)
        ev_action = getattr(action, "ev_action", None)
        if slot_index != slot.index or ev_action not in ("on", "off"):
            raise ValueError("Fixed EV reference schedule is invalid")
        schedule.append((slot.index, ev_action))
    return tuple(schedule)


def _reference_failure(status, reason, **extra):
    """Return a uniform non-executable result for a failed bounded search."""
    return {"status": status, "reason": reason, "result": None, "slots": None, "windows": None, "checks": {}, "costs": None, "provenance": {}, **extra}


def _generate_low_rates_native_reference(base, options, fixed_ev_schedule, candidate_checks, slots=None):
    """Find a bounded paired-safe reference from copied low-rate geometry and fixed EV decisions only."""
    from car_charging_plan import _number, _replay, build_native_slots
    from battery_window import commands_respect_windows
    from const import PREDICT_STEP

    source = getattr(base, "low_rates", None)
    if not isinstance(options, dict) or options.get("mode") != "apply" or options.get("ev_charging_priority") != "cost":
        return _reference_failure("unavailable", "generated_reference_inputs_invalid")
    if not isinstance(source, (list, tuple)) or not source:
        return _reference_failure("unavailable", "low_rates_unavailable")
    copied_source = deepcopy(source)
    raw_horizon = getattr(base, "forecast_minutes", None)
    if type(raw_horizon) is not int or raw_horizon <= 0 or raw_horizon % PREDICT_STEP:
        return _reference_failure("unavailable", "generated_reference_horizon_invalid")
    if len(copied_source) > raw_horizon // PREDICT_STEP:
        return _reference_failure("unavailable", "low_rates_geometry_too_large")

    try:
        now = base.minutes_now
        horizon = base.forecast_minutes
        capacity = _number(base.soc_max, "generated_reference_capacity", 0.001)
        reserve = _number(base.reserve, "generated_reference_reserve", 0, capacity)
        if type(now) is not int or type(horizon) is not int or horizon <= 0 or horizon % PREDICT_STEP:
            raise ValueError("Invalid native reference horizon")
        horizon_end = now + horizon
        geometry = []
        previous_end = None
        for source_index, window in enumerate(copied_source):
            if not isinstance(window, dict) or type(window.get("start")) is not int or type(window.get("end")) is not int:
                raise ValueError("Invalid low-rate window geometry")
            start, end = window["start"], window["end"]
            if end <= start or (start - now) % PREDICT_STEP or (end - now) % PREDICT_STEP:
                raise ValueError("Misaligned low-rate window geometry")
            average = _number(window.get("average"), "low_rates_window_average", -math.inf)
            clipped_start, clipped_end = max(start, now), min(end, horizon_end)
            if clipped_start >= clipped_end:
                continue
            if previous_end is not None and clipped_start < previous_end:
                raise ValueError("Overlapping low-rate window geometry")
            previous_end = clipped_end
            geometry.append({"source_index": source_index, "start": clipped_start, "end": clipped_end, "average": average})
    except (ValueError, TypeError, AttributeError):
        return _reference_failure("unavailable", "low_rates_geometry_invalid")
    if not geometry:
        return _reference_failure("unavailable", "low_rates_geometry_outside_horizon")

    try:
        native_slots = list(build_native_slots(base, options) if slots is None else slots)
        expected_steps = horizon // PREDICT_STEP
        if len(native_slots) != expected_steps:
            raise ValueError("Native reference slot count is incomplete")
        for index, slot in enumerate(native_slots):
            if type(slot.index) is not int or slot.index != index or slot.start_min != index * PREDICT_STEP or slot.duration_min != PREDICT_STEP:
                raise ValueError("Native reference slot timeline is invalid")
            for name in ("ac_load_kwh", "pv_kwh", "import_price", "export_price", "battery_min_kwh"):
                _number(getattr(slot, name), "generated_reference_" + name, 0 if name in ("ac_load_kwh", "pv_kwh", "battery_min_kwh") else -math.inf)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError):
        return _reference_failure("unavailable", "native_reference_slots_invalid")

    manual_names = ("manual_charge_times", "manual_demand_times", "manual_freeze_charge_times", "manual_export_times", "manual_freeze_export_times")
    manual_active = any(getattr(base, name, None) for name in manual_names)
    # A per-slot mode list cannot encode the target of a manual charge, freeze,
    # or export. Do not substitute a cheaper target for that explicit intent.
    if any(getattr(base, name, None) for name in manual_names if name != "manual_demand_times"):
        return _reference_failure("blocked", "manual_target_override_unsupported")
    has_allowed_modes = [hasattr(slot, "allowed_battery_modes") for slot in native_slots]
    if any(has_allowed_modes) and not all(has_allowed_modes):
        return _reference_failure("blocked", "native_allowed_modes_incomplete")
    if manual_active and not all(has_allowed_modes):
        return _reference_failure("blocked", "manual_override_active")
    allowed_modes = []
    if all(has_allowed_modes):
        for slot in native_slots:
            modes = getattr(slot, "allowed_battery_modes")
            if not isinstance(modes, (tuple, list)) or not modes or any(mode not in ("demand", "charge") for mode in modes):
                return _reference_failure("blocked", "native_allowed_modes_invalid")
            allowed_modes.append(tuple(modes))
    else:
        allowed_modes = [("demand", "charge")] * len(native_slots)

    if not isinstance(fixed_ev_schedule, (tuple, list)) or len(fixed_ev_schedule) != len(native_slots):
        return _reference_failure("unavailable", "fixed_ev_schedule_invalid")
    schedule = []
    for index, item in enumerate(fixed_ev_schedule):
        if not isinstance(item, (tuple, list)) or len(item) != 2 or type(item[0]) is not int or item[0] != native_slots[index].index or item[1] not in ("on", "off"):
            return _reference_failure("unavailable", "fixed_ev_schedule_invalid")
        schedule.append((item[0], item[1]))
    schedule = tuple(schedule)
    if not _paired_complete(candidate_checks, horizon):
        return _reference_failure("unavailable", "candidate_ev_evidence_incomplete")
    try:
        for name in ("nominal", "pv10"):
            _number(candidate_checks[name].get("final_ev_kwh"), "candidate_final_ev_kwh", 0)
            _number(candidate_checks[name].get("validated_ev_target_kwh"), "candidate_ev_target_kwh", 0)
    except (ValueError, TypeError, AttributeError):
        return _reference_failure("unavailable", "candidate_ev_evidence_invalid")

    try:
        # Match _intent: intermediate Deye targets require shared support and
        # must be strictly above the rounded reserve. Keep executor guards intact.
        shared_support = _number(options.get("grid_charge_support_w", 0), "generated_reference_grid_support", 0) > 0
        reserve_floor = max(1, math.ceil(reserve * 100 / capacity) + 1)
        positive_levels = tuple(range(reserve_floor, 101)) if shared_support else (100,)
        positive_levels = tuple(sorted({100, *positive_levels}))
        minimum_window = _number(options.get("battery_min_window_minutes", PREDICT_STEP), "generated_reference_minimum_window", PREDICT_STEP)
        if minimum_window % PREDICT_STEP:
            raise ValueError("Invalid battery window minimum")
        rate = _number(base.car_charging_rate[0], "generated_reference_ev_rate", 0.001)
    except (ValueError, TypeError, KeyError, AttributeError):
        return _reference_failure("unavailable", "generated_reference_inputs_invalid")

    if len(geometry) <= GENERATED_REFERENCE_MAX_WINDOWS:
        selected_geometry = geometry
    else:
        sample_indexes = sorted({0, round((len(geometry) - 1) / 2), len(geometry) - 1})
        selected_geometry = [geometry[index] for index in sample_indexes]
    coarse_levels = sorted({0, min(positive_levels), *[level for level in (25, 50, 75, 100) if level in positive_levels]})

    ev_windows = []
    for slot, (_slot_index, ev_action) in zip(native_slots, schedule):
        if ev_action == "on":
            start = now + slot.start_min
            end = start + slot.duration_min
            energy = rate * slot.duration_min / 60
            ev_windows.append({"start": start, "end": end, "kwh": energy, "average": slot.import_price, "cost": energy * slot.import_price, "octopus": False})

    evaluated = {}
    safe_references = []
    failure_reasons = set()
    paired_evaluations = 0
    refinement_passes = 0
    budget_exhausted = False

    def evaluate(levels):
        """Replay one unique native action proposal in both full-horizon scenarios."""
        nonlocal paired_evaluations
        levels = tuple(levels)
        if levels in evaluated:
            return evaluated[levels]
        if paired_evaluations + 2 > GENERATED_REFERENCE_MAX_PAIRED_EVALUATIONS:
            raise RuntimeError("generated_reference_search_budget_exhausted")
        targets_by_source = {window["source_index"]: level for window, level in zip(selected_geometry, levels)}
        charge_windows = []
        charge_limits = []
        for window, level in zip(selected_geometry, levels):
            if level == 0:
                continue
            target = capacity * level / 100
            if target + 1e-9 < reserve:
                failure_reasons.add("reserve_level_below_configured_floor")
                evaluated[levels] = None
                return None
            target = max(target, reserve)
            if not any(window["start"] <= now + slot.start_min and now + slot.start_min + slot.duration_min <= window["end"] for slot in native_slots):
                failure_reasons.add("low_rates_geometry_without_native_slots")
                evaluated[levels] = None
                return None
            charge_windows.append({"start": window["start"], "end": window["end"], "average": window["average"]})
            charge_limits.append(target)

        actions = []
        for slot, (_slot_index, ev_action) in zip(native_slots, schedule):
            start = now + slot.start_min
            end = start + slot.duration_min
            active = []
            for window in selected_geometry:
                overlaps = window["start"] < end and window["end"] > start
                if overlaps and not (window["start"] <= start and end <= window["end"]):
                    failure_reasons.add("low_rates_geometry_not_slot_aligned")
                    evaluated[levels] = None
                    return None
                if overlaps and targets_by_source[window["source_index"]] > 0:
                    active.append(window)
            if len(active) > 1:
                failure_reasons.add("low_rates_geometry_overlaps_native_slots")
                evaluated[levels] = None
                return None
            mode = "charge" if active else "demand"
            if mode not in allowed_modes[slot.index]:
                failure_reasons.add("generated_action_violates_native_allowed_modes")
                evaluated[levels] = None
                return None
            actions.append(SimpleNamespace(slot_index=slot.index, battery_mode=mode, ev_action=ev_action, discharge_limit_kw=None))
        commands = []
        for slot, action in zip(native_slots, actions):
            start = now + slot.start_min
            end = start + slot.duration_min
            target = next((level for window, level in zip(selected_geometry, levels) if window["start"] <= start < window["end"] and level > 0), 0)
            commands.append({"start": start, "end": end, "target_soc": target if action.battery_mode == "charge" else 0, "freeze": False, "power_w": None, "ev_on": action.ev_action == "on"})
        try:
            respects_windows = commands_respect_windows(commands, minimum_window, options.get("initial_window_mode"), options.get("initial_window_remaining_minutes", 0))
        except (ValueError, TypeError, KeyError, AttributeError):
            failure_reasons.add("generated_action_window_validation_failed")
            evaluated[levels] = None
            return None
        if not respects_windows:
            failure_reasons.add("generated_action_violates_battery_window_minimum")
            evaluated[levels] = None
            return None
        result = SimpleNamespace(status="generated_low_rates_native_reference", total_cost=None, actions=tuple(actions))
        windows = (deepcopy(ev_windows), charge_windows, charge_limits)
        from power_readiness import prepare_candidate

        try:
            result = prepare_candidate(base, options, result, native_slots, windows)
            windows = getattr(result, "native_policy_windows", windows)
        except ValueError:
            failure_reasons.add("power_readiness_controls_invalid")
            evaluated[levels] = None
            return None
        checks = {}
        try:
            for name, scenario in (("nominal", 0), ("pv10", 1)):
                checks[name] = _replay(base, options, result, native_slots, windows, scenario)
                checks[name].pop("_charge_peak_kwh", None)
                paired_evaluations += 1
        except Exception:
            failure_reasons.add("generated_reference_replay_failed")
            evaluated[levels] = None
            return None
        if not _paired_passed(checks, horizon):
            failure_reasons.add("generated_reference_unsafe_or_incomplete")
            for check in checks.values():
                reasons = check.get("reasons", []) if isinstance(check, dict) else []
                if isinstance(reasons, list):
                    failure_reasons.update(reason for reason in reasons if isinstance(reason, str) and len(reason) <= 80)
            evaluated[levels] = None
            return None
        for name in ("nominal", "pv10"):
            try:
                final_ev = _number(checks[name].get("final_ev_kwh"), "generated_reference_final_ev_kwh", 0)
                reference_target = _number(checks[name].get("validated_ev_target_kwh"), "generated_reference_ev_target_kwh", 0)
                candidate_ev = _number(candidate_checks[name].get("final_ev_kwh"), "candidate_final_ev_kwh", 0)
                candidate_target = _number(candidate_checks[name].get("validated_ev_target_kwh"), "candidate_ev_target_kwh", 0)
            except (ValueError, TypeError, AttributeError):
                failure_reasons.add("candidate_ev_evidence_invalid")
                evaluated[levels] = None
                return None
            if abs(reference_target - candidate_target) > 0.002 or final_ev < reference_target - 0.002:
                failure_reasons.add("generated_reference_ev_target_not_met")
                evaluated[levels] = None
                return None
            if abs(final_ev - candidate_ev) > 0.002:
                failure_reasons.add("ev_delivery_mismatch")
                evaluated[levels] = None
                return None
        costs = _native_adjusted_cost(base, checks)
        if costs is None:
            failure_reasons.add("generated_reference_costs_incomplete")
            evaluated[levels] = None
            return None
        record = {"result": result, "slots": native_slots, "windows": windows, "checks": checks, "costs": costs, "levels": levels}
        # Memoise the proposal identity, not every full 48-hour preview/check
        # payload. Only the bounded best-safe set needs to retain those objects.
        evaluated[levels] = None
        safe_references.append(record)
        safe_references.sort(key=lambda item: (max(item["costs"].values()), sum(item["costs"].values()), item["levels"]))
        del safe_references[GENERATED_REFERENCE_MAX_SEEDS:]
        return record

    try:
        for levels in product(coarse_levels, repeat=len(selected_geometry)):
            evaluate(levels)
        for pass_index in range(GENERATED_REFERENCE_REFINEMENT_PASSES):
            if not safe_references:
                break
            refinement_passes = pass_index + 1
            before = safe_references[0]["levels"]
            # Two adjacent perturbations cannot refine a coarse 50% seed down
            # to a valid 26% target. Search representable whole-percent levels
            # along each coordinate, retaining the best paired-safe outcome.
            for index in range(len(selected_geometry)):
                seed = safe_references[0]["levels"]
                for level in (0, *positive_levels):
                    proposal = list(seed)
                    proposal[index] = level
                    evaluate(tuple(proposal))
            if safe_references[0]["levels"] == before:
                break
    except RuntimeError as exc:
        if str(exc) == "generated_reference_search_budget_exhausted":
            budget_exhausted = True
        else:
            raise

    if not safe_references:
        if budget_exhausted:
            return _reference_failure("blocked", "search_budget_exhausted", provenance={"paired_evaluations": paired_evaluations})
        reason = "ev_delivery_mismatch" if "ev_delivery_mismatch" in failure_reasons else "generated_reference_unsafe_or_incomplete"
        return _reference_failure("unsafe", reason, provenance={"paired_evaluations": paired_evaluations, "failure_reasons": sorted(failure_reasons)})
    selected = min(safe_references, key=lambda item: (max(item["costs"].values()), sum(item["costs"].values()), item["levels"]))
    import hashlib
    import json

    schedule_fingerprint = hashlib.sha256(json.dumps(schedule, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    provenance = {
        "source_kind": "copied_low_rates_geometry_fixed_ev_schedule",
        "source_windows": [{"start": window["start"], "end": window["end"]} for window in geometry],
        "selected_source_window_indexes": [window["source_index"] for window in selected_geometry],
        "selected_target_levels_percent": list(selected["levels"]),
        "power_readiness": deepcopy(getattr(selected["result"], "power_readiness", {"active": False})),
        "selected_effective_charge_limits_kwh": deepcopy(selected["windows"][2]),
        "eligible_target_levels_percent": coarse_levels,
        "fixed_ev_schedule_fingerprint": schedule_fingerprint,
        "paired_evaluations": paired_evaluations,
        "evaluated_proposals": len(evaluated),
        "refinement_passes": refinement_passes,
        "search_budget_exhausted": budget_exhausted,
        "evaluation_unit": "individual_native_scenarios_in_complete_paired_proposals",
        "rejected_proposal_reasons": sorted(failure_reasons),
        "search_scope": "up_to_three_native_windows_coarse_levels_then_two_whole_percent_coordinate_passes_best_safe_within_budget",
    }
    return {"status": "paired_safe", "reason": None, **selected, "provenance": provenance}


def _public_checks(checks):
    """Drop the solver's disabled-EV sentinel from household-facing evidence."""
    public = deepcopy(checks)
    for check in public.values():
        check.pop("final_ev_kwh", None)
        check.pop("validated_ev_target_kwh", None)
    return public


def _native_comparison(base, options, snapshot, checks, reference_costs, candidate_costs=None, status="baseline_replayed"):
    """Describe actual-native comparison separately from the all-Demand diagnostic."""
    report = {
        "status": status,
        "baseline_kind": "ordinary_native_household_plan",
        "baseline_executable": True,
        "baseline_plan_valid": snapshot["plan_valid"],
        "baseline_charge_windows": deepcopy(snapshot["charge_windows"]),
        "baseline_charge_limits": deepcopy(snapshot["charge_limits"]),
        "baseline_initial_battery_soc_kwh": base.soc_kw,
        "baseline_grid_charge_support_w": options["grid_charge_support_w"],
        "baseline_import_limit_w": options["import_limit_w"],
        "baseline_checks": _public_checks(checks),
        "baseline_objective": deepcopy(reference_costs),
        "baseline_passed": all(check.get("passed") is True for check in checks.values()) if _paired_complete(checks, checks["nominal"].get("checked_minutes", 0)) else False,
        "same_terminal_value": True,
        "componentwise_non_regression": None,
        "positive_improvement_required": True,
        "tolerance": HOUSEHOLD_COST_TOLERANCE,
        "comparison_basis": "per_scenario_native_horizon_cost_plus_cycle_cost_minus_same_terminal_soc_value",
    }
    if candidate_costs is not None:
        report["candidate_objective"] = deepcopy(candidate_costs)
        report["candidate_minus_baseline"] = {name: candidate_costs[name] - reference_costs[name] for name in ("nominal", "pv10")}
        report["componentwise_non_regression"] = all(candidate_costs[name] <= reference_costs[name] + HOUSEHOLD_COST_TOLERANCE for name in ("nominal", "pv10"))
        report["positive_improvement"] = any(candidate_costs[name] < reference_costs[name] - HOUSEHOLD_COST_TOLERANCE for name in ("nominal", "pv10"))
    return report


def _demand_counterfactual_report(base, options, slots):
    """Return an explicitly non-executable all-Demand counterfactual report."""
    try:
        _, _, checks = _replay_demand_counterfactual(base, options, None, slots)
        complete = _paired_complete(checks, base.forecast_minutes)
        return {
            "status": "baseline_replayed" if complete else "baseline_replay_incomplete",
            "baseline_kind": "direct_all_demand_counterfactual",
            "baseline_executable": False,
            "baseline_checks": _public_checks(checks),
            "baseline_objective": _native_adjusted_cost(base, checks) if complete else None,
            "baseline_passed": _paired_passed(checks, base.forecast_minutes),
            "same_terminal_value": True,
            "comparison_basis": "paired_native_cost_plus_cycle_cost_minus_terminal_soc_value_diagnostic_only",
        }
    except Exception:
        return {
            "status": "baseline_replay_failed",
            "baseline_kind": "direct_all_demand_counterfactual",
            "baseline_executable": False,
            "baseline_checks": {},
            "baseline_objective": None,
            "baseline_passed": False,
        }


def evaluate_household_fallback(base, settings, unavailable_reason):
    """Stage only a paired-safe domestic schedule that improves on the ordinary native plan."""
    if not isinstance(settings, dict) or settings.get("grid_charge_support") is None or settings.get("control_model") != "deye_solarman":
        return None
    native_snapshot = _snapshot_native_plan(base)
    try:
        options = household_options(base, settings)
        from car_charging_plan import _collect_blockers, _forecast_inputs, build_native_slots

        blockers = [blocker for blocker in _collect_blockers(base, options) if blocker != "ev_from_battery_required"]
        if blockers:
            return _plan_report(base, "household_plan_inputs_blocked", domestic_plan_blockers=blockers)
        _forecast_inputs(base, options)
        from power_readiness import readiness_enabled, stage_household_readiness

        if readiness_enabled(base):
            service = stage_household_readiness(base, options, build_native_slots(base, options), unavailable_reason)
            if service is not None:
                return service
            if settings.get("household_economic_fallback") is not True:
                return _plan_report(base, unavailable_reason, domestic_only=False, power_readiness={"active": False})
        if native_snapshot is None:
            return _plan_report(
                base,
                "native_baseline_unavailable",
                native_comparison={"status": "baseline_unavailable", "baseline_kind": "ordinary_native_household_plan", "baseline_executable": True, "baseline_checks": {}},
            )

        slots = build_native_slots(base, options)
        try:
            _, _, native_checks = _replay_native_plan(base, options, native_snapshot, slots)
            if not _paired_complete(native_checks, base.forecast_minutes):
                raise ValueError("Native household baseline replay was incomplete")
            native_costs = _native_adjusted_cost(base, native_checks)
            if native_costs is None:
                raise ValueError("Native household baseline costs were unavailable")
        except (ValueError, TypeError, KeyError, AttributeError):
            return _plan_report(
                base,
                "native_baseline_unavailable",
                native_comparison={"status": "baseline_replay_failed", "baseline_kind": "ordinary_native_household_plan", "baseline_executable": True, "baseline_checks": {}},
            )

        native_comparison = _native_comparison(base, options, native_snapshot, native_checks, native_costs)
        options["native_reference_costs"] = deepcopy(native_costs)
        search_diagnostics = {}
        candidate = _solve_and_replay(base, options, search_diagnostics)
        if candidate is None:
            demand_comparison = _demand_counterfactual_report(base, options, slots)
            rejected = search_diagnostics.get("componentwise_rejections", [])
            reason = "no_componentwise_savings_over_native" if rejected else "household_plan_not_found"
            native_comparison["status"] = "not_componentwise_cheaper" if rejected else "baseline_replayed_no_candidate"
            native_comparison["componentwise_rejected_candidates"] = deepcopy(rejected)
            return _plan_report(
                base,
                reason,
                solver_status="no_componentwise_savings" if rejected else "bounded_no_plan",
                native_comparison=native_comparison,
                demand_comparison=demand_comparison,
            )
        result, slots, windows, checks, options = candidate
        if not _paired_passed(checks, base.forecast_minutes):
            return _plan_report(base, "household_candidate_replay_failed", native_checks=_public_checks(checks), native_comparison=native_comparison)

        candidate_costs = _native_adjusted_cost(base, checks)
        demand_comparison = _demand_counterfactual_report(base, options, slots)
        if candidate_costs is None:
            return _plan_report(
                base,
                "paired_native_cost_comparison_unsupported",
                native_comparison=native_comparison,
                demand_comparison=demand_comparison,
            )
        if not _componentwise_improvement(candidate_costs, native_costs):
            native_comparison = _native_comparison(base, options, native_snapshot, native_checks, native_costs, candidate_costs, "not_componentwise_cheaper")
            return _plan_report(
                base,
                "no_componentwise_savings_over_native",
                native_comparison=native_comparison,
                demand_comparison=demand_comparison,
            )
        native_comparison = _native_comparison(base, options, native_snapshot, native_checks, native_costs, candidate_costs, "componentwise_cheaper_than_native")

        from car_charging_execute import command_fingerprint, execution_fingerprint

        command_steps = []
        for action, slot in zip(result.actions, slots):
            target = 0
            if action.battery_mode == "charge":
                target = round(next(limit for window, limit in zip(windows[1], windows[2]) if window["start"] <= base.minutes_now + slot.start_min < window["end"]) * 100 / base.soc_max)
            command_steps.append(
                {
                    "start": base.minutes_now + slot.start_min,
                    "end": base.minutes_now + slot.start_min + slot.duration_min,
                    "target_soc": target,
                    "freeze": False,
                    "power_w": None,
                    "ev_on": False,
                }
            )
        if any(window for window in windows[0]) or any(command["ev_on"] for command in command_steps):
            return _plan_report(base, "domestic_ev_off_invariant_failed")

        report = _plan_report(
            base,
            unavailable_reason,
            status="candidate",
            household_executor="grid_charge_support",
            ready_to_activate=False,
            execution_enabled=False,
            grid_charge_support_w=options["grid_charge_support_w"],
            import_limit_w=options["import_limit_w"],
            modelled_import_limit_w=options["import_limit_w"],
            control_model="deye_solarman",
            counterfactual_controls=False,
            supported_battery_modes=["demand", "charge"],
            battery_min_window_minutes=options["battery_min_window_minutes"],
            session_closed=False,
            ev_session_renewed=False,
            proposed_ev_windows=[],
            proposed_charge_windows=windows[1],
            proposed_charge_limits=windows[2],
            native_checks=_public_checks(checks),
            solver_objective=result.total_cost,
            native_adjusted_costs=candidate_costs,
            solver_grid_cost=sum(action.grid_import_kwh * action.import_price - action.grid_export_kwh * action.export_price for action in result.actions),
            solver_ev_ac_kwh=0,
            solver_pv_curtailed_kwh=result.diagnostics.pv_curtailed_total_kwh,
            cost_unit=base.currency_symbols[1],
            cost_convention="native_adjusted_costs_compare_nominal_and_pv10_separately_against_the_ordinary_native_plan; horizon_energy_cost_excludes_cycle_cost_terminal_value_and_standing_charge",
            grid_charge_support_convention="existing_shared_support_with_load_priority_and_bounded_battery_support",
            bounded_search=True,
            native_comparison=native_comparison,
            demand_comparison=demand_comparison,
            household_unavailable_reason=unavailable_reason,
            activation_requirements=["household_fallback_opt_in", "execution_authorised_or_apply_commissioning", "shared_grid_support_on", "observed_ev_relay_off"],
            execution_blockers=[],
            apply_candidate_staged=True,
            solver_forecast_basis=options["forecast_basis"],
        )
        base.car_charging_pending = {
            "domestic_only": True,
            "report": deepcopy(report),
            "options": deepcopy(options),
            "commands": deepcopy(command_steps),
            "ev_windows": [],
            "initial_battery_soc_kwh": base.soc_kw,
            "fingerprint": execution_fingerprint(base),
            "command_fingerprint": command_fingerprint(command_steps),
            "replay_inputs": deepcopy((result, slots, windows)),
        }
        return report
    except HouseholdFallbackBlocked as exc:
        status = "household_waiting" if exc.waiting else "household_fallback"
        return _plan_report(base, exc.reason, status=status)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return _plan_report(base, "household_plan_inputs_invalid", household_error_type=type(exc).__name__)


def execution_evidence(base, settings, pending):
    """Revalidate only domestic hardware/support evidence, never EV SOC or permission."""
    from car_charging_plan import _collect_blockers, _forecast_inputs

    options = household_options(base, settings)
    blockers = [blocker for blocker in _collect_blockers(base, options) if blocker != "ev_from_battery_required"]
    if blockers:
        raise HouseholdFallbackBlocked("household_plan_inputs_blocked")
    if not _paired_passed(pending.get("report", {}).get("native_checks", {}), base.forecast_minutes):
        raise HouseholdFallbackBlocked("native_checks_failed")
    if pending.get("options", {}).get("grid_charge_support_w") != options["grid_charge_support_w"]:
        raise HouseholdFallbackBlocked("grid_charge_support_changed")
    if getattr(base, "set_read_only", False) or getattr(base, "set_read_only_axle", False):
        raise HouseholdFallbackBlocked("read_only")
    if any(getattr(base, name, False) for name in ("had_errors", "fatal_error", "inverter_needs_reset")):
        raise HouseholdFallbackBlocked("native_state_not_ready")
    if base.get_arg("template", False):
        raise HouseholdFallbackBlocked("template_mode")
    inverters = getattr(base, "inverters", None)
    if not isinstance(inverters, list) or len(inverters) != 1:
        raise HouseholdFallbackBlocked("single_inverter_required")
    inverter = inverters[0]
    if getattr(inverter, "inverter_type", None) != "DeyeSolarman" or getattr(inverter, "in_calibration", True):
        raise HouseholdFallbackBlocked("template_inverter_required")
    from inverter_service_control import validate_template

    try:
        validate_template(inverter)
    except (ValueError, TypeError, KeyError, AttributeError):
        raise HouseholdFallbackBlocked("template_contract_invalid") from None
    if getattr(base, "manual_soc_max_keep", None):
        raise HouseholdFallbackBlocked("manual_ceiling_unsupported")
    if getattr(base, "octopus_intelligent_charging", False) or base.get_arg("octopus_intelligent_slot", None, indirect=False):
        raise HouseholdFallbackBlocked("external_ev_scheduler_unsupported")
    hooks = getattr(getattr(base, "plugin_system", None), "hooks", {})
    if hooks.get("on_plan_executed"):
        raise HouseholdFallbackBlocked("external_plan_consumer_unsupported")
    minute = _parse_observation_minute(base, options)
    floor = max(base.reserve, base.all_active_keep.get(int(minute), 0) * base.soc_max / 100)
    if not math.isfinite(base.soc_kw) or base.soc_kw < floor - 0.002:
        raise HouseholdFallbackBlocked("domestic_soc_invalid")
    _forecast_inputs(base, options)
    if pending.get("readiness_service") is True:
        from power_readiness import readiness_input_fingerprint

        if pending.get("readiness_input_fingerprint") != readiness_input_fingerprint(base):
            raise HouseholdFallbackBlocked("power_readiness_inputs_changed")
    from car_charging_plan import _parse_entity_timestamp

    now = _parse_entity_timestamp(options["observations_checked_at"], "household_observation_clock")
    return options, now, (now - base.midnight_utc).total_seconds() / 60


def _handback_relay_off(base, settings):
    """Require a live, well-formed relay-OFF observation without reading grid support."""
    from car_charging_plan import _aware, _observation_clock

    if not isinstance(settings, dict) or not isinstance(settings.get("relay_entity"), str) or not settings["relay_entity"]:
        raise HouseholdFallbackBlocked("relay_observation_unavailable", waiting=True)
    now = _aware(_observation_clock(), "household_observation_clock")
    ha_interface = getattr(base, "ha_interface", None)
    is_alive = getattr(ha_interface, "is_alive", None)
    try:
        if not callable(is_alive) or is_alive() is not True:
            raise HouseholdFallbackBlocked("household_control_transport_unavailable", waiting=True)
    except HouseholdFallbackBlocked:
        raise
    except Exception:
        raise HouseholdFallbackBlocked("household_control_transport_unavailable", waiting=True) from None
    relay = _read_live_control_state(base, settings["relay_entity"])
    if relay is None or _future_observation(relay, now):
        raise HouseholdFallbackBlocked("relay_observation_unavailable", waiting=True)
    if relay["state"] != "off":
        raise HouseholdFallbackBlocked("waiting_for_ev_relay_off", waiting=True)
    return now


def _validate_handback_template(base):
    """Keep the native writer on its validated Deye profile and local safety gates."""
    if getattr(base, "set_read_only", False) or getattr(base, "set_read_only_axle", False):
        raise HouseholdFallbackBlocked("read_only")
    if any(getattr(base, name, False) for name in ("had_errors", "fatal_error", "inverter_needs_reset")):
        raise HouseholdFallbackBlocked("native_state_not_ready")
    if base.get_arg("template", False):
        raise HouseholdFallbackBlocked("template_mode")
    inverters = getattr(base, "inverters", None)
    if not isinstance(inverters, list) or len(inverters) != 1:
        raise HouseholdFallbackBlocked("single_inverter_required")
    inverter = inverters[0]
    if getattr(inverter, "inverter_type", None) != "DeyeSolarman" or getattr(inverter, "in_calibration", True):
        raise HouseholdFallbackBlocked("template_inverter_required")
    from inverter_service_control import validate_template

    try:
        validate_template(inverter)
    except (ValueError, TypeError, KeyError, AttributeError):
        raise HouseholdFallbackBlocked("template_contract_invalid") from None
    if getattr(base, "manual_soc_max_keep", None):
        raise HouseholdFallbackBlocked("manual_ceiling_unsupported")
    if getattr(base, "octopus_intelligent_charging", False) or base.get_arg("octopus_intelligent_slot", None, indirect=False):
        raise HouseholdFallbackBlocked("external_ev_scheduler_unsupported")
    hooks = getattr(getattr(base, "plugin_system", None), "hooks", {})
    if hooks.get("on_plan_executed"):
        raise HouseholdFallbackBlocked("external_plan_consumer_unsupported")
    return inverter


def safe_demand_handback(base, settings):
    """Cancel a domestic target after live relay-OFF/template checks, independent of support state."""
    _handback_relay_off(base, settings)
    inverter = _validate_handback_template(base)
    from inverter_service_control import dispatch_template

    try:
        observed = dispatch_template(inverter, 0)
    except Exception:
        raise HouseholdFallbackBlocked("safe_demand_not_observed", attempted=True) from None
    if observed is not True:
        raise HouseholdFallbackBlocked("safe_demand_not_observed", attempted=True)
    try:
        _handback_relay_off(base, settings)
        _validate_handback_template(base)
    except HouseholdFallbackBlocked as exc:
        raise HouseholdFallbackBlocked(exc.reason, waiting=exc.waiting, attempted=True) from None
    return True


def validate_native_handoff(base, settings, report):
    """Require current paired-safe evidence for the exact ordinary plan before native handoff."""
    from car_charging_plan import _collect_blockers, _forecast_inputs, _observation_clock, _parse_entity_timestamp

    _handback_relay_off(base, settings)
    comparison = report.get("native_comparison") if isinstance(report, dict) else None
    if not isinstance(comparison, dict) or comparison.get("baseline_kind") != "ordinary_native_household_plan":
        raise HouseholdFallbackBlocked("native_baseline_unavailable")
    try:
        generated = _parse_entity_timestamp(report.get("generated_at"), "native_handoff_report_generated_at")
        age_seconds = (_observation_clock() - generated).total_seconds()
    except (ValueError, TypeError, KeyError, AttributeError):
        raise HouseholdFallbackBlocked("native_safety_evidence_invalid") from None
    if not 0 <= age_seconds <= 300:
        raise HouseholdFallbackBlocked("native_safety_evidence_stale")

    snapshot = _snapshot_native_plan(base)
    if snapshot is None or comparison.get("baseline_plan_valid") is not True or comparison.get("baseline_charge_windows") != snapshot["charge_windows"] or comparison.get("baseline_charge_limits") != snapshot["charge_limits"]:
        raise HouseholdFallbackBlocked("native_schedule_changed")

    try:
        options = household_options(base, settings)
        blockers = [blocker for blocker in _collect_blockers(base, options) if blocker != "ev_from_battery_required"]
        if blockers:
            raise HouseholdFallbackBlocked("native_handoff_inputs_blocked")
        _forecast_inputs(base, options)
    except HouseholdFallbackBlocked:
        raise
    except (ValueError, TypeError, KeyError, AttributeError):
        raise HouseholdFallbackBlocked("native_handoff_inputs_invalid") from None

    checks = comparison.get("baseline_checks")
    initial_soc = comparison.get("baseline_initial_battery_soc_kwh")
    same_replay_inputs = (
        isinstance(initial_soc, (int, float))
        and not isinstance(initial_soc, bool)
        and math.isfinite(initial_soc)
        and abs(initial_soc - base.soc_kw) <= 0.002
        and comparison.get("baseline_grid_charge_support_w") == options["grid_charge_support_w"]
        and comparison.get("baseline_import_limit_w") == options["import_limit_w"]
    )
    replayed_now = False
    if not same_replay_inputs or not _paired_complete(checks, base.forecast_minutes):
        try:
            _, _, checks = _replay_native_plan(base, options, snapshot)
        except (ValueError, TypeError, KeyError, AttributeError):
            raise HouseholdFallbackBlocked("native_baseline_replay_failed") from None
        replayed_now = True

    if not _paired_complete(checks, base.forecast_minutes):
        raise HouseholdFallbackBlocked("native_baseline_replay_incomplete")
    costs = _native_adjusted_cost(base, checks)
    if costs is None:
        raise HouseholdFallbackBlocked("native_baseline_economics_unavailable")

    passed = _paired_passed(checks, base.forecast_minutes)
    comparison.update(
        status="current_native_replayed" if replayed_now else "current_native_revalidated",
        baseline_checks=_public_checks(checks),
        baseline_objective=costs,
        baseline_passed=passed,
        baseline_initial_battery_soc_kwh=base.soc_kw,
        baseline_grid_charge_support_w=options["grid_charge_support_w"],
        baseline_import_limit_w=options["import_limit_w"],
    )
    if not passed:
        raise HouseholdFallbackBlocked("native_baseline_safety_failed")
    return checks


def _parse_observation_minute(base, options):
    """Convert the checked observation clock to Predbat's absolute-minute axis."""
    from car_charging_plan import _parse_entity_timestamp

    now = _parse_entity_timestamp(options["observations_checked_at"], "household_observation_clock")
    return (now - base.midnight_utc).total_seconds() / 60


def replay_pending_household(base, options, pending):
    """Replay unchanged OFF-EV commands from current domestic SOC in both cases."""
    from car_charging_plan import _replay

    result, slots, windows = pending["replay_inputs"]
    checks = {name: _replay(base, options, result, slots, windows, scenario) for name, scenario in (("nominal", 0), ("pv10", 1))}
    for check in checks.values():
        check.pop("_charge_peak_kwh", None)
    return checks


def execute_household_candidate(base):
    """Dispatch only the current OFF-EV command and publish a short, non-EV receipt."""
    from car_charging_config import read_car_charging_config
    from car_charging_execute import (
        _control_fingerprint,
        _intent,
        _publish_ev,
        _report,
        _withdraw_for_household,
        command_fingerprint,
        execution_fingerprint,
    )
    from car_charging_plan import _parse_entity_timestamp

    pending = getattr(base, "car_charging_pending", None)
    settings = read_car_charging_config(base)
    attempted = False
    try:
        if not isinstance(pending, dict) or pending.get("domestic_only") is not True:
            raise HouseholdFallbackBlocked("no_domestic_candidate")
        report = pending.get("report", {})
        if report.get("status") != "candidate" or report.get("domestic_only") is not True or report.get("apply_candidate_staged") is not True:
            raise HouseholdFallbackBlocked("domestic_candidate_not_accepted")
        options, now, minute = execution_evidence(base, settings, pending)
        generated = _parse_entity_timestamp(report.get("generated_at"), "household_candidate_generated_at")
        if not 0 <= (now - generated).total_seconds() <= 300:
            raise HouseholdFallbackBlocked("candidate_stale_or_future")
        if pending.get("fingerprint") != execution_fingerprint(base):
            raise HouseholdFallbackBlocked("configuration_changed")
        commands = pending.get("commands")
        if not isinstance(commands, list) or len(commands) != base.forecast_minutes // 5:
            raise HouseholdFallbackBlocked("incomplete_command_horizon")
        if pending.get("command_fingerprint") != command_fingerprint(commands):
            raise HouseholdFallbackBlocked("candidate_commands_changed")
        if not all(command.get("ev_on") is False and command.get("freeze") is False and command.get("power_w") is None for command in commands):
            raise HouseholdFallbackBlocked("domestic_ev_off_invariant_failed")
        for index, command in enumerate(commands):
            start = base.minutes_now + index * 5
            if command.get("start") != start or command.get("end") != start + 5:
                raise HouseholdFallbackBlocked("command_timeline_mismatch")
            _intent(command, base.inverters[0], True)
        command = commands[0]
        if not command["start"] <= minute < command["end"]:
            raise HouseholdFallbackBlocked("current_command_expired")
        if not _paired_passed(report.get("native_checks", {}), base.forecast_minutes):
            raise HouseholdFallbackBlocked("native_checks_failed")

        from battery_window import commands_respect_windows, observed_window, record_observed_window

        if not commands_respect_windows(
            commands,
            options["battery_min_window_minutes"],
            options["initial_window_mode"],
            options["initial_window_remaining_minutes"],
        ):
            raise HouseholdFallbackBlocked("battery_window_candidate_invalid")
        previous_window = observed_window(base, now) if options["battery_min_window_minutes"] > 5 else None
        if abs(base.soc_kw - pending["initial_battery_soc_kwh"]) > 0.002:
            checks = replay_pending_household(base, options, pending)
            if not _paired_passed(checks, base.forecast_minutes):
                raise HouseholdFallbackBlocked("dispatch_replay_failed")
            report = {**report, "native_checks": checks, "dispatch_soc_revalidated_kwh": base.soc_kw}
            pending["report"] = deepcopy(report)
            base.car_charging_report = deepcopy(report)

        target, freeze, power = _intent(command, base.inverters[0], True)
        if freeze or power is not None:
            raise HouseholdFallbackBlocked("unsupported_template_command")
        base.inverter_rate_intent = {}
        attempted = True
        from inverter_service_control import dispatch_template

        if dispatch_template(base.inverters[0], target, freeze=False, discharge_power_w=None, charge_ceiling=True) is not True:
            raise HouseholdFallbackBlocked("inverter_command_not_observed")

        options_after, after, after_minute = execution_evidence(base, settings, pending)
        if pending.get("fingerprint") != execution_fingerprint(base):
            raise HouseholdFallbackBlocked("configuration_changed_during_dispatch")
        if options_after["grid_charge_support_w"] != options["grid_charge_support_w"]:
            raise HouseholdFallbackBlocked("grid_charge_support_changed")
        if not command["start"] <= after_minute < command["end"]:
            raise HouseholdFallbackBlocked("command_expired_during_dispatch")

        now = after
        expires = min(base.midnight_utc + timedelta(minutes=command["end"]), now + timedelta(seconds=300))
        base.car_charging_execution = {
            "generated_at": now.isoformat(),
            "plan_generated_at": report["generated_at"],
            "expires_at": expires.isoformat(),
            "command": deepcopy(command),
            "fingerprint": execution_fingerprint(base),
            "control_fingerprint": _control_fingerprint(base.inverters[0]),
            "observations_match": True,
            "hardware_confirmed": False,
            "domestic_only": True,
        }
        base.isCharging = target > 0
        base.isExporting = False
        record_observed_window(base, command, now, previous_window)
        _publish_ev(base)
        _report(base, "applied", True, "inverter_observed_not_hardware_confirmed", attempted=True)
        return ("Charging" if target > 0 else "Demand", "Domestic-only plan; HA observations only")
    except (HouseholdFallbackBlocked, ValueError, TypeError, KeyError, AttributeError) as exc:
        reason = exc.reason if isinstance(exc, HouseholdFallbackBlocked) else "execution_validation_failed"
        return _withdraw_for_household(base, reason, attempted)


def validate_household_receipt(base, receipt):
    """Validate an applied domestic receipt without reading EV SOC/permission."""
    from car_charging_config import read_car_charging_config
    from car_charging_execute import _control_fingerprint, execution_fingerprint
    from car_charging_plan import _parse_entity_timestamp

    settings = read_car_charging_config(base)
    pending = getattr(base, "car_charging_pending", None)
    if not isinstance(pending, dict) or pending.get("domestic_only") is not True:
        raise HouseholdFallbackBlocked("no_domestic_candidate")
    _, now, minute = execution_evidence(base, settings, pending)
    if not isinstance(receipt, dict) or receipt.get("domestic_only") is not True:
        raise HouseholdFallbackBlocked("domestic_receipt_invalid")
    if receipt.get("fingerprint") != execution_fingerprint(base):
        raise HouseholdFallbackBlocked("configuration_changed")
    if receipt.get("control_fingerprint") != _control_fingerprint(base.inverters[0]):
        raise HouseholdFallbackBlocked("observed_controls_changed")
    if not 0 <= (now - _parse_entity_timestamp(receipt.get("generated_at"), "receipt_generated_at")).total_seconds() < 300:
        raise HouseholdFallbackBlocked("receipt_expired")
    if now >= _parse_entity_timestamp(receipt.get("expires_at"), "receipt_expires_at"):
        raise HouseholdFallbackBlocked("command_authority_expired")
    command = receipt.get("command", {})
    if not command.get("start") <= minute < command.get("end") or command.get("ev_on") is not False:
        raise HouseholdFallbackBlocked("domestic_receipt_command_invalid")
