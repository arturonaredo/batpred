"""Opt-in cheap-period energy readiness, never a raised physical safety reserve.

The policy requests ordinary shared charge targets so cheap grid energy supplies
normal loads and the battery remains usable for demand above the import setting.
All resulting controls still require complete independent native safety replay.
"""

from copy import copy, deepcopy
from dataclasses import is_dataclass, replace
import math
from types import SimpleNamespace


def readiness_settings(base):
    """Validate explicit policy settings in native minor-currency and percentage units."""
    raw = getattr(base, "args", {}).get("battery_power_readiness")
    if raw is None:
        return {"enabled": False}
    defaults = {"enabled": False, "auto_ev_now": False, "target_soc_percent": 35, "maximum_soc_percent": 60, "minimum_cheap_minutes": 360, "cheap_price_ceiling": 12.0}
    if not isinstance(raw, dict) or set(raw) - set(defaults):
        raise ValueError("Invalid power readiness settings")
    settings = {**defaults, **raw}
    if type(settings["enabled"]) is not bool or type(settings["auto_ev_now"]) is not bool:
        raise ValueError("Invalid power readiness enable flag")
    for name, low, high in (("target_soc_percent", 1, 95), ("maximum_soc_percent", 1, 95), ("minimum_cheap_minutes", 30, 2880), ("cheap_price_ceiling", -1000, 1000)):
        value = settings[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("Invalid power readiness " + name)
    if settings["maximum_soc_percent"] < settings["target_soc_percent"] or settings["minimum_cheap_minutes"] % 5:
        raise ValueError("Invalid power readiness bounds")
    if any(settings[name] != int(settings[name]) for name in ("target_soc_percent", "maximum_soc_percent")):
        raise ValueError("Power readiness targets require whole percentages")
    return settings


def readiness_enabled(base):
    """Return only the explicit service opt-in, without inferring authority from tariffs."""
    return readiness_settings(base)["enabled"]


def cheap_readiness_periods(base, slots, minimum=None):
    """Share tariff eligibility between domestic targets and temporary EV scheduling priority."""
    settings = readiness_settings(base)
    if not settings["enabled"]:
        return []
    minimum = max(settings["minimum_cheap_minutes"], minimum or 0)
    prices = []
    for slot in slots:
        values = [base.rate_import.get(base.minutes_now + slot.start_min + offset) for offset in range(slot.duration_min)]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError("Incomplete power readiness rates")
        prices.append(sum(values) / slot.duration_min)
    if any(type(price) not in (int, float) or not math.isfinite(price) for price in prices):
        raise ValueError("Invalid power readiness rates")
    # This is a power-availability service: a brief cheaper opportunity later
    # cannot erase a long band that is already below the user's cheap ceiling.
    threshold = settings["cheap_price_ceiling"]
    periods = []
    index = 0
    while index < len(slots):
        if prices[index] > threshold:
            index += 1
            continue
        first = index
        while index < len(slots) and prices[index] <= threshold:
            index += 1
        start = base.minutes_now + slots[first].start_min
        end = base.minutes_now + slots[index - 1].start_min + slots[index - 1].duration_min
        observed_prior_minutes = 0
        if first == 0:
            for minute in range(start - 5, start - int(minimum) - 1, -5):
                price = base.rate_import.get(minute)
                if type(price) not in (int, float) or not math.isfinite(price) or price > threshold:
                    break
                observed_prior_minutes += 5
        if end - start + observed_prior_minutes < minimum:
            continue
        periods.append({"first": first, "stop": index, "start": start, "end": end})
    return periods


def current_readiness_period(base, observed_now, minimum=None):
    """Require a known qualifying cheap band covering the actual wall-clock observation."""
    slots = [SimpleNamespace(start_min=minute, duration_min=5) for minute in range(0, base.forecast_minutes, 5)]
    periods = cheap_readiness_periods(base, slots, minimum)
    minute = (observed_now - base.midnight_utc).total_seconds() / 60
    return next((period for period in periods if period["start"] <= minute < period["end"]), None)


def readiness_ev_priority(base, requested, mode, observed_now):
    """Temporarily prefer an immediate safe start without writing or losing the requested selector."""
    settings = readiness_settings(base)
    if mode != "apply" or requested not in ("cost", "asap") or not settings["enabled"] or not settings["auto_ev_now"]:
        return requested
    try:
        return "now" if current_readiness_period(base, observed_now, minimum=360) else requested
    except (ValueError, KeyError, AttributeError, TypeError):
        return requested


def readiness_profile(base, options, result, slots):
    """Derive calendar-independent cheap intervals and energy needed for forecast peak bursts."""
    settings = readiness_settings(base)
    inactive = {"active": False, "periods": []}
    if not settings["enabled"]:
        return inactive
    if options.get("control_model") != "deye_solarman" or not options.get("grid_charge_support_w"):
        raise ValueError("Power readiness requires shared grid support")
    if not slots or len(result.actions) != len(slots):
        raise ValueError("Incomplete power readiness action timeline")
    periods = []
    for period in cheap_readiness_periods(base, slots):
        first, index, start, end = (period[name] for name in ("first", "stop", "start", "end"))
        debt = maximum_debt = 0.0
        discharge_efficiency = base.inverter_loss * base.battery_loss_discharge
        charge_efficiency = base.inverter_loss * base.battery_loss
        if not 0 < discharge_efficiency <= 1 or not 0 < charge_efficiency <= 1:
            raise ValueError("Invalid power readiness efficiencies")
        for slot, action in zip(slots[first:index], result.actions[first:index]):
            minute = slot.start_min
            duration = slot.duration_min
            house = max(base.load_minutes_step[minute], base.load_minutes_step10[minute])
            pv = min(base.pv_forecast_minute_step[minute], base.pv_forecast_minute10_step[minute]) * base.inverter_loss
            pv = min(pv, base.inverter_limit * duration)
            ev = base.car_charging_rate[0] * duration / 60 if action.ev_action == "on" else 0
            headroom = options["grid_charge_support_w"] * duration / 60000 - (house + ev - pv)
            draw = -headroom / discharge_efficiency if headroom < 0 else -min(headroom * charge_efficiency, base.battery_rate_max_charge * duration)
            debt = max(0.0, debt + draw)
            maximum_debt = max(maximum_debt, debt)
        required_percent = math.ceil((base.reserve + maximum_debt * 1.1) * 100 / base.soc_max - 1e-9)
        target = min(settings["maximum_soc_percent"], max(settings["target_soc_percent"], required_percent, math.ceil(base.reserve * 100 / base.soc_max) + 1))
        if target <= math.ceil(base.reserve * 100 / base.soc_max):
            raise ValueError("Readiness target cannot be represented above safety reserve")
        periods.append({"start": start, "end": end, "target_soc_percent": math.ceil(target), "forecast_peak_energy_kwh": maximum_debt, "target_capped": required_percent > settings["maximum_soc_percent"]})
    if not periods:
        return inactive
    return {
        "active": True,
        "target_soc_percent": max(period["target_soc_percent"] for period in periods),
        "periods": periods,
        "cheap_threshold": settings["cheap_price_ceiling"],
        "minimum_cheap_minutes": settings["minimum_cheap_minutes"],
        "physical_reserve_unchanged_kwh": base.reserve,
        "buffer_usable_for_peak_support": True,
        "cost_basis": "explicit_power_availability_service_not_measured_energy_savings",
        "solver_flow_estimates_pre_policy": True,
    }


def prepare_candidate(base, options, result, slots, windows=None):
    """Overlay the same independent readiness targets on candidate or reference controls."""
    profile = readiness_profile(base, options, result, slots)
    if not profile["active"]:
        return result
    from car_charging_plan import _native_windows, _solver_charge_run_limits
    from battery_window import commands_respect_windows

    if windows is None:
        windows = _native_windows(base, result, slots)
        limits = _solver_charge_run_limits(base, result, slots)
        windows = (windows[0], windows[1], limits)
    ev_windows, inherited_windows, inherited_limits = deepcopy(windows)
    actions, charge_windows, charge_limits, commands = [], [], [], []
    for slot, action in zip(slots, result.actions):
        start = base.minutes_now + slot.start_min
        end = start + slot.duration_min
        target = next((limit for window, limit in zip(inherited_windows, inherited_limits) if window["start"] <= start and end <= window["end"]), 0)
        requested = next((period["target_soc_percent"] for period in profile["periods"] if period["start"] <= start and end <= period["end"]), 0)
        # Existing mode commitments and solar-only EV supply remain authoritative.
        committed = slot.start_min < options.get("initial_window_remaining_minutes", 0) and options.get("initial_window_mode") != "charge"
        solar_ev = options.get("ev_charging_priority") == "solar" and action.ev_action == "on"
        mode = action.battery_mode
        if requested and not committed and not solar_ev:
            if "charge" not in getattr(slot, "allowed_battery_modes", ("demand", "charge")):
                raise ValueError("Power readiness conflicts with explicit manual battery mode")
            mode = "charge"
            target = max(target, base.soc_max * requested / 100)
        new_action = replace(action, battery_mode=mode) if is_dataclass(action) else copy(action)
        if not is_dataclass(action):
            new_action.battery_mode = mode
        actions.append(new_action)
        if mode in ("charge", "hold"):
            if charge_windows and charge_windows[-1]["end"] == start and charge_limits[-1] == target and charge_windows[-1]["average"] == slot.import_price:
                charge_windows[-1]["end"] = end
            else:
                charge_windows.append({"start": start, "end": end, "average": slot.import_price})
                charge_limits.append(target)
        commands.append({"start": start, "end": end, "target_soc": round(target * 100 / base.soc_max) if mode == "charge" else 0, "freeze": mode == "hold", "power_w": None, "ev_on": action.ev_action == "on"})
    if not commands_respect_windows(commands, options.get("battery_min_window_minutes", 5), options.get("initial_window_mode"), options.get("initial_window_remaining_minutes", 0)):
        raise ValueError("Power readiness violates battery mode commitment")
    prepared = SimpleNamespace(**{**vars(result), "actions": tuple(actions)})
    prepared.power_readiness = profile
    prepared.native_policy_windows = (ev_windows, charge_windows, charge_limits)
    return prepared


def readiness_input_fingerprint(base):
    """Bind a domestic service to actual tariff and forecast inputs while SOC is independently replayed."""
    import hashlib
    import json

    offsets = range(0, base.forecast_minutes, 5)
    payload = {
        "settings": readiness_settings(base),
        "minutes_now": base.minutes_now,
        "forecast_minutes": base.forecast_minutes,
        "rates": [base.rate_import.get(minute) for minute in range(base.minutes_now - int(readiness_settings(base).get("minimum_cheap_minutes", 360)), base.minutes_now + base.forecast_minutes)],
        "forecasts": {name: [getattr(base, name).get(offset) for offset in offsets] for name in ("load_minutes_step", "load_minutes_step10", "pv_forecast_minute_step", "pv_forecast_minute10_step")},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def stage_household_readiness(base, options, slots, unavailable_reason):
    """Stage an explicitly authorised OFF-EV power service without pretending it is an energy saving."""
    from car_charging_plan import _candidate_commands, _replay_candidate
    from car_charging_household import _native_adjusted_cost, _paired_passed, _plan_report, _public_checks
    from car_charging_execute import command_fingerprint, execution_fingerprint

    actions = tuple(SimpleNamespace(slot_index=slot.index, battery_mode="demand", ev_action="off", discharge_limit_kw=None) for slot in slots)
    seed = SimpleNamespace(status="power_readiness_service", actions=actions, total_cost=None)
    result = prepare_candidate(base, options, seed, slots, ([], [], []))
    if not getattr(result, "power_readiness", {}).get("active"):
        return None
    windows, checks = _replay_candidate(base, options, result, slots)
    if not _paired_passed(checks, base.forecast_minutes):
        return _plan_report(base, "power_readiness_native_replay_failed", native_checks=_public_checks(checks), power_readiness=result.power_readiness)
    costs = _native_adjusted_cost(base, checks)
    if costs is None:
        return _plan_report(base, "power_readiness_costs_unavailable")
    commands = _candidate_commands(base, result, slots, windows)
    report = _plan_report(
        base,
        unavailable_reason,
        status="candidate",
        household_executor="grid_charge_support",
        power_readiness_service=True,
        power_readiness=deepcopy(result.power_readiness),
        battery_plan_improvement=False,
        service_costs=costs,
        cost_convention="explicit_power_availability_service_paired_native_cost_not_claimed_savings",
        control_model="deye_solarman",
        grid_charge_support_w=options["grid_charge_support_w"],
        import_limit_w=options["import_limit_w"],
        battery_min_window_minutes=options["battery_min_window_minutes"],
        proposed_charge_windows=deepcopy(windows[1]),
        proposed_charge_limits=deepcopy(windows[2]),
        native_checks=_public_checks(checks),
        apply_candidate_staged=True,
        execution_blockers=[],
        hardware_confirmed=False,
    )
    base.car_charging_pending = {
        "domestic_only": True,
        "readiness_service": True,
        "readiness_input_fingerprint": readiness_input_fingerprint(base),
        "report": deepcopy(report),
        "options": deepcopy(options),
        "commands": commands,
        "ev_windows": [],
        "initial_battery_soc_kwh": base.soc_kw,
        "fingerprint": execution_fingerprint(base),
        "command_fingerprint": command_fingerprint(commands),
        "replay_inputs": deepcopy((result, slots, windows)),
    }
    return report
