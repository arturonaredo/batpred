"""Opt-in joint scheduling against native forecasts, isolated from executable plans.

The solver makes real coupled EV/battery decisions, then a fresh native Prediction
replays them. This first fork integration is deliberately shadow-only: the native
window executor coalesces adjacent targets, and a mathematical availability window
is not positive evidence that an actual charger is connected. No schedule produced
here is assigned to car_charging_slots or to the inverter's *_best controls.
"""

from copy import copy
from datetime import datetime, timezone
import math

from const import PREDICT_STEP, PREDBAT_MAX_CARS
from prediction import Prediction


class UnsupportedJointSite(ValueError):
    """A native configuration cannot be represented by this single-EV prototype."""


def _number(value, name, minimum=0, maximum=None):
    """Reject coercion, booleans and non-finite configuration/forecast values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError("Invalid joint-site numeric field: " + name)
    return float(value)


def _stamp(value, name):
    """Require an offset-aware timestamp, never guessing a timezone or next day."""
    if not isinstance(value, str):
        raise ValueError("Joint-site timestamp required: " + name)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("Invalid joint-site timestamp: " + name) from None
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Joint-site timestamp needs an offset: " + name)
    return result.astimezone(timezone.utc)


def _options(base):
    """Read a dated expected session without treating it as permission to switch."""
    settings = base.args.get("joint_site")
    if settings is None or settings == {}:
        return None
    if not isinstance(settings, dict):
        raise ValueError("joint_site must be a mapping")
    allowed = {"mode", "session", "load_port_limit_w", "ev_pause_percent", "ev_restart_percent", "min_off_minutes", "beam_width", "terminal_min_percent", "soc_max_age_minutes"}
    if set(settings) - allowed:
        raise ValueError("Unknown joint_site option")
    mode = settings.get("mode", "off")
    if mode == "off":
        return None
    if mode != "shadow":
        raise ValueError("joint_site supports only off or shadow; execution is not implemented")
    _number(base.soc_max, "soc_max", 0.001)
    _number(base.reserve, "reserve", 0, base.soc_max)
    session = settings.get("session")
    if not isinstance(session, dict) or set(session) != {"available_from", "departure", "soc_kwh", "soc_observed_at"}:
        raise ValueError("A complete dated joint_site session is required")
    now = base.now_utc.astimezone(timezone.utc)
    observed = _stamp(session["soc_observed_at"], "soc_observed_at")
    max_age = _number(settings.get("soc_max_age_minutes", 60), "soc_max_age_minutes", 1, 1440)
    age = (now - observed).total_seconds() / 60
    if age < 0 or age > max_age:
        raise ValueError("EV SoC observation is stale or future-dated")
    arrival = _stamp(session["available_from"], "available_from")
    departure = _stamp(session["departure"], "departure")
    if departure <= now or arrival >= departure or (departure - now).total_seconds() > base.forecast_minutes * 60:
        raise ValueError("Expected EV session is outside the forecast/departure window")
    # Availability is rounded inward: never charge early or beyond departure.
    start_offset = max(math.ceil((arrival - now).total_seconds() / (60 * PREDICT_STEP)) * PREDICT_STEP, 0)
    end_offset = math.floor((departure - now).total_seconds() / (60 * PREDICT_STEP)) * PREDICT_STEP
    pause = _number(settings.get("ev_pause_percent", 15), "ev_pause_percent", 0, 100)
    restart = _number(settings.get("ev_restart_percent", 25), "ev_restart_percent", pause, 100)
    off_minutes = _number(settings.get("min_off_minutes", 10), "min_off_minutes", PREDICT_STEP, 60)
    beam = _number(settings.get("beam_width", 128), "beam_width", 8, 512)
    if beam != int(beam) or off_minutes != int(off_minutes):
        raise ValueError("Beam width and cooldown must be integers")
    return {
        "arrival": start_offset,
        "deadline": end_offset,
        "ev_soc": _number(session["soc_kwh"], "soc_kwh", 0, base.car_charging_battery_size[0]),
        "load_cap": _number(settings.get("load_port_limit_w"), "load_port_limit_w", 1) / 1000,
        "pause": pause,
        "restart": restart,
        "off_minutes": int(off_minutes),
        "beam": int(beam),
        "terminal_min": _number(settings.get("terminal_min_percent", base.reserve / base.soc_max * 100), "terminal_min_percent", 0, 100) * base.soc_max / 100,
    }


def _forecast_inputs(base, options):
    """Require a complete series and explicit topology; never fill missing load/PV with zero."""
    if base.num_cars != 1 or base.num_inverters != 1 or base.iboost_enable or not base.car_energy_reported_load:
        raise UnsupportedJointSite("Shadow planner currently supports one metered EV, one inverter and no iBoost")
    if not base.set_charge_window or not base.set_charge_freeze or base.set_charge_freeze_only or base.set_charge_low_power:
        raise UnsupportedJointSite("Full-rate native charge and freeze controls are required")
    if not base.car_charging_from_battery:
        raise UnsupportedJointSite("Native car_charging_from_battery is disabled")
    if any(getattr(base, name, None) for name in ("manual_charge_times", "manual_demand_times", "manual_export_times", "manual_freeze_charge_times", "manual_freeze_export_times", "all_active_keep", "all_active_keep_max", "io_adjusted")):
        raise UnsupportedJointSite("Manual/event overrides are not modelled by this prototype")
    if type(base.forecast_minutes) is not int or not 0 < base.forecast_minutes <= 2880 or base.forecast_minutes % PREDICT_STEP:
        raise ValueError("Joint-site horizon must be whole five-minute steps, at most 48 hours")
    _number(base.import_limit, "import_limit", 1)
    _number(base.soc_max, "soc_max", 0.001)
    _number(base.soc_kw, "soc_kw", base.reserve, base.soc_max)
    _number(base.car_charging_limit[0], "car_charging_limit", 0, base.car_charging_battery_size[0])
    _number(base.car_charging_rate[0], "car_charging_rate", 0.001)
    for name in ("inverter_loss", "battery_loss", "battery_loss_discharge", "car_charging_loss"):
        _number(getattr(base, name), name, 0.001, 1)
    for name in ("inverter_limit", "battery_rate_max_charge", "battery_rate_max_charge_dc", "battery_rate_max_discharge", "export_limit", "battery_rate_max_scaling", "battery_rate_max_scaling_discharge"):
        _number(getattr(base, name), name)
    if options["pause"] < base.reserve / base.soc_max * 100:
        raise ValueError("EV pause threshold must not be below the domestic reserve")
    for offset in range(0, base.forecast_minutes, PREDICT_STEP):
        for name in ("load_minutes_step", "load_minutes_step10", "pv_forecast_minute_step", "pv_forecast_minute10_step"):
            _number(getattr(base, name).get(offset), name)
        for minute in range(base.minutes_now + offset, base.minutes_now + offset + PREDICT_STEP):
            for name in ("rate_import", "rate_export"):
                _number(getattr(base, name).get(minute), name, -1000000, 1000000)


def build_candidate(base, options):
    """Search coupled fixed-rate EV and battery decisions using native forecast inputs."""
    # Imported lazily: a disabled installation does not load the experimental solver.
    from joint_site import BatteryConfig, EVConfig, PlannerConfig, SiteConfig, Slot, plan

    slots = []
    for index, minute in enumerate(range(0, base.forecast_minutes, PREDICT_STEP)):
        absolute = base.minutes_now + minute
        import_rate = sum(base.rate_import[m] for m in range(absolute, absolute + PREDICT_STEP)) / PREDICT_STEP
        export_rate = sum(base.rate_export[m] for m in range(absolute, absolute + PREDICT_STEP)) / PREDICT_STEP
        pv_ac = base.pv_forecast_minute_step[minute] * (base.inverter_loss if base.inverter_hybrid else 1)
        if not base.inverter_hybrid and base.pv_ac_limit > 0:
            pv_ac = min(pv_ac, base.pv_ac_limit * PREDICT_STEP)
        slots.append(Slot(index, minute, PREDICT_STEP, base.load_minutes_step[minute], pv_ac, import_rate, export_rate))
    battery = BatteryConfig(
        capacity_kwh=base.soc_max,
        initial_kwh=base.soc_kw,
        reserve_kwh=base.reserve,
        max_charge_kw=min(base.battery_rate_max_charge, base.inverter_limit) * 60 * base.battery_rate_max_scaling / base.inverter_loss,
        max_discharge_kw=min(base.battery_rate_max_discharge, base.inverter_limit) * 60 * base.battery_rate_max_scaling_discharge * base.inverter_loss,
        charge_efficiency=base.battery_loss * base.inverter_loss,
        discharge_efficiency=base.battery_loss_discharge * base.inverter_loss,
        end_min_kwh=max(options["terminal_min"], base.reserve),
        cycle_cost_per_kwh=_number(base.metric_battery_cycle, "metric_battery_cycle"),
        terminal_value_per_kwh=_number(base.battery_value_rate(base.minutes_now + base.forecast_minutes) * base.metric_battery_value_scaling, "terminal_value"),
    )
    ev = EVConfig(
        battery_kwh=base.car_charging_battery_size[0],
        start_soc_kwh=options["ev_soc"],
        stop_soc_kwh=base.car_charging_limit[0],
        rate_kw=base.car_charging_rate[0],
        efficiency=base.car_charging_loss,
        deadline_min=options["deadline"],
        availability=tuple(s.index for s in slots if options["arrival"] <= s.start_min and s.start_min + s.duration_min <= options["deadline"]),
        pause_battery_percent=options["pause"],
        restart_battery_percent=options["restart"],
        min_off_minutes=options["off_minutes"],
        initial_off_minutes=options["off_minutes"],
    )
    site = SiteConfig(base.import_limit / 1000, base.export_limit * 60, options["load_cap"])
    result = plan(slots, ev, battery, site, PlannerConfig(beam_width=options["beam"]))
    return result, slots


def _native_windows(base, result, slots):
    """Translate modes to native charge/freeze windows without altering executable lists."""
    if len(result.actions) != len(slots) or any(action.slot_index != slot.index for action, slot in zip(result.actions, slots)):
        raise ValueError("Joint candidate must contain one action for every native forecast step")
    ev_windows = []
    charge_windows = []
    limits = []
    for action, slot in zip(result.actions, slots):
        start = base.minutes_now + slot.start_min
        end = start + slot.duration_min
        if action.ev_action == "on":
            # Reserve full power even in the final partly-filled EV step. Native's usual
            # target clamp must not disguise a fixed charger's instantaneous demand.
            energy = base.car_charging_rate[0] * slot.duration_min / 60
            ev_windows.append({"start": start, "end": end, "kwh": energy, "average": slot.import_price, "cost": energy * slot.import_price, "octopus": False})
        if action.battery_mode != "demand":
            target = base.soc_max if action.battery_mode == "charge" else base.reserve
            if charge_windows and charge_windows[-1]["end"] == start and limits[-1] == target:
                charge_windows[-1]["end"] = end
            else:
                charge_windows.append({"start": start, "end": end, "average": slot.import_price})
                limits.append(target)
    return ev_windows, charge_windows, limits


def _replay(base, options, result, slots, windows, scenario):
    """Independently check raw native import, reserve and EV hysteresis using a new Prediction."""
    ev_windows, charge_windows, limits = windows
    model = copy(base)
    model.car_charging_soc = [options["ev_soc"]] * PREDBAT_MAX_CARS
    model.car_charging_soc_next = [None] * PREDBAT_MAX_CARS
    full_slot_target = options["ev_soc"] + sum(w["kwh"] for w in ev_windows) * base.car_charging_loss + 1
    model.car_charging_limit_model = [full_slot_target] * PREDBAT_MAX_CARS
    model.car_charging_slots = [ev_windows] + [[] for _ in range(PREDBAT_MAX_CARS - 1)]
    # Saved Python runs expose raw import diagnostics; avoid allocating an unused
    # kernel context. The active native kernel/cache remains untouched.
    model.prediction_kernel_enable = False
    model.debug_enable = False
    model.metric_standing_charge = 0
    pred = Prediction(model, base.pv_forecast_minute_step, base.pv_forecast_minute10_step, base.load_minutes_step, base.load_minutes_step10)
    pred.run_prediction(limits, charge_windows, [], [], scenario, end_record=base.forecast_minutes, save="test")
    diagnostic = pred.import_limit_diagnostics
    reasons = []
    if not diagnostic or diagnostic["checked_minutes"] != base.forecast_minutes:
        reasons.append("incomplete_native_prediction")
    elif diagnostic["minutes_over"]:
        reasons.append("site_import_exceeded")
    trace = pred.predict_soc
    reserve_min = min(list(trace.values()) + [pred.final_soc])
    if reserve_min < base.reserve - 0.002:
        reasons.append("domestic_reserve")
    if pred.final_soc < max(options["terminal_min"], base.reserve) - 0.002:
        reasons.append("terminal_reserve")
    previous_on = False
    off_minutes = options["off_minutes"]
    for action, slot in zip(result.actions, slots):
        on = action.ev_action == "on"
        start_soc = trace[slot.start_min]
        end_soc = trace.get(slot.start_min + slot.duration_min, pred.final_soc)
        if on and min(start_soc, end_soc) <= options["pause"] * base.soc_max / 100 + 0.002:
            reasons.append("ev_pause_threshold")
        if on and not previous_on and start_soc < options["restart"] * base.soc_max / 100 + 0.002:
            reasons.append("ev_restart_threshold")
        if on and (slot.start_min < options["arrival"] or slot.start_min + slot.duration_min > options["deadline"]):
            reasons.append("ev_availability_or_deadline")
        if on and not previous_on and off_minutes < options["off_minutes"]:
            reasons.append("ev_cooldown")
        scenario_load = base.load_minutes_step10 if scenario == 1 else base.load_minutes_step
        load_kw = scenario_load[slot.start_min] * 60 / slot.duration_min + (base.car_charging_rate[0] if on else 0)
        if load_kw > options["load_cap"] + 1e-9:
            reasons.append("load_port_exceeded")
        off_minutes = 0 if on else off_minutes + slot.duration_min
        previous_on = on
    if pred.final_car_soc[0] < base.car_charging_limit[0] - 0.002:
        reasons.append("ev_target_not_met")
    return {
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "checked_minutes": diagnostic["checked_minutes"] if diagnostic else 0,
        "peak_import_w": diagnostic["peak_w"] if diagnostic else None,
        "final_battery_kwh": pred.final_soc,
        "minimum_battery_kwh": reserve_min,
        "horizon_energy_cost": pred.final_metric - base.cost_today_sofar,
        "model": "native_python_full_power_ev_envelope",
    }


def _empty_report(status):
    """Build fresh, deliberately non-executable output; no stale schedule survives a failed run."""
    return {"status": status, "execution_enabled": False, "enforced": False, "actuator_signals_modified": False, "proposed_ev_windows": [], "proposed_charge_windows": [], "proposed_charge_limits": [], "native_checks": {}}


def _log_failure(base, stage, error):
    """Identify programmer/transport failures without echoing private exception values."""
    try:
        base.log("Warn: Joint shadow planner failed during {} ({}); executable plan was not modified".format(stage, type(error).__name__))
    except Exception:
        pass


def _publish_report(base, report, publish):
    """A shadow sensor or logger failure must not abort the native control cycle."""
    base.joint_site_report = report
    if publish:
        try:
            base.dashboard_item(base.prefix + ".joint_site_plan", state=report["status"], attributes=report)
        except Exception as exc:
            report["publication_failed"] = True
            _log_failure(base, "publication", exc)


def evaluate_joint_site(base, publish=True):
    """Evaluate and optionally publish a shadow candidate, never the actuator-facing EV sensor."""
    report = _empty_report("invalid_input")
    stage = "configuration"
    try:
        options = _options(base)
        if options is None:
            if getattr(base, "joint_site_report", None):
                report = _empty_report("disabled")
                _publish_report(base, report, publish)
            return None
        stage = "forecast"
        _forecast_inputs(base, options)
        stage = "solver"
        result, slots = build_candidate(base, options)
        if not result.actions:
            report["status"] = "no_plan_found"
        else:
            stage = "native_replay"
            windows = _native_windows(base, result, slots)
            checks = {"nominal": _replay(base, options, result, slots, windows, 0), "pv10": _replay(base, options, result, slots, windows, 1)}
            report.update(
                status="candidate" if all(check["passed"] for check in checks.values()) else "native_rejected",
                proposed_ev_windows=windows[0],
                proposed_charge_windows=windows[1],
                proposed_charge_limits=windows[2],
                native_checks=checks,
                solver_objective=result.total_cost,
                solver_grid_cost=sum(action.grid_import_kwh * action.import_price - action.grid_export_kwh * action.export_price for action in result.actions),
                cost_unit=base.currency_symbols[1],
                cost_convention="solver_objective_includes_cycle_cost_and_terminal_credit; native_horizon_energy_cost_excludes_both_and_standing_charge",
                ev_window_convention="full_step_power_reservation_not_partial_target_energy",
                initial_execution_state_assumption="off_for_minimum_cooldown_not_connection_permission",
                solver_ev_ac_kwh=result.diagnostics.ev_delivered_total_kwh,
                bounded_search=True,
                execution_blockers=["window_executor_mapping_not_validated", "actual_connection_gate_not_implemented", "hardware_topology_not_verified"],
            )
    except UnsupportedJointSite as exc:
        report = _empty_report("unsupported")
        report["reason"] = str(exc)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        input_stage = stage in ("configuration", "forecast")
        report = _empty_report("invalid_input" if input_stage else "error")
        report["reason"] = "Check the dated session, supported settings and complete finite native forecasts" if input_stage else "Shadow calculation failed; executable native plan was not modified"
        if not input_stage:
            report["failure_stage"] = stage
            report["error_type"] = type(exc).__name__
            _log_failure(base, stage, exc)
    except Exception as exc:
        # A broken experimental solver must not disrupt the established native
        # planner, leak raw sensor/config values or publish an old successful plan.
        report = _empty_report("error")
        report["reason"] = "Unexpected shadow-planner failure; executable native plan was not modified"
        report["failure_stage"] = stage
        report["error_type"] = type(exc).__name__
        _log_failure(base, stage, exc)
    _publish_report(base, report, publish)
    return report
