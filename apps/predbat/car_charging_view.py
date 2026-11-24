"""Read-only HTML presentation of the native EV/battery plan and candidate schedule.

This module projects a validated combined candidate into the existing web Plan
table without changing the native executable schedule:

* building an independent, battery-window aligned replay preview from the native
  traces (``base.predict_soc``, ``base.predict_grid_power``) and the solver
  ``actions`` so the candidate is shown row-by-row, not as a single
  averaging table cell;
* rendering a server-side safe ownership banner, not a second table, and
  adapting rows for the existing renderer without editable simulation controls;
* when the parent executor publishes a fresh, unexpired, paired-checked
  ``status='applied'`` receipt, surfacing the actual corrected plan
  (paired rows, "applied" mode, no "Simulation only — not executed" banner)
  instead of the simulated candidate. A stale / missing / future / false
  authority receipt clears the rows and surfaces a compact
  execution-blocked reason without ever falling back to the unrelated
  native plan and without claiming success. A narrowly validated domestic
  power-readiness observation may have no chart preview; it remains observed
  but unavailable, with no fabricated trajectory or hardware-confirmation claim.

No inputs are mutated; the module never calls a HA service, never writes to
the inverter and never acquires the executable plan.
"""

from datetime import datetime, timedelta, timezone
from html import escape
import math

from const import TIME_FORMAT

# A complete 48-hour horizon at the smallest supported five-minute window.
PREVIEW_ROW_LIMIT = 576
_NO_CANDIDATE_STATUSES = frozenset(
    {
        "disabled",
        "invalid_input",
        "unsupported",
        "no_plan_found",
        "conditional_no_plan_found",
        "error",
        "departure_passed",
        "native_rejected",
    }
)
_EXCEEDED_STATES = ("exceeded", "over", "true", "on", "yes", "1")
_OK_STATES = ("ok", "within", "false", "off", "no", "0")
_STALE_AGE_MINUTES = 15
# Domestic no-preview receipts are only eligible within the view's explicit
# 8–96 hour planning envelope. The public report omits ``forecast_hours``, so
# bound the paired replay evidence to that same conservative display ceiling.
_DOMESTIC_READINESS_MAX_HORIZON_HOURS = 96
# Receipt coherence window: matches the executor's RECEIPT_TTL_SECONDS
# default of 300 seconds. Both ends of the interval are strict so a
# future-dated receipt or a now == observed timestamp is rejected.
_RECEIPT_COHERENCE_WINDOW_SECONDS = 300
# Known replay provenance, not evidence of safety or execution authority.
_REPLAY_DISPLAY_STATUSES = frozenset({"feasible", "power_readiness_service", "ordinary_native_joint_reference", "generated_low_rates_native_reference", "ordinary_native_household_plan"})
# Bounded executor-failure labels; do not echo arbitrary internal strings.
_ALLOWED_BLOCKED_REASONS = frozenset(
    {
        "apply_commissioning_required",
        "local_deye_required",
        "local_deye_adapter_missing",
        "deye_commissioning_required",
        "inverter_capability_or_calibration",
        "read_only",
        "native_inputs_or_overrides_blocked",
        "native_state_not_ready",
        "deye_dispatch_model_mismatch",
        "manual_ceiling_unsupported",
        "entity_observation_unavailable",
        "permission_not_granted",
        "connection_not_confirmed",
        "physical_observation_stale_or_future",
        "entity_timestamp_future",
        "ev_cooldown",
        "relay_state_invalid",
        "import_limit_mismatch",
        "domestic_manual_floor",
        "discharge_power_out_of_range",
        "invalid_target",
        "invalid_hold_target",
        "invalid_command_boolean",
        "command_timeline_mismatch",
        "command_expired_during_dispatch",
        "command_authority_expired",
        "no_pending_candidate",
        "candidate_not_accepted",
        "candidate_stale_or_future",
        "configuration_changed",
        "configuration_changed_during_dispatch",
        "incomplete_command_horizon",
        "inverter_command_not_observed",
        "native_checks_failed",
        "execution_validation_failed",
        "publication_failed",
    }
)


def _is_finite(value):
    """Truthy only for a real, finite number; reject booleans, NaN and inf."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _format_iso(base, absolute_minute):
    """Use the same offset-aware absolute clock as the native prediction."""
    midnight = getattr(base, "midnight_utc", None)
    if midnight is None:
        return ""
    return (midnight + timedelta(minutes=absolute_minute)).strftime(TIME_FORMAT)


def _parse_iso(value):
    """Best-effort parse of an offset-aware ISO timestamp; never raise."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else None


def _safe_iso(value):
    """Return an unescaped ISO timestamp only when it parses with timezone."""
    return escape(value) if _parse_iso(value) is not None else ""


def _age_minutes(timestamp, now):
    """Return the age in minutes, or None when the timestamp is missing/invalid."""
    parsed = _parse_iso(timestamp)
    if parsed is None or not isinstance(now, datetime) or now.tzinfo is None:
        return None
    try:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return (now - parsed).total_seconds() / 60.0
    except (TypeError, AttributeError):
        return None


def _trace_complete(pred, base, start_relative, duration_min):
    """Reject any replay that lacks a full native horizon trace; never fill zeros."""
    if pred is None:
        return False
    grid = getattr(pred, "predict_grid_power", None)
    soc = getattr(pred, "predict_soc", None)
    if not isinstance(grid, dict) or not isinstance(soc, dict):
        return False
    for minute in range(start_relative, start_relative + duration_min, 5):
        if not _is_finite(grid.get(_format_iso(base, base.minutes_now + minute))):
            return False
        if not _is_finite(soc.get(minute)):
            return False
    endpoint = start_relative + duration_min
    value = getattr(pred, "final_soc", None) if endpoint == base.forecast_minutes else soc.get(endpoint)
    if not _is_finite(value):
        return False
    return True


def _bin_modes(actions_by_start, slice_starts):
    """Collect the actual, ordered unique battery modes from the bin's actions."""
    seen = []
    for slice_start in slice_starts:
        mode = getattr(actions_by_start.get(slice_start), "battery_mode", None)
        if isinstance(mode, str) and mode and mode not in seen:
            seen.append(mode)
    return ",".join(seen)


def build_replay_preview(base, pred, result, slots, scenario):
    """Build configured-window replay rows with a single battery intent per row.

    Returns a list of at most ``PREVIEW_ROW_LIMIT`` dictionaries, one per
    configured window that overlaps the planning horizon, additionally split
    at mode, target, EV and tariff changes. Each row uses the
    absolute ``base.minutes_now + slot.start_min`` clock, never a relative
    index, and reports truthful first/last partial-bin start/end ISO
    timestamps derived from ``base.midnight_utc``. Rejects incomplete or
    non-finite traces rather than zero-filling them. Does not mutate any
    input.
    """
    if not isinstance(slots, (list, tuple)) or not slots:
        return []
    # Reference alternatives use provenance labels rather than the solver's
    # "feasible" label. They still need the exact same complete native trace;
    # paired safety checks and execution authority remain separate view gates.
    status = getattr(result, "status", None)
    if not isinstance(status, str) or status not in _REPLAY_DISPLAY_STATUSES:
        return []
    actions = tuple(getattr(result, "actions", ()) or ())
    if len(actions) != len(slots):
        return []
    midnight = getattr(base, "midnight_utc", None)
    soc_max = getattr(base, "soc_max", None)
    if midnight is None or not _is_finite(soc_max) or soc_max <= 0:
        return []
    final_soc = getattr(pred, "final_soc", None)
    rates = getattr(base, "car_charging_rate", [])
    rate_kw = rates[0] if rates else None
    if not _is_finite(final_soc) or not _is_finite(rate_kw) or rate_kw <= 0:
        return []
    minutes_now = getattr(base, "minutes_now", None)
    horizon = getattr(base, "forecast_minutes", None)
    if not isinstance(minutes_now, int) or minutes_now % 5 or not isinstance(horizon, int) or horizon <= 0 or horizon % 5:
        return []
    if len(slots) * 5 != horizon or getattr(midnight, "tzinfo", None) is None:
        return []
    if not _trace_complete(pred, base, 0, horizon):
        return []
    load_map = getattr(base, "load_minutes_step10" if scenario == 1 else "load_minutes_step", {})
    pv_map = getattr(base, "pv_forecast_minute10_step" if scenario == 1 else "pv_forecast_minute_step", {})
    row_minutes = getattr(base, "plan_preview_row_minutes", 30)
    if type(row_minutes) is not int or row_minutes not in (5, 15, 30, 60):
        return []
    plan_hours = getattr(base, "forecast_plan_hours", None)
    if plan_hours is not None and (not _is_finite(plan_hours) or not 8 <= plan_hours <= 96 or plan_hours % 1):
        return []
    display_minutes = plan_hours * 60 if plan_hours is not None else horizon
    rows = []
    previous_key = None
    for position, (action, slot) in enumerate(zip(actions, slots)):
        minute = position * 5
        if slot.start_min != minute or slot.duration_min != 5 or action.slot_index != slot.index:
            return []
        if action.ev_action not in ("on", "off") or action.battery_mode not in ("demand", "hold", "charge"):
            return []
        load = load_map.get(minute)
        pv = pv_map.get(minute)
        if not _is_finite(load) or not _is_finite(pv) or load < 0 or pv < 0:
            return []
        absolute = minutes_now + minute
        target = None
        if action.battery_mode == "charge":
            target = getattr(base, "plan_preview_charge_limits", {}).get(absolute, 100)
            if not _is_finite(target) or not 0 < target <= 100.1:
                return []
        start_soc = pred.predict_soc[minute] * 100 / soc_max
        end_soc = (final_soc if minute + 5 == horizon else pred.predict_soc[minute + 5]) * 100 / soc_max
        peak = max(0.0, pred.predict_grid_power[_format_iso(base, absolute)] * 1000 + 0.5)
        key = (absolute // row_minutes, minute >= display_minutes, action.battery_mode, target, action.ev_action, getattr(slot, "import_price", None), getattr(slot, "export_price", None))
        if key != previous_key:
            rows.append(
                {
                    "start": _format_iso(base, absolute),
                    "end": "",
                    "minutes": 0,
                    "battery_modes": [],
                    "ev_on_minutes": 0,
                    "ev_reserved_kwh": 0.0,
                    "house_kwh": 0.0,
                    "pv_kwh": 0.0,
                    "battery_start_percent": start_soc,
                    "battery_end_percent": end_soc,
                    "battery_min_percent": min(start_soc, end_soc),
                    "grid_peak_w": 0.0,
                }
            )
            previous_key = key
        row = rows[-1]
        row["end"] = _format_iso(base, absolute + 5)
        row["minutes"] += 5
        if action.battery_mode not in row["battery_modes"]:
            row["battery_modes"].append(action.battery_mode)
        if action.ev_action == "on":
            row["ev_on_minutes"] += 5
            row["ev_reserved_kwh"] += rate_kw * 5 / 60
        if action.battery_mode == "charge":
            row["charge_target_percent"] = max(row.get("charge_target_percent", 0), target)
        # Native forecast maps contain kWh per step, not average kW.
        row["house_kwh"] += load
        row["pv_kwh"] += pv
        row["battery_end_percent"] = end_soc
        row["battery_min_percent"] = min(row["battery_min_percent"], start_soc, end_soc)
        row["grid_peak_w"] = max(row["grid_peak_w"], peak)
        # Additive display evidence only: keep the validated replay bins and
        # every safety/authority input unchanged. Battery power is positive
        # for discharge, negative for charge, as in the native prediction.
        battery_kw = getattr(pred, "predict_battery_power", {}).get(_format_iso(base, absolute))
        row.setdefault("display_steps", []).append(
            {
                "start": _format_iso(base, absolute),
                "end": _format_iso(base, absolute + 5),
                "battery_kw": battery_kw if _is_finite(battery_kw) else None,
                "grid_kw": pred.predict_grid_power[_format_iso(base, absolute)],
                "soc_start": start_soc,
                "soc_end": end_soc,
            }
        )
        # Prices and cumulative cost must belong to this replay too. Never
        # borrow the native plan's costs when displaying a different schedule.
        if _is_finite(getattr(slot, "import_price", None)) and _is_finite(getattr(slot, "export_price", None)):
            row["import_rate"] = row.get("import_rate", 0) + slot.import_price * 5
            row["export_rate"] = row.get("export_rate", 0) + slot.export_price * 5
        metrics = getattr(pred, "metric_time", {})
        cost_start = metrics.get(_format_iso(base, absolute))
        cost_end = getattr(pred, "final_metric", None) if minute + 5 == horizon else metrics.get(_format_iso(base, absolute + 5))
        if "total_cost" not in row and _is_finite(cost_start):
            row["total_cost"] = cost_start / 100
        if _is_finite(cost_end) and _is_finite(row.get("total_cost")):
            row["cost_change"] = cost_end / 100 - row["total_cost"]
        durations = row.setdefault("mode_minutes", {})
        durations[action.battery_mode] = durations.get(action.battery_mode, 0) + 5
    if len(rows) > PREVIEW_ROW_LIMIT:
        return []
    for row in rows:
        row["battery_modes"] = ",".join(row["battery_modes"])
        for key in ("import_rate", "export_rate"):
            if key in row:
                row[key] /= row["minutes"]
    return rows


def _applied_receipt_fresh(report, now):
    """Accept only fresh, coherent applied execution receipts.

    The parent ``car_charging_execute._report`` populates the report with:

    * ``status='applied'``,
    * ``requested_mode='apply'``,
    * ``execution_enabled=True``,
    * ``actuator_signals_modified=True``,
    * ``execution_observed_at`` (ISO) — the instant the executor observed
      the inverter accept the command,
    * ``execution_valid_until`` (ISO) — the receipt expiry.

    Coherence requires ``now >= observed_at``, ``now < valid_until`` and an
    interval ``(valid_until - observed_at) <= 300s``. Anything else — a
    missing field, a future stamp, a TTL that has run out, a stale receipt,
    ``execution_enabled`` not exactly ``True``,
    ``actuator_signals_modified`` not exactly ``True``,
    ``requested_mode`` not exactly ``'apply'``, a non-applied status — is
    treated as a non-authoritative receipt and yields ``False``. The view
    never falls back to the unrelated native plan and never claims success.
    """
    if not isinstance(report, dict):
        return False
    if report.get("status") != "applied":
        return False
    if report.get("requested_mode") != "apply":
        return False
    if report.get("execution_enabled") is not True:
        return False
    if report.get("actuator_signals_modified") is not True:
        return False
    if "execution_blockers" in report:
        blockers = report.get("execution_blockers")
        # Fail closed: only an empty list (or absent key) is acceptable.
        # Strings, dicts, True/1, or non-empty lists all reject.
        if not isinstance(blockers, list) or blockers:
            return False
    observed = _parse_iso(report.get("execution_observed_at"))
    valid_until = _parse_iso(report.get("execution_valid_until"))
    if observed is None or valid_until is None:
        return False
    if not isinstance(now, datetime) or now.tzinfo is None:
        return False
    interval_seconds = (valid_until - observed).total_seconds()
    if interval_seconds <= 0 or interval_seconds > _RECEIPT_COHERENCE_WINDOW_SECONDS:
        return False
    if now < observed or now >= valid_until:
        return False
    return True


def _applied_display_evidence_valid(report, now):
    """Validate applied authority and timestamps independently of chart rows."""
    if not isinstance(report, dict):
        return False
    if not _applied_receipt_fresh(report, now):
        return False
    if report.get("uses_stale_soc_assumption") is True:
        return False
    if report.get("counterfactual_controls") is True:
        return False
    # Coherence between the report's ``generated_at`` and the receipt's
    # observed_at window. A future-dated or way-stale ``generated_at`` is
    # not a valid applied display even with a fresh receipt.
    generated_at = _parse_iso(report.get("generated_at"))
    observed = _parse_iso(report.get("execution_observed_at"))
    if generated_at is None or observed is None:
        return False
    if generated_at > observed or generated_at > now or now - generated_at >= timedelta(minutes=10):
        return False
    return True


def _applied_display_allowed(report, now, paired):
    """Require both a valid applied receipt and paired rows for the active chart."""
    return paired is not None and _applied_display_evidence_valid(report, now)


def _domestic_readiness_without_preview_allowed(report, now):
    """Recognise only fresh, paired-safe domestic execution with no EV chart."""
    if not isinstance(report, dict) or report.get("status") != "applied":
        return False
    if report.get("preview") not in (None, {}):
        return False
    if not _applied_display_evidence_valid(report, now):
        return False
    if (
        report.get("domestic_only") is not True
        or report.get("power_readiness_service") is not True
        or report.get("household_executor") != "grid_charge_support"
        or report.get("ev_available") is not False
        or report.get("ev_soc_not_required") is not True
        or report.get("proposed_ev_windows") != []
        or report.get("execution_blockers") != []
        or report.get("execution_reason") != "inverter_observed_not_hardware_confirmed"
        or report.get("hardware_confirmed") is not False
    ):
        return False
    readiness = report.get("power_readiness")
    checks = report.get("native_checks")
    if not isinstance(readiness, dict) or readiness.get("active") is not True:
        return False
    if not isinstance(checks, dict) or set(checks) != {"nominal", "pv10"}:
        return False
    plan_hours = report.get("forecast_plan_hours")
    if not _is_finite(plan_hours) or not 8 <= plan_hours <= _DOMESTIC_READINESS_MAX_HORIZON_HOURS or plan_hours % 1:
        return False
    expected_minutes = int(plan_hours * 60)
    max_checked_minutes = _DOMESTIC_READINESS_MAX_HORIZON_HOURS * 60
    checked_minutes = []
    for scenario in ("nominal", "pv10"):
        check = checks.get(scenario)
        if not isinstance(check, dict) or check.get("passed") is not True:
            return False
        minutes = check.get("checked_minutes")
        if type(minutes) is not int or minutes <= 0:
            return False
        checked_minutes.append(minutes)
    return checked_minutes[0] == checked_minutes[1] and expected_minutes <= checked_minutes[0] <= max_checked_minutes


def _paired_preview(report, now):
    """Accept only complete, fresh and passed paired native trajectories."""
    if not isinstance(report, dict) or report.get("status") not in ("candidate", "conditional_candidate", "partial_candidate", "conditional_partial_candidate", "applied"):
        return None
    age = _age_minutes(report.get("generated_at"), now)
    if age is None or age < -1 or age > _STALE_AGE_MINUTES:
        return None
    checks = report.get("native_checks", {})
    if not all(isinstance(checks.get(key), dict) and checks[key].get("passed") is True for key in ("nominal", "pv10")):
        return None
    if report["status"] in ("partial_candidate", "conditional_partial_candidate"):
        required_target = ("requested_ev_target_kwh", "requested_ev_target_percent", "planning_ev_target_kwh", "planned_ev_kwh_by_departure", "planned_ev_percent_by_departure", "ev_target_shortfall_kwh")
        if not all(_is_finite(report.get(key)) for key in required_target):
            return None
        partial_mode = report.get("requested_mode") == "shadow" or (report.get("requested_mode") == "apply" and report.get("partial_target_authorised") is True)
        if not partial_mode or report.get("requested_ev_target_met") is not False or any(report.get(key) is not False for key in ("execution_enabled", "enforced", "actuator_signals_modified")):
            return None
        departure = _parse_iso(report.get("ev_departure_at"))
        if departure is None or departure <= now or not 0 <= report["planning_ev_target_kwh"] <= report["planned_ev_kwh_by_departure"] < report["requested_ev_target_kwh"]:
            return None
        if not 0 <= report["planned_ev_percent_by_departure"] < report["requested_ev_target_percent"] <= 100:
            return None
        if abs(report["planned_ev_percent_by_departure"] - report["planned_ev_kwh_by_departure"] * report["requested_ev_target_percent"] / report["requested_ev_target_kwh"]) > 0.01:
            return None
        if not all(_is_finite(checks[key].get("final_ev_kwh")) and checks[key]["final_ev_kwh"] >= report["planning_ev_target_kwh"] - 0.002 for key in ("nominal", "pv10")):
            return None
        if abs(min(checks[key]["final_ev_kwh"] for key in ("nominal", "pv10")) - report["planned_ev_kwh_by_departure"]) > 0.002:
            return None
        if abs(report["requested_ev_target_kwh"] - report["planned_ev_kwh_by_departure"] - report["ev_target_shortfall_kwh"]) > 0.002:
            return None
    preview = report.get("preview", {})
    if not isinstance(preview, dict):
        return None
    nominal, pv10 = preview.get("nominal"), preview.get("pv10")
    if not isinstance(nominal, list) or not isinstance(pv10, list) or not nominal or len(nominal) != len(pv10) or len(nominal) > PREVIEW_ROW_LIMIT:
        return None
    previous = None
    required = ("battery_start_percent", "battery_end_percent", "battery_min_percent", "grid_peak_w", "house_kwh", "pv_kwh", "ev_on_minutes", "ev_reserved_kwh", "minutes")
    for left, right in zip(nominal, pv10):
        for row in (left, right):
            if not isinstance(row, dict) or not all(_is_finite(row.get(key)) for key in required):
                return None
            target = row.get("charge_target_percent", 100)
            if not _is_finite(target) or not 0 < target <= 100.1:
                return None
            start, end = _parse_iso(row.get("start")), _parse_iso(row.get("end"))
            if start is None or end is None or not 0 < row["minutes"] <= 60 or (end - start).total_seconds() != row["minutes"] * 60:
                return None
            # Native SOC is rounded to 0.001 kWh, capacity is not. Retain its
            # genuine value (e.g. 100.001%) rather than rejecting a full battery.
            if not all(-0.1 <= row[key] <= 100.1 for key in ("battery_start_percent", "battery_end_percent", "battery_min_percent")):
                return None
            if any(row[key] < 0 for key in ("grid_peak_w", "house_kwh", "pv_kwh", "ev_reserved_kwh")) or not 0 <= row["ev_on_minutes"] <= row["minutes"]:
                return None
            if row.get("battery_modes") not in (
                "demand",
                "charge",
                "hold",
                "charge,demand",
                "demand,charge",
                "hold,demand",
                "demand,hold",
                "charge,hold",
                "hold,charge",
                "demand,charge,hold",
                "charge,demand,hold",
                "hold,charge,demand",
                "hold,demand,charge",
                "demand,hold,charge",
                "charge,hold,demand",
            ):
                return None
            modes = row["battery_modes"].split(",")
            if len(modes) > 1:
                mode_minutes = row.get("mode_minutes")
                if not isinstance(mode_minutes, dict) or set(mode_minutes) != set(modes):
                    return None
                durations = [mode_minutes.get(mode) for mode in modes]
                if not all(_is_finite(duration) and duration > 0 and duration % 5 == 0 for duration in durations):
                    return None
                if sum(durations) != row["minutes"]:
                    return None
        if (left["start"], left["end"]) != (right["start"], right["end"]) or (previous is not None and previous != _parse_iso(left["start"])):
            return None
        left_modes = left["battery_modes"].split(",")
        right_modes = right["battery_modes"].split(",")
        if left_modes != right_modes:
            return None
        if len(left_modes) > 1 or len(right_modes) > 1:
            if left["battery_modes"] != right["battery_modes"] or left["mode_minutes"] != right["mode_minutes"]:
                return None
        if abs(left.get("charge_target_percent", 100) - right.get("charge_target_percent", 100)) > 0.0001:
            return None
        previous = _parse_iso(left["end"])
    total_minutes = sum(row["minutes"] for row in nominal)
    if any(_is_finite(checks[key].get("checked_minutes")) and checks[key]["checked_minutes"] != total_minutes for key in ("nominal", "pv10")):
        return None
    return nominal, pv10


def build_car_charging_plan_view(native_plan, report, now=None):
    """Project the candidate into the existing web table schema, never executable state.

    No native row, cost, target or rowspan is inherited. Only display metadata
    survives. Invalid/expired active reports clear the candidate instead of
    quietly presenting the unrelated native plan as a combined trajectory.

    For a valid applied display (``status == 'applied'`` AND paired rows AND
    fresh receipt AND no stale-SoC/counterfactual hypothesis AND coherent
    ``generated_at``) the same paired preview rows are emitted, but
    ``view['plan_execution']`` is ``True`` and ``view['plan_simulation']``
    is ``False`` — this is the actual applied plan, not a simulation. When
    the applied display is invalid (expired / missing / future / false
    authority / blocked / invalid paired preview / stale ``generated_at``)
    the rows are cleared and ``view['execution_blocked']`` carries a bounded
    reason. A narrowly validated domestic power-readiness observation with
    no chart preview is marked unavailable without being called blocked.
    Parent ``status='apply_blocked'`` reports still land in the blocked path.
    """
    if not isinstance(report, dict) or report.get("status") in ("disabled", "household_fallback"):
        return native_plan
    now = now if now is not None else datetime.now(timezone.utc)
    native = native_plan if isinstance(native_plan, dict) else {}
    view = {key: native[key] for key in ("time", "currency_symbols", "reason_templates", "import_cost_threshold", "export_cost_threshold", "recorded_cost") if key in native}
    view.update({"timestamp": report.get("generated_at"), "num_cars": 0 if report.get("domestic_only") is True else 1, "iboost_enable": False, "carbon_enable": False, "rows": []})
    view.setdefault("currency_symbols", ["", ""])
    status = report.get("status")
    is_applied = status == "applied"
    is_blocked = status == "apply_blocked"
    paired = _paired_preview(report, now) if status not in (None, "disabled") else None
    domestic_without_preview = is_applied and paired is None and _domestic_readiness_without_preview_allowed(report, now)
    # Validate the complete paired replay before limiting the visible horizon.
    # The preview builder splits at this exact boundary, preserving real SOC,
    # energy and cost endpoints instead of interpolating a partial row.
    if paired is not None and "forecast_plan_hours" in report:
        plan_hours = report["forecast_plan_hours"]
        if not _is_finite(plan_hours) or not 8 <= plan_hours <= 96 or plan_hours % 1:
            paired = None
        else:
            cutoff = _parse_iso(paired[0][0]["start"]) + timedelta(hours=plan_hours)
            if any(_parse_iso(row["start"]) < cutoff < _parse_iso(row["end"]) for row in paired[0]):
                paired = None
            else:
                paired = tuple([row for row in rows if _parse_iso(row["end"]) <= cutoff] for rows in paired)
    applied_fresh = _applied_display_allowed(report, now, paired)
    if applied_fresh:
        view["plan_simulation"] = False
        view["plan_execution"] = True
        view["execution_observed_at"] = report.get("execution_observed_at")
        view["execution_valid_until"] = report.get("execution_valid_until")
    elif domestic_without_preview:
        # Preserve the fresh command observation, but never invent chart rows
        # from the native forecast or imply that hardware state was confirmed.
        view["plan_simulation"] = False
        view["plan_execution"] = True
        view["plan_unavailable"] = True
        view["num_cars"] = 0
        view["execution_observed_at"] = report.get("execution_observed_at")
        view["execution_valid_until"] = report.get("execution_valid_until")
        return view
    elif is_applied or is_blocked:
        # Applied or apply_blocked: surface the execution-blocked view with a
        # bounded reason. Never fall back to the native plan, never claim
        # success, never show the simulated candidate as a substitute.
        view["plan_simulation"] = False
        view["plan_execution"] = False
        view["execution_blocked"] = _applied_block_reason(report, now)
        return view
    else:
        # Shadow / candidate / partial / etc. — keep ``plan_simulation=True``
        # so the renderer treats it as the read-only simulated plan.
        view["plan_simulation"] = True
    if paired is None:
        # Non-applied status with no paired rows (e.g. shadow unavailable,
        # off, error). Preserve the original generic-replay behaviour.
        view["plan_unavailable"] = True
        return view
    for row, pv10 in zip(*paired):
        start = _parse_iso(row["start"])
        midnight = _parse_iso(view.get("time")) or start.replace(hour=0, minute=0, second=0, microsecond=0)
        names = row["battery_modes"].split(",")
        # Keep the established table's state labels and colour rules.
        change = row["battery_end_percent"] - row["battery_start_percent"]
        soc_sym = "&rarr;" if abs(change) < 0.15 else ("&nearr;" if change > 0 else "&searr;")
        if len(names) > 1:
            mode_labels = {"demand": "Demand", "hold": "HoldChrg", "charge": "Chrg"}
            mode_minutes = row["mode_minutes"]
            state = "Mixed"
            state_text = " + ".join("{} {}m".format(mode_labels[name], int(mode_minutes[name])) for name in names)
            reasons = []
            if "charge" in names:
                charge_limit = round(row.get("charge_target_percent", 100))
                show_limit = "Chrg ≤{}".format(charge_limit)
            else:
                show_limit = ""
        else:
            state = "Chrg" if "charge" in names else ("HoldChrg" if "hold" in names else "Demand")
            state_text = "Chrg&nearr;" if state == "Chrg" else ("HoldChrg&rarr;" if state == "HoldChrg" else soc_sym)
            reasons = []
            show_limit = str(round(row.get("charge_target_percent", 100))) if "charge" in names else ""
        steps = _display_steps(row)
        action = _display_action(row, steps)
        reason = _display_reason(action, names)
        if not steps or any(step.get("battery_kw") is None for step in steps):
            reason += " (net SoC; power trace unavailable)"
        rate = row.get("import_rate")
        threshold = native.get("import_cost_threshold")
        rate_color = "#FFFFFF"
        if _is_finite(rate) and _is_finite(threshold):
            rate_color = "#74C1FF" if rate <= 0 else ("#3AEE85" if rate <= threshold else ("#F18261" if rate > threshold * 1.5 else "#FFFFAA"))
        export_rate = row.get("export_rate")
        export_threshold = native.get("export_cost_threshold")
        export_color = "#FFFFFF"
        if _is_finite(export_rate) and _is_finite(export_threshold):
            export_color = "#F18261" if export_rate >= 1.5 * export_threshold else ("#FFFFAA" if export_rate >= export_threshold else ("#48deff" if export_rate > 0.5 * export_threshold else "#dcdcdc"))
        cost = row.get("cost_change")
        cost_color = "#FFFFFF"
        if _is_finite(cost):
            cost_color = "#F18261" if cost >= 0.1 else ("#FFFF00" if cost >= 0.005 else ("#3AEE85" if cost <= -0.005 else "#FFFFFF"))
        displayed_pv = round(row["pv_kwh"], 2)
        view["rows"].append(
            {
                "time": row["start"],
                "slot_minute": int((start - midnight).total_seconds() / 60),
                "import_rate": row.get("import_rate") if _is_finite(row.get("import_rate")) else 0,
                "export_rate": row.get("export_rate") if _is_finite(row.get("export_rate")) else 0,
                "plan_rates_unavailable": not (_is_finite(row.get("import_rate")) and _is_finite(row.get("export_rate"))),
                "rate_color_import": rate_color,
                "rate_color_export": export_color,
                "cost_color": cost_color,
                "state": state,
                "state_text": state_text,
                "plan_action": action,
                "plan_reason": reason,
                "plan_phase": "current" if start <= now < _parse_iso(row["end"]) else "forecast",
                "plan_steps": steps,
                "state_color": "#3AEE85" if state == "Chrg" else ("#34DBEB" if state == "HoldChrg" else "#FFFFFF"),
                "show_limit": show_limit,
                "reasons": reasons,
                "split": False,
                "pv_forecast": displayed_pv,
                "pv_forecast10": round(pv10["pv_kwh"], 2),
                "load_forecast": round(row["house_kwh"], 2),
                "load_forecast10": round(pv10["house_kwh"], 2),
                "clipped": 0,
                "car_charging": round(row["ev_reserved_kwh"], 3),
                "plan_ev_on_minutes": row["ev_on_minutes"],
                "soc_percent": row["battery_start_percent"],
                "soc_change": row["battery_end_percent"] - row["battery_start_percent"],
                "soc_sym": soc_sym,
                "soc_color": "#F18261" if row["battery_start_percent"] < 20 else ("#FFFF00" if row["battery_start_percent"] < 50 else "#3AEE85"),
                "load_color": "#F18261" if row["house_kwh"] >= 0.5 else ("#FFFF00" if row["house_kwh"] >= 0.25 else ("#AAFFAA" if row["house_kwh"] > 0 else "#FFFFFF")),
                "pv_color": "#FFAAAA" if displayed_pv >= 0.2 else ("#FFFF00" if displayed_pv >= 0.1 else "#FFFFFF"),
                "car_color": "#FFFF00" if row["ev_reserved_kwh"] > 0 else "#FFFFFF",
                "soc_end_percent": row["battery_end_percent"],
                "plan_soc_pv10_start_percent": pv10["battery_start_percent"],
                "plan_soc_pv10_end_percent": pv10["battery_end_percent"],
                "plan_grid_peak_w": row["grid_peak_w"],
                "plan_grid_peak_pv10_w": pv10["grid_peak_w"],
                "plan_interval_end": row["end"],
                "cost_change": row.get("cost_change"),
                "total_cost": row.get("total_cost"),
            }
        )
    last = paired[0][-1]
    if _is_finite(last.get("total_cost")) and _is_finite(last.get("cost_change")):
        view["totals"] = {
            "pv_forecast": round(sum(row["pv_kwh"] for row in paired[0]), 2),
            "load_forecast": round(sum(row["house_kwh"] for row in paired[0]), 2),
            "car_charging": round(sum(row["ev_reserved_kwh"] for row in paired[0]), 2),
            "soc_percent": round(last["battery_end_percent"]),
            "total_cost": round(last["total_cost"] + last["cost_change"], 2),
            "clipped": 0,
        }
    return view


def _display_steps(row):
    """Accept optional bounded flow detail, never as authority for the plan."""
    steps = row.get("display_steps")
    if not isinstance(steps, list) or len(steps) != row["minutes"] / 5:
        return []
    previous = _parse_iso(row["start"])
    for step in steps:
        if not isinstance(step, dict) or not all(_is_finite(step.get(key)) for key in ("grid_kw", "soc_start", "soc_end")):
            return []
        start, end = _parse_iso(step.get("start")), _parse_iso(step.get("end"))
        if start != previous or end is None or (end - start).total_seconds() != 300:
            return []
        if step.get("battery_kw") is not None and not _is_finite(step["battery_kw"]):
            return []
        previous = end
    return steps if previous == _parse_iso(row["end"]) else []


def _display_action(row, steps):
    """Describe forecast flow, not an enabled charge window or rounded SoC."""
    powers = [step.get("battery_kw") for step in steps]
    if powers and all(_is_finite(power) for power in powers):
        charging = any(power < -0.03 for power in powers)
        discharging = any(power > 0.03 for power in powers)
    else:
        change = row["battery_end_percent"] - row["battery_start_percent"]
        charging, discharging = change > 0.01, change < -0.01
    if charging and discharging:
        return "Mixed"
    if charging:
        return "Charge"
    if discharging:
        return "Discharge"
    return "Hold" if "charge" in row["battery_modes"] or "hold" in row["battery_modes"] else "Idle"


def _display_reason(action, modes):
    """Explain observable display semantics without inventing optimiser motives."""
    if action == "Hold":
        return "Charge window; no battery gain forecast" if "charge" in modes else "Battery held by plan"
    if action == "Charge":
        return "Charge towards target" if "charge" in modes else "Solar surplus stored"
    if action == "Discharge":
        return "Battery supplies demand / export"
    if action == "Mixed":
        return "Flow changes within interval; see detail"
    return "No material battery flow forecast"


def _format_finite(value, decimals=2):
    """Escape a finite numeric value; never render NaN, inf, bool or strings."""
    if not _is_finite(value):
        return ""
    return escape("{{:.{0}f}}".format(int(decimals)).format(float(value)))


def _parse_native_diag(native_diagnostic):
    """Return (state, peak_w, minutes_over, limit_w); None for missing/non-finite."""
    if not isinstance(native_diagnostic, dict):
        return None, None, None, None
    attributes = native_diagnostic.get("attributes")
    if not isinstance(attributes, dict):
        attributes = {}

    def _opt(key):
        """Accept only finite diagnostic quantities, never malformed state data."""
        value = attributes.get(key)
        return value if _is_finite(value) else None

    return native_diagnostic.get("state"), _opt("peak_w"), _opt("minutes_over"), _opt("limit_w")


def _is_exceeded(state, peak_w, minutes_over, limit_w):
    """True when the native diagnostic says the import limit was exceeded."""
    if isinstance(state, str):
        normalised = state.strip().lower()
        if normalised in _EXCEEDED_STATES:
            return True
        if normalised in _OK_STATES:
            return False
    if _is_finite(peak_w) and _is_finite(limit_w) and peak_w > limit_w:
        return True
    return False


def _applied_block_reason(report, now=None):
    """Return a bounded execution-blocked reason.

    Reasons are computed from the *actual* coherence checks (missing
    fields, wrong types, interval/valid_until problems, future/expired vs
    ``now``). The report's ``execution_reason`` is only echoed when its
    value is in a small allowlist of known parent labels; arbitrary
    strings are rejected and replaced with a generic tag. Never claims
    success; never echoes untrusted text.
    """
    if not isinstance(report, dict):
        return "no execution receipt"
    now = now if isinstance(now, datetime) and now.tzinfo is not None else None
    reason = report.get("execution_reason")
    if isinstance(reason, str) and reason in _ALLOWED_BLOCKED_REASONS:
        return reason
    blockers = report.get("execution_blockers")
    if "execution_blockers" in report and not (isinstance(blockers, list) and not blockers):
        return "execution blocked: declared blockers"
    observed = _parse_iso(report.get("execution_observed_at"))
    valid_until = _parse_iso(report.get("execution_valid_until"))
    if observed is None and valid_until is None:
        return "missing execution_observed_at and execution_valid_until"
    if observed is None:
        return "missing execution_observed_at"
    if valid_until is None:
        return "missing execution_valid_until"
    if valid_until <= observed:
        return "execution_valid_until not after execution_observed_at"
    interval = (valid_until - observed).total_seconds()
    if interval > _RECEIPT_COHERENCE_WINDOW_SECONDS:
        return "execution_valid_until exceeds {}s coherence window".format(_RECEIPT_COHERENCE_WINDOW_SECONDS)
    if now is not None and now.tzinfo is not None:
        if now < observed:
            return "execution_observed_at is in the future"
        if now >= valid_until:
            return "execution receipt expired"
    if report.get("uses_stale_soc_assumption") is True:
        return "unconfirmed EV SOC assumption"
    if report.get("counterfactual_controls") is True:
        return "counterfactual controls are not applied authority"
    generated = _parse_iso(report.get("generated_at"))
    if generated is None:
        return "missing generated_at"
    if generated > observed or (now is not None and generated > now):
        return "generated_at is in the future"
    if now is not None and now - generated >= timedelta(minutes=10):
        return "generated_at is stale"
    if _paired_preview(report, now) is None:
        return "paired replay preview missing or invalid"
    return "execution receipt not coherent"


def _unavailable_reason(report):
    """Explain rejection without exposing arbitrary diagnostic strings or stale rows."""
    labels = {
        "site_import_exceeded": "grid import limit",
        "ev_pause_threshold": "EV battery pause threshold",
        "ev_restart_threshold": "EV battery restart threshold",
        "domestic_reserve": "house battery reserve",
        "terminal_reserve": "end-of-plan battery reserve",
        "domestic_manual_floor": "manual house battery minimum",
        "ev_target_not_met": "EV departure target",
        "ev_cooldown": "EV restart delay",
        "load_port_exceeded": "house load-port limit",
    }
    rejected = []
    checks = report.get("native_checks", {})
    checks = checks if isinstance(checks, dict) else {}
    for scenario in ("nominal", "pv10"):
        check = checks.get(scenario, {})
        reasons = check.get("reasons", []) if isinstance(check, dict) else []
        reasons = reasons if isinstance(reasons, (list, tuple)) else []
        problems = [label for code, label in labels.items() if code in reasons]
        if problems:
            rejected.append(("conservative forecast" if scenario == "pv10" else "nominal forecast") + ": " + ", ".join(problems))
    if rejected:
        return "Latest candidate failed safety checks (" + "; ".join(rejected) + "). No validated replacement was found; the previous simulation is hidden."
    if report.get("status") in ("no_plan_found", "conditional_no_plan_found"):
        return "The bounded search did not find a complete validated schedule. This does not prove the departure target is impossible; the previous simulation is hidden."
    return "No complete paired native-replay preview is available. Rejected or unavailable candidates are not shown; no earlier schedule is reused."


def _household_fallback_debug(report):
    """Build the hidden Show-Debug markup for a household-fallback report.

    The branch must never project authority: the existing Plan table, columns,
    colours and precision stay unchanged. Only the debug span is augmented, and
    the arrival forecast (set by ``evaluate_car_charging`` when the recurring
    policy withdrew and the native executor handed back to household control)
    is rendered with full HTML escaping so Show Debug surfaces the
    low-confidence assumption, empirical range, sample count, source
    timestamps and shortfall that the live planning model just used.

    The function is intentionally small: it does not touch ``report``,
    ``native_diagnostic`` or any other shared state, and it never calls a
    HA service or writes to an inverter.
    """
    parts = ['<div class="car-charging-preview-debug" style="display:none;">']
    parts.append("EV authority withdrawn; native domestic control retained. Reason: ")
    parts.append(escape(str(report.get("execution_reason", "ev_unavailable"))))
    arrival = report.get("arrival_forecast") if isinstance(report, dict) else None
    if not isinstance(arrival, dict):
        parts.append("</div>")
        return "".join(parts)
    parts.append("<br>")
    parts.append("Arrival forecast: ")
    parts.append(escape(str(arrival.get("status", "unavailable"))))
    status = arrival.get("status")
    if status == "anticipated":
        parts.append("<br>")
        parts.append("Expected arrival: ")
        parts.append(_safe_iso(arrival.get("expected_arrival")))
        parts.append(" (p10 ")
        parts.append(_safe_iso(arrival.get("arrival_p10")))
        parts.append(", p90 ")
        parts.append(_safe_iso(arrival.get("arrival_p90")))
        parts.append(")")
        parts.append("<br>")
        parts.append("Confidence: ")
        parts.append(escape(str(arrival.get("confidence", "low"))))
        parts.append("<br>")
        parts.append("Sample size: ")
        parts.append(escape("{} eligible of {} completed trips ({})".format(arrival.get("eligible_trips", 0), arrival.get("completed_trips", 0), arrival.get("cohort", ""))))
        parts.append("<br>")
        parts.append("Source timestamp: ")
        parts.append(_safe_iso(arrival.get("prepared_at")))
        parts.append(" (history fetched ")
        parts.append(_safe_iso(arrival.get("history_fetched_at")))
        parts.append(", latest state ")
        parts.append(_safe_iso(arrival.get("latest_history_state_at")))
        parts.append(", departure observed ")
        parts.append(_safe_iso(arrival.get("departure_observed_at")))
        parts.append(")")
        parts.append("<br>")
        parts.append("Hypothetical demand: empty-car scenario (not a vehicle reading), ")
        parts.append(escape("{} kWh requested, {} kWh at site AC, {} kWh shortfall".format(_format_finite(arrival.get("requested_target_kwh"), 2), _format_finite(arrival.get("model_ac_kwh"), 2), _format_finite(arrival.get("shortfall_kwh"), 2))))
        parts.append("<br>")
        parts.append("Ready by: ")
        parts.append(_safe_iso(arrival.get("ready_by")))
        parts.append(" (guaranteed target met: ")
        parts.append(escape(str(arrival.get("guaranteed_target_met"))))
        parts.append(", execution permission: ")
        parts.append(escape(str(arrival.get("execution_permission"))))
        parts.append(")")
    else:
        parts.append("<br>")
        parts.append("Reason: ")
        parts.append(escape(str(arrival.get("reason", ""))))
        if arrival.get("prepared_at") is not None:
            parts.append("<br>")
            parts.append("Source timestamp: ")
            parts.append(_safe_iso(arrival.get("prepared_at")))
    parts.append("</div>")
    return "".join(parts)


def render_car_charging_preview(report, native_diagnostic, now=None):
    """Render only the ownership/freshness banner above the existing Plan table.

    The visible mainline never claims physical EV success: only the current
    command observed by the executor is reflected in the rows; the future
    forecast remains under revalidation. ISO timestamps live under Show
    Debug, not in the mainline.

    For shadow / off / error / unavailable candidates the banner labels the
    table as a read-only simulation and the compact "Simulation only — not
    executed" line is preserved. For a valid applied display (paired rows +
    fresh receipt + no stale-SoC/counterfactual hypothesis + coherent
    ``generated_at``) the healthy explanatory copy stays out of this preview
    block; the graph's initially closed details drawer carries it. An
    explicitly authorized partial target remains visible. For an
    applied-but-invalid receipt
    (expired / missing / future / false authority / blocked / invalid paired
    preview / stale ``generated_at``) the compact carries a bounded
    execution-blocked reason. A fresh domestic power-readiness observation
    with passing paired safety checks but no chart rows is reported as
    observed, with no trajectory or EV schedule fabricated. Parent
    ``status='apply_blocked'`` reports take the execution-blocked branch.
    A native import-limit diagnostic is labelled as a baseline forecast and
    remains visible when the applied report has no chart preview.
    """
    if not isinstance(report, dict) and not isinstance(native_diagnostic, dict):
        return ""
    now = now if now is not None else datetime.now(timezone.utc)
    if isinstance(report, dict) and report.get("status") == "household_fallback":
        return _household_fallback_debug(report)
    active = isinstance(report, dict) and report.get("status") != "disabled"
    status = report.get("status", "") if isinstance(report, dict) else ""
    is_applied = status == "applied"
    is_blocked = status == "apply_blocked"
    paired = _paired_preview(report, now) if active else None
    domestic_without_preview = is_applied and paired is None and _domestic_readiness_without_preview_allowed(report, now)
    applied_fresh = _applied_display_allowed(report, now, paired)
    compact = []
    debug_parts = []
    if paired is not None:
        if applied_fresh:
            compact.append("Active plan")
            if report.get("partial_target_authorised") is True:
                compact.append("EV {}% / {}% requested".format(_format_finite(report.get("planned_ev_percent_by_departure"), 1), _format_finite(report.get("requested_ev_target_percent"), 1)))
        elif is_applied or is_blocked:
            compact.append("execution blocked: {}".format(_applied_block_reason(report, now)))
        else:
            compact.append("Simulation only — not executed")
            if status in ("partial_candidate", "conditional_partial_candidate"):
                compact.append("EV {}% / {}% requested".format(_format_finite(report["planned_ev_percent_by_departure"], 1), _format_finite(report["requested_ev_target_percent"], 1)))
                compact.append("requested target NOT validated")
            if report.get("uses_stale_soc_assumption") or str(status).startswith("conditional_"):
                compact.append("EV SOC unconfirmed")
            if _is_exceeded(*_parse_native_diag(native_diagnostic)):
                compact.append("native grid limit exceeded")
            compact.append("Details: Show Debug")
    elif domestic_without_preview:
        compact.append("Domestic power-readiness command observed")
        compact.append("paired chart preview unavailable")
    elif (is_applied or is_blocked) and active:
        compact.append("execution blocked: {}".format(_applied_block_reason(report, now)))
    if applied_fresh:
        # The chart's initially closed detail drawer carries the healthy
        # applied-plan context. Keep the preview root and its debug evidence,
        # but do not put a second explanatory headline above the graph.
        banner = ""
    elif domestic_without_preview:
        banner = (
            '<div class="car-charging-preview-banner"><strong>Domestic power-readiness command observed:</strong> '
            "paired chart preview unavailable; no battery trajectory or EV schedule is shown. "
            "Home Assistant/inverter observation is not hardware-confirmed.</div>"
        )
    elif (is_applied or is_blocked) and active:
        banner = (
            '<div class="car-charging-preview-banner" role="alert"><strong>Execution blocked:</strong> '
            '<span class="car-charging-preview-debug" style="display:none;">'
            "the most recent applied receipt is not a valid display "
            "(expired, missing, future, false authority, blocked, missing "
            "preview, or stale generated_at). The table is empty by "
            "design — no native fallback, no simulated candidate is shown."
            "</span></div>"
        )
    else:
        banner = (
            '<div class="car-charging-preview-banner"><strong>EV candidate — read-only simulation:</strong> '
            '<span class="car-charging-preview-debug" style="display:none;">'
            "the existing Plan table displays this candidate and its own native-replay SOC, not the native control schedule. "
            "No inverter or EV commands are issued. The table keeps its original SOC/arrow format. Replay endpoints, PV10 and grid peaks are available in Dev (car_charging_report); "
            "EV kWh is a full-power reservation, not measured delivery.</span></div>"
            if active
            else '<div class="car-charging-preview-banner"><strong>Native plan view:</strong> the existing table shows the native schedule, not a combined candidate.</div>'
        )
    state, peak_w, minutes_over, limit_w = _parse_native_diag(native_diagnostic)
    native_exceeded = _is_exceeded(state, peak_w, minutes_over, limit_w)
    parts = ['<section class="car-charging-preview" style="margin-bottom:16px;">']
    if compact:
        visible_compact = compact[1:] if applied_fresh and compact[:1] == ["Active plan"] else compact
        if visible_compact:
            parts.append('<div class="car-charging-preview-summary">' + escape(" · ".join(visible_compact)) + "</div>")
        if applied_fresh:
            debug_parts.append("Observed: {}".format(_safe_iso(report.get("execution_observed_at"))))
            debug_parts.append("Valid until: {}".format(_safe_iso(report.get("execution_valid_until"))))
        if domestic_without_preview:
            debug_parts.append("Display: paired chart preview unavailable")
            debug_parts.append("Command observed: {}".format(_safe_iso(report.get("execution_observed_at"))))
            debug_parts.append("Valid until: {}".format(_safe_iso(report.get("execution_valid_until"))))
            debug_parts.append("Hardware confirmed: no")
        elif (is_applied or is_blocked) and not applied_fresh:
            debug_parts.append("Reason: {}".format(escape(_applied_block_reason(report, now))))
        debug_parts.append("Status: {}".format(escape(str(status))))
        debug_parts.append("Generated: {}".format(_safe_iso(report.get("generated_at"))))
        if report.get("soc_observed_at"):
            debug_parts.append("EV SOC observation: {}".format(escape(str(report["soc_observed_at"]))))
        reference_at = _safe_iso(report.get("planning_reference_at"))
        if reference_at:
            debug_parts.append("Planning reference: {}".format(reference_at))
    if compact and not (applied_fresh or is_applied or is_blocked):
        parts.append('<div class="car-charging-preview-debug" style="display:none;">' + banner + "</div>")
    elif not applied_fresh:
        parts.append(banner)
    # For shadow / partial / unavailable candidates without a compact line
    # the XSS-sensitive status field still belongs under Show Debug.
    if isinstance(report, dict) and not compact and status:
        parts.append('<div class="car-charging-preview-status car-charging-preview-debug" style="display:none;">' "<span>Status: <strong>{}</strong></span></div>".format(escape(str(status))))
    if native_exceeded and not applied_fresh:
        warning_scope = "This is a forecast diagnostic, not current measured load; the domestic command observation is shown separately." if domestic_without_preview else "This is a forecast diagnostic, not current measured load."
        parts.append(
            (
                '<div class="car-charging-preview-warning car-charging-preview-debug" style="display:none;">'
                if compact and not (is_applied or is_blocked)
                else '<div class="car-charging-preview-warning" role="alert" style="padding:10px;background:#fee;color:#a00;border:1px solid #c00;">'
            )
            + "<strong>Native import forecast exceeds the configured limit (Native import limit exceeded):</strong> "
            "Forecast peak {peak} W over a {limit} W configured import limit for {minutes} minute(s) in the native forecast; "
            '{scope} <span class="car-charging-preview-debug" style="display:none;">This is not an enforcement '
            "action and not a service outage claim.".format(
                peak=_format_finite(peak_w, 3),
                limit=_format_finite(limit_w, 0),
                minutes=_format_finite(minutes_over, 0),
                scope=warning_scope,
            )
            + "</span></div>"
        )
    elif native_exceeded and applied_fresh:
        parts.append(
            '<div class="car-charging-preview-warning car-charging-preview-debug" role="alert" style="display:none;padding:10px;background:#fee;color:#a00;border:1px solid #c00;"><strong>Native import '
            "limit exceeded (obsolete native counterfactual):</strong> shown for diagnostic distinction only; the displayed plan is the active applied receipt. "
            '<span class="car-charging-preview-debug" style="display:none;">Peak {peak} W over a {limit} W limit '
            "for {minutes} minute(s) in the native control plan (not the displayed active plan). This is "
            "a diagnostic indicator of the obsolete native plan, not an enforcement "
            "action against the applied receipt.".format(
                peak=_format_finite(peak_w, 0),
                limit=_format_finite(limit_w, 0),
                minutes=_format_finite(minutes_over, 0),
            )
            + "</span></div>"
        )
    if not isinstance(report, dict):
        parts.append("<p>No EV/battery planning report is available. No candidate is shown.</p></section>")
        return "".join(parts)
    if domestic_without_preview:
        parts.append('<div class="car-charging-preview-info">Command observation is current; no paired chart preview is available. No trajectory or EV schedule is shown.</div>')
        if debug_parts:
            parts.append('<div class="car-charging-preview-debug" style="display:none;">' + "<br>".join(debug_parts) + "</div>")
        parts.append("</section>")
        return "".join(parts)
    if (is_applied or is_blocked) and not applied_fresh:
        # Stale-SoC and conditional labels are only meaningful for shadow /
        # partial candidates; an applied or apply_blocked report is handled
        # by the compact line above.
        parts.append('<div class="car-charging-preview-stale">Report is in an applied/execution-blocked state. No simulated candidate is shown.</div>')
        if debug_parts:
            parts.append('<div class="car-charging-preview-debug" style="display:none;">' + "<br>".join(debug_parts) + "</div>")
            debug_parts = []
        parts.append("</section>")
        return "".join(parts)
    if not applied_fresh and (bool(report.get("uses_stale_soc_assumption")) or (isinstance(status, str) and status.startswith("conditional_"))):
        parts.append(('<div class="car-charging-preview-debug" style="display:none;">' if compact else '<div class="car-charging-preview-stale">') + "EV SOC is old: stale SoC assumption, conditional simulation, never valid for execution.</div>")
    age = _age_minutes(report.get("generated_at"), now)
    if age is not None and (age > _STALE_AGE_MINUTES or age < -1):
        parts.append('<div class="car-charging-preview-stale">' "Report is older than 15 minutes; treating as stale and showing " "no earlier schedule (future timestamps are also invalid).</div></section>")
        return "".join(parts)
    if age is None:
        parts.append('<div class="car-charging-preview-stale">' "Report timestamp is missing or malformed; no candidate is shown.</div></section>")
        return "".join(parts)
    if active and paired is None and not (is_applied or is_blocked):
        parts.append("<p>" + escape(_unavailable_reason(report)) + "</p>")
    elif status in ("partial_candidate", "conditional_partial_candidate") and not applied_fresh:
        parts.append(
            ('<div class="car-charging-preview-debug" style="display:none;">' if compact else '<div class="car-charging-preview-warning" role="alert">')
            + "Partial charging plan: EV forecast at least {planned}% by {departure}; requested target {requested}% remains unchanged and is NOT validated by this plan. "
            "This fresh plan passes both native power/reserve checks; it is the best validated alternative found, not proof of a maximum or of physical delivery.</div>".format(
                planned=_format_finite(report["planned_ev_percent_by_departure"], 1),
                requested=_format_finite(report["requested_ev_target_percent"], 1),
                departure=escape(report["ev_departure_at"]),
            )
        )
    if debug_parts:
        parts.append('<div class="car-charging-preview-debug" style="display:none;">' + "<br>".join(debug_parts) + "</div>")
    parts.append("</section>")
    return "".join(parts)


def get_car_charging_preview_update_js():
    """No second poller: the existing Plan refresh now updates banner and rows together."""
    return ""
