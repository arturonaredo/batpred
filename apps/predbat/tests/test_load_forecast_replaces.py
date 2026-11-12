# -----------------------------------------------------------------------------
# Predbat Home Battery System
# Copyright Trefor Southwell 2026 - All Rights Reserved
# This application maybe used for personal use only and not for commercial use
# -----------------------------------------------------------------------------
# fmt off
# pylint: disable=consider-using-f-string
# pylint: disable=line-too-long
# pylint: disable=attribute-defined-outside-init

"""Tests for replacing a paired historical load component with its forecast."""

from datetime import datetime, timedelta, timezone

from config import APPS_SCHEMA
from tests.test_load_forecast_history import build_load_minutes
from utils import is_debug_excluded_key


FORECAST_ENTITY = "sensor.space_heat_forecast"
ENERGY_ENTITY = "sensor.heating_energy_total"
ACTIVE_ENTITY = "binary_sensor.space_heating_active"
HOUSE_ENTITY = "sensor.whole_house_energy"
EV_ENTITY = "sensor.ev_energy"
FORECAST_REFERENCE = FORECAST_ENTITY + "$external"


def _forecast_records(now_utc, hours=49, invalid=None):
    """Build a cumulative 10-minute forecast covering the requested hours."""
    start = now_utc - timedelta(minutes=10)
    end = now_utc + timedelta(hours=hours)
    records = []
    timestamp = start
    previous_energy = 0.0
    while timestamp <= end:
        elapsed_minutes = int((timestamp - start).total_seconds() // 60)
        energy = elapsed_minutes * 0.0008
        if invalid == "nan" and elapsed_minutes == 600:
            energy = float("nan")
        elif invalid == "nonmonotonic" and elapsed_minutes == 600:
            energy = previous_energy - 0.1
        records.append({"last_updated": timestamp.isoformat(), "energy": energy})
        previous_energy = energy
        timestamp += timedelta(minutes=10)
    if invalid == "gap":
        records = [record for record in records if not now_utc + timedelta(hours=10) < datetime.fromisoformat(record["last_updated"]) < now_utc + timedelta(hours=12)]
    return records


def _history_records(now_utc, days, state, energy_per_step=0.003, shuffle=False, reset=False):
    """Build energy-counter and binary-state history with the requested lookback."""
    start = now_utc - timedelta(days=days)
    energy_records = []
    active_records = []
    energy = 0.0
    step_count = int(days * 24 * 60 / 5)
    for index in range(step_count + 1):
        timestamp = start + timedelta(minutes=index * 5)
        if index > 0:
            energy += energy_per_step
        if reset and index == step_count // 2:
            energy = 0.0
        energy_records.append(
            {
                "state": str(round(energy, 6)),
                "last_updated": timestamp.isoformat(),
                "attributes": {"unit_of_measurement": "kWh"},
            }
        )
        active_records.append({"state": state, "last_updated": timestamp.isoformat()})
    if shuffle:
        energy_records.reverse()
        active_records.reverse()
    return energy_records, active_records, energy


def _configure_pair(
    my_predbat,
    now_utc,
    *,
    active_state="on",
    history_days=15,
    active_history_days=None,
    load_history_days=15,
    forecast_hours=49,
    forecast_age_minutes=0,
    forecast_invalid=None,
    forecast_unit="kWh",
    counter_unit="kWh",
    energy_unavailable_range=None,
    zero_heat=False,
    shuffle=False,
    reset=False,
    enabled=True,
    forecast_only=False,
    replacement_spec=None,
    subminute_counter=False,
    active_subminute_transition=False,
    exact_duplicate_counter_timestamp=False,
):
    """Set up deterministic mocked HA state/history for one paired forecast."""
    energy_records, active_records, energy_total = _history_records(
        now_utc,
        history_days,
        active_state,
        energy_per_step=0.0 if zero_heat else 0.003,
        shuffle=shuffle,
        reset=reset,
    )
    if history_days < 15:
        energy_total = float(energy_records[-1]["state"])
    if active_history_days is not None:
        _, active_records, _ = _history_records(now_utc, active_history_days, active_state, energy_per_step=0.0, shuffle=shuffle)
    if subminute_counter:
        counter_start = now_utc - timedelta(days=3, hours=9, minutes=50)
        counter_end = counter_start + timedelta(minutes=5)
        before_row = next(row for row in energy_records if datetime.fromisoformat(row["last_updated"]) == counter_start)
        after_row = next(row for row in energy_records if datetime.fromisoformat(row["last_updated"]) == counter_end)
        before_value = float(before_row["state"])
        energy_records.extend(
            [
                {
                    "state": str(round(before_value + 0.0005, 6)),
                    "last_updated": (counter_start + timedelta(minutes=1, seconds=1, microseconds=606335)).isoformat(),
                    "attributes": {"unit_of_measurement": "kWh"},
                },
                {
                    "state": str(round(before_value + 0.001, 6)),
                    "last_updated": (counter_start + timedelta(minutes=1, seconds=21, microseconds=606335)).isoformat(),
                    "attributes": {"unit_of_measurement": "kWh"},
                },
            ]
        )
        assert abs(float(after_row["state"]) - before_value - 0.003) < 0.000001
    if active_subminute_transition:
        transition_time = now_utc - timedelta(days=1, hours=4)
        active_records.extend(
            [
                {"state": "off", "last_updated": (transition_time + timedelta(seconds=10, microseconds=123456)).isoformat()},
                {"state": "on", "last_updated": (transition_time + timedelta(seconds=30, microseconds=123456)).isoformat()},
            ]
        )
        active_records.sort(key=lambda row: datetime.fromisoformat(row["last_updated"]))
    if exact_duplicate_counter_timestamp:
        energy_records.append(dict(energy_records[-1]))
    if energy_unavailable_range is not None:
        gap_start, gap_end = energy_unavailable_range
        before_rows = [row for row in energy_records if datetime.fromisoformat(row["last_updated"]) < gap_start]
        after_rows = [row for row in energy_records if datetime.fromisoformat(row["last_updated"]) >= gap_end]
        gap_delta = float(after_rows[0]["state"]) - float(before_rows[-1]["state"])
        for row in energy_records:
            updated = datetime.fromisoformat(row["last_updated"])
            if gap_start <= updated < gap_end:
                row["state"] = "unavailable"
            elif updated >= gap_end:
                row["state"] = str(float(row["state"]) - gap_delta)
        energy_total -= gap_delta

    current_active = active_state
    source_records = _forecast_records(now_utc, hours=forecast_hours, invalid=forecast_invalid)
    forecast_raw = {
        "state": "ready",
        "last_updated": (now_utc - timedelta(minutes=forecast_age_minutes)).isoformat(),
        "attributes": {"external": source_records, "unit_of_measurement": forecast_unit},
    }
    energy_raw = {
        "state": str(energy_total),
        "last_updated": now_utc.isoformat(),
        "attributes": {"unit_of_measurement": counter_unit, "state_class": "total_increasing"},
    }
    active_raw = {"state": current_active, "last_updated": now_utc.isoformat(), "attributes": {}}

    def get_state_wrapper(entity_id=None, default=None, attribute=None, raw=False, **kwargs):
        """Return only the mocked records used by this focused test."""
        records = {
            FORECAST_ENTITY: forecast_raw,
            ENERGY_ENTITY: energy_raw,
            ACTIVE_ENTITY: active_raw,
        }
        record = records.get(entity_id)
        if record is None:
            return default
        if raw:
            return record
        if attribute is not None:
            return record.get("attributes", {}).get(attribute, default)
        return record.get("state", default)

    def get_history_wrapper(entity_id, days=30, required=True, tracked=True):
        """Return mocked recorder history in Home Assistant's one-entity list shape."""
        if entity_id == ENERGY_ENTITY:
            history = energy_records
        elif entity_id == ACTIVE_ENTITY:
            history = active_records
        else:
            return None
        return [history]

    if replacement_spec is None:
        replacement_spec = {"energy": ENERGY_ENTITY, "active": ACTIVE_ENTITY}
    my_predbat.args = {
        "load_forecast": [FORECAST_REFERENCE],
        "load_forecast_replaces": {FORECAST_REFERENCE: replacement_spec},
        "load_today": HOUSE_ENTITY,
        "car_charging_energy": EV_ENTITY,
    }
    my_predbat.get_state_wrapper = get_state_wrapper
    my_predbat.get_history_wrapper = get_history_wrapper
    my_predbat.now_utc = now_utc
    my_predbat.midnight_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    my_predbat.minutes_now = 12 * 60
    my_predbat.forecast_minutes = 48 * 60
    my_predbat.plan_interval_minutes = 30
    my_predbat.forecast_days = 3
    my_predbat.max_days_previous = 16  # days_previous=[15] uses 15 complete historical days plus today
    my_predbat.load_forecast_history = enabled
    my_predbat.load_forecast_only = forecast_only
    my_predbat.load_minutes = build_load_minutes({day: 0.002 for day in range(1, load_history_days + 1)})
    my_predbat.load_minutes_age = load_history_days
    my_predbat.base_load = 0.0
    my_predbat.holiday_days_left = 0
    my_predbat.holiday_load_scaling = 0.7
    my_predbat.get_holiday_minutes = lambda now, days: None
    my_predbat.days_previous = [15]
    my_predbat.days_previous_weight = [1.0]
    my_predbat.car_charging_hold = True
    my_predbat.car_charging_energy = build_load_minutes({day: 0.0004 for day in range(1, load_history_days + 1)})
    my_predbat.iboost_energy_subtract = False
    my_predbat.iboost_energy_today = None
    return energy_records, active_records


def _forecast_step(my_predbat, forecast, start_minute):
    """Read the energy over one predicted five-minute slot."""
    return sum(my_predbat.get_from_incrementing(forecast, minute, backwards=False) for minute in range(start_minute, start_minute + 5))


def _run_pair(my_predbat, now_utc, **options):
    """Fetch a pair, compute automatic history, and return the combined forecast."""
    _configure_pair(my_predbat, now_utc, **options)
    original_log = getattr(my_predbat, "log", None)
    log_lines = []
    my_predbat.log = lambda message, *args, **kwargs: log_lines.append(str(message))
    try:
        forecast, _ = my_predbat.fetch_extra_load_forecast(now_utc)
        history_forecast = my_predbat.compute_load_forecast_history(now_utc)
    finally:
        if original_log is None:
            del my_predbat.log
        else:
            my_predbat.log = original_log
    my_predbat.load_forecast_test_logs = log_lines
    combined = dict(forecast)
    for minute, value in history_forecast.items():
        combined[minute] = combined.get(minute, 0.0) + value
    return forecast, history_forecast, combined


def test_load_forecast_replaces(my_predbat):
    """A paired forecast removes only its active historical component, never EV or actual load."""
    print("**** Running load_forecast_replaces tests ****")
    failed = False
    now_utc = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)

    # The API is opt-in and uses a dict so old string-valued mappings remain valid.
    if APPS_SCHEMA.get("load_forecast_replaces", {}).get("type") != "dict":
        print("ERROR: load_forecast_replaces is missing from APPS_SCHEMA as an optional dict")
        failed = True
    if not is_debug_excluded_key("__load_forecast_replacements"):
        print("ERROR: live replacement history arrays must be excluded from debug/MCP state snapshots")
        failed = True

    # Active heat (0.003 kWh/5m) is removed from the 0.010 house total after EV (0.002); the
    # fresh space-heating forecast (0.004) is then added. Past comparison buckets keep full history.
    house_before = None
    ev_before = None
    forecast, history_forecast, combined = _run_pair(my_predbat, now_utc, shuffle=True, reset=True)
    house_before = dict(my_predbat.load_minutes)
    ev_before = dict(my_predbat.car_charging_energy)
    future = _forecast_step(my_predbat, combined, my_predbat.minutes_now)
    past = _forecast_step(my_predbat, combined, my_predbat.minutes_now - 120)
    if abs(future - 0.009) > 0.0003:
        print("ERROR: expected 0.005 non-heat house + 0.004 forecast = 0.009 kWh/5m, got {}".format(future))
        failed = True
    if abs(past - 0.008) > 0.0003:
        print("ERROR: elapsed forecast bucket changed; expected original 0.008 got {}".format(past))
        failed = True
    if dict(my_predbat.load_minutes) != house_before or dict(my_predbat.car_charging_energy) != ev_before:
        print("ERROR: replacement mutated raw house history or EV evidence")
        failed = True
    replacements = getattr(my_predbat, "__load_forecast_replacements", [])
    if not forecast or not history_forecast or len(replacements) != 1:
        print("ERROR: valid fresh pair was not loaded as one forecast replacement")
        failed = True
    accepted_log = "\n".join(my_predbat.load_forecast_test_logs)
    accepted_evidence_lines = list(my_predbat.load_forecast_test_logs)
    if (
        "Load forecast replacement binding summary: requested_pairs=1 accepted_pairs=1 fallback_pairs=0" not in accepted_log
        or "counter_history_covered_minutes=21600" not in accepted_log
        or "active_history_covered_minutes=21600" not in accepted_log
    ):
        print("ERROR: accepted replacement log is missing aggregate binding or exact 15-day history coverage: {}".format(accepted_log))
        failed = True
    application_log = next((line for line in my_predbat.load_forecast_test_logs if "Load forecast replacement application:" in line), "")
    try:
        removed_kwh = float(application_log.split("future_weighted_space_heating_removed_kWh=", 1)[1].split()[0])
    except (IndexError, ValueError):
        removed_kwh = 0.0
    if removed_kwh <= 0:
        print("ERROR: weighted future space-heating removal is not observable")
        failed = True

    # The raw whole-house bucket and EV meter are distinct inputs; replacement must not substitute one for the other.
    _, raw_house_bucket = my_predbat.get_filtered_load_minute(my_predbat.load_minutes, 1440, historical=False, step=5, base_in_raw=False)
    ev_bucket = sum(my_predbat.get_from_incrementing(my_predbat.car_charging_energy, minute) for minute in range(1440, 1445))
    if abs(raw_house_bucket - 0.010) > 0.0001 or abs(ev_bucket - 0.002) > 0.0001 or abs(raw_house_bucket - ev_bucket) < 0.001:
        print("ERROR: test fixture did not prove raw house history is distinct from EV energy")
        failed = True

    # Counter reads 20 seconds apart within one clock minute are distinct samples. Minute quantisation in
    # minute_data must preserve the 0.003 kWh across the enclosing five-minute interval and the pair must apply.
    _, _, subminute_combined = _run_pair(my_predbat, now_utc, subminute_counter=True)
    subminute_replacements = getattr(my_predbat, "__load_forecast_replacements", [])
    counter_start = now_utc - timedelta(days=3, hours=9, minutes=50)
    counter_end = counter_start + timedelta(minutes=5)
    counter_start_minute = int((now_utc - counter_start).total_seconds() / 60)
    counter_end_minute = int((now_utc - counter_end).total_seconds() / 60)
    subminute_energy = subminute_replacements[0]["energy_minutes"] if subminute_replacements else None
    preserved_energy = (
        subminute_energy.get(counter_end_minute, 0.0) - subminute_energy.get(counter_start_minute, 0.0)
        if subminute_energy
        else 0.0
    )
    subminute_future = _forecast_step(my_predbat, subminute_combined, my_predbat.minutes_now)
    if len(subminute_replacements) != 1 or abs(preserved_energy - 0.003) > 0.0001 or abs(subminute_future - 0.009) > 0.0003:
        print(
            "ERROR: 20-second counter readings in one minute must retain 0.003 kWh and allow safe replacement; "
            "accepted={} preserved={} future_step={}".format(len(subminute_replacements), preserved_energy, subminute_future)
        )
        failed = True

    # The oldest day sampled for tomorrow at 00:00 is 15 days plus the current 12-hour offset ago.
    # With 16 days of source data, every weighted sample must still remove the same 0.003 kWh bucket.
    _, oldest_day_history, _ = _run_pair(my_predbat, now_utc, history_days=16, load_history_days=16)
    oldest_day_step = _forecast_step(my_predbat, oldest_day_history, 24 * 60)
    if abs(oldest_day_step - 0.005) > 0.000001:
        print("ERROR: the day-15 midnight sample should retain EV/base load and subtract heating fully, got {}".format(oldest_day_step))
        failed = True

    # fetch_sensor_data resets the house-history age before fetching extra forecasts, then loads
    # the house history afterwards. Coverage must not depend on that not-yet-loaded age.
    _configure_pair(my_predbat, now_utc, history_days=16, load_history_days=16)
    my_predbat.load_minutes_age = 0
    my_predbat.fetch_extra_load_forecast(now_utc)
    my_predbat.load_minutes_age = 16
    first_cycle_history = my_predbat.compute_load_forecast_history(now_utc)
    first_cycle_step = _forecast_step(my_predbat, first_cycle_history, 24 * 60)
    if abs(first_cycle_step - 0.005) > 0.000001:
        print("ERROR: first-cycle fetch order must also replace the oldest weighted day fully, got {}".format(first_cycle_step))
        failed = True

    # Keep the pair valid with the original 15-day source threshold, but do not infer an active
    # state beyond the active recorder's oldest sample when the counter itself has 16 days.
    _run_pair(my_predbat, now_utc, history_days=16, active_history_days=15, load_history_days=16)
    active_extent_replacements = getattr(my_predbat, "__load_forecast_replacements", [])
    oldest_sample_minute = 15 * 24 * 60 + my_predbat.minutes_now
    if (
        len(active_extent_replacements) != 1
        or active_extent_replacements[0]["energy_valid_minutes"].get(oldest_sample_minute, 0) != 1
        or active_extent_replacements[0]["active_minutes"].get(oldest_sample_minute, 0) != 0
    ):
        print("ERROR: 15-day active history must remain unknown at a 15.5-day counter sample")
        failed = True

    # A same-minute active-state transition is ambiguous: retain that minute's house energy while
    # still applying the rest of the valid pair, rather than subtracting heating/ACS indiscriminately.
    _, _, _ = _run_pair(my_predbat, now_utc, active_subminute_transition=True)
    transition_replacements = getattr(my_predbat, "__load_forecast_replacements", [])
    transition_bucket = my_predbat.sample_load_history_bucket(1680, None, 999999, replacement_mask=(True,))
    if len(transition_replacements) != 1 or not transition_bucket or abs(transition_bucket[2] - 0.0024) > 0.0002:
        print(
            "ERROR: same-minute active transition must retain one minute of meter energy; "
            "accepted={} bucket={}".format(len(transition_replacements), transition_bucket)
        )
        failed = True

    # Exact duplicate recorder timestamps remain invalid even though distinct sub-minute rows are accepted.
    duplicate_forecast, _, _ = _run_pair(my_predbat, now_utc, exact_duplicate_counter_timestamp=True)
    if duplicate_forecast or getattr(my_predbat, "__load_forecast_replacements", []):
        print("ERROR: exact duplicate counter timestamps should still reject their replacement pair")
        failed = True

    # A short unavailable counter interval with no cumulative delta is preserved locally, not treated as a
    # reason to reject all 15 days of otherwise traceable history.
    gap_start = now_utc - timedelta(days=3, hours=9, minutes=50)
    gap_end = gap_start + timedelta(minutes=6)
    _, _, gap_combined = _run_pair(my_predbat, now_utc, energy_unavailable_range=(gap_start, gap_end))
    gap_step = _forecast_step(my_predbat, gap_combined, 24 * 60 + 2 * 60 + 10)
    gap_log = "\n".join(my_predbat.load_forecast_test_logs)
    gap_evidence_lines = list(my_predbat.load_forecast_test_logs)
    gap_application_log = next((line for line in my_predbat.load_forecast_test_logs if "Load forecast replacement application:" in line), "")
    try:
        gap_removed_kwh = float(gap_application_log.split("future_weighted_space_heating_removed_kWh=", 1)[1].split()[0])
    except (IndexError, ValueError):
        gap_removed_kwh = 0.0
    if (
        not getattr(my_predbat, "__load_forecast_replacements", [])
        or "accepted_pairs=1 fallback_pairs=0" not in gap_log
        or "counter_history_covered_minutes=21600" in gap_log
        or "counter_history_uncovered_minutes=15" not in gap_log
        or gap_removed_kwh >= removed_kwh - 0.0001
        or not (0.0089 < gap_step < 0.012)
    ):
        print("ERROR: unavailable counter interval should retain only local house history, got step={} logs={}".format(gap_step, gap_log))
        failed = True

    # ACS/idle consumption while space heating is OFF or UNKNOWN remains in the historical base.
    for state in ("off", "unknown", "unavailable"):
        _, _, combined = _run_pair(my_predbat, now_utc, active_state=state)
        step = _forecast_step(my_predbat, combined, my_predbat.minutes_now)
        if abs(step - 0.012) > 0.0003:
            print("ERROR: {} active history should retain meter energy; expected 0.012 got {}".format(state, step))
            failed = True

    # A partial forecast contributes only where it covers a complete 5-minute slot; elsewhere the
    # full historical heating sample remains, avoiding an under-forecast at the uncovered tail.
    _, _, combined = _run_pair(my_predbat, now_utc, forecast_hours=24)
    covered = _forecast_step(my_predbat, combined, my_predbat.minutes_now + 20 * 60)
    uncovered = _forecast_step(my_predbat, combined, my_predbat.minutes_now + 30 * 60)
    if abs(covered - 0.009) > 0.0003 or abs(uncovered - 0.008) > 0.0003:
        print("ERROR: partial coverage expected 0.009 covered / 0.008 uncovered, got {} / {}".format(covered, uncovered))
        failed = True

    # Stale, malformed, or history-incomplete pairs must use the untouched historical forecast.
    coverage_fallback_evidence = ""
    for label, options in (
        ("stale forecast", {"forecast_age_minutes": 16}),
        ("NaN forecast", {"forecast_invalid": "nan"}),
        ("non-monotonic forecast", {"forecast_invalid": "nonmonotonic"}),
        ("forecast gap", {"forecast_invalid": "gap"}),
        ("wrong forecast units", {"forecast_unit": "Wh"}),
        ("short counter history", {"history_days": 5}),
        ("short active-state history", {"active_history_days": 5}),
        ("active-state history short by one day", {"active_history_days": 14}),
        ("wrong counter units", {"counter_unit": "Wh"}),
        ("unsupported mode", {"enabled": False}),
        ("forecast-only mode at paired-fetch time", {"forecast_only": True}),
    ):
        forecast, _, combined = _run_pair(my_predbat, now_utc, **options)
        step = _forecast_step(my_predbat, combined, my_predbat.minutes_now)
        if forecast or abs(step - 0.008) > 0.0003 or getattr(my_predbat, "__load_forecast_replacements", []):
            print("ERROR: {} did not fall back to full history (forecast={}, step={})".format(label, bool(forecast), step))
            failed = True
        if label == "active-state history short by one day":
            fallback_log = "\n".join(my_predbat.load_forecast_test_logs)
            coverage_fallback_evidence = fallback_log
            if (
                "active-state history does not cover the complete automatic-history window" not in fallback_log
                or "requested_pairs=1 accepted_pairs=0 fallback_pairs=1" not in fallback_log
                or "status=fallback_full_history" not in fallback_log
            ):
                print("ERROR: the 15-day active-history fallback reason is not visible in logs")
                failed = True

    # A zero-energy counter is valid data, not a missing-history sentinel.
    forecast, _, combined = _run_pair(my_predbat, now_utc, zero_heat=True)
    step = _forecast_step(my_predbat, combined, my_predbat.minutes_now)
    if not forecast or abs(step - 0.012) > 0.0003:
        print("ERROR: valid zero-heat history should retain base load plus forecast, got {}".format(step))
        failed = True

    # The legacy simple-string value is still accepted for a dedicated component-only meter.
    forecast, _, combined = _run_pair(my_predbat, now_utc, replacement_spec=ENERGY_ENTITY)
    step = _forecast_step(my_predbat, combined, my_predbat.minutes_now)
    if not forecast or abs(step - 0.009) > 0.0003:
        print("ERROR: legacy component-only counter mapping did not replace heating history, got {}".format(step))
        failed = True

    # A counter already configured for EV must not be borrowed as a heating counter.
    _configure_pair(my_predbat, now_utc)
    my_predbat.args["load_forecast_replaces"][FORECAST_REFERENCE] = {"energy": EV_ENTITY, "active": ACTIVE_ENTITY}
    forecast, _ = my_predbat.fetch_extra_load_forecast(now_utc)
    if forecast or getattr(my_predbat, "__load_forecast_replacements", []):
        print("ERROR: EV energy counter was accepted as a heating-history replacement")
        failed = True

    if not failed:
        print("PASS: active-only replacement, conservative fallback, future coverage and EV/raw-load isolation")
        print("EVIDENCE: raw_house_bucket={:.4f}kWh ev_bucket={:.4f}kWh distinct=true raw_load_and_ev_unchanged=true".format(raw_house_bucket, ev_bucket))
        for line in accepted_evidence_lines + gap_evidence_lines:
            if "Load forecast replacement binding summary:" in line or "Load forecast replacement application:" in line:
                print("EVIDENCE: {}".format(line))
        for line in coverage_fallback_evidence.splitlines():
            if (
                "Rejecting paired load forecast" in line
                or "Load forecast replacement binding summary:" in line
                or "Load forecast replacement application:" in line
            ):
                print("EVIDENCE: {}".format(line))
    return failed
