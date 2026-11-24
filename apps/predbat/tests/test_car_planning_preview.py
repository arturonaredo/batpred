"""Read-only preview rendering and replay-row construction tests.

Fixtures mirror the actual Predbat native replay contract used by the
``car_charging_plan`` adapter:

* ``base.midnight_utc`` is offset-aware; ``base.minutes_now`` is the
  current minute-of-day; ``base.forecast_minutes`` is the planning
  horizon in minutes (always a multiple of 5); ``base.soc_max``,
  ``base.reserve`` and ``base.car_charging_rate[0]`` are the only fields
  the preview reads.
* ``pred.predict_soc`` is keyed by RELATIVE minute 0, 5, 10, ... up to
  ``forecast_minutes``; the terminal endpoint is ``pred.final_soc`` and
  is only used for the last step.
* ``pred.predict_grid_power`` is keyed by ISO-8601 timestamp strings
  formatted with ``TIME_FORMAT`` from ``base.midnight_utc + timedelta(
  minutes=base.minutes_now + relative_minute)``, one entry per 5-min
  step.
* ``result.status == "feasible"`` and ``result.actions[i].ev_action``
  is ``"on"`` / ``"off"``; ``action.slot_index`` matches ``slot.index``;
  ``action.battery_mode`` is one of ``demand``/``hold``/``charge``.
* Forecast maps are numeric kWh per 5-min slice — summed directly into
  the row, never rescaled.

Every fixture is built in-memory; no HA service, network or actuator is
ever touched.

The ``render_car_charging_preview`` function is BANNER ONLY: it never
emits a ``<table>`` or a ``<details>`` block. The combined candidate is
shown in the existing Plan table by the parent integration; the preview
itself only labels whether a paired, fresh, native-passed candidate
exists, surfaces the native import-limit diagnostic warning unchanged,
and never claims execution authority. ``get_car_charging_preview_update_js``
returns an empty script — the parent owns the polling and the full API
response; the preview div is updated atomically by the existing
plan_data poll.
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from const import TIME_FORMAT
from car_charging_view import (
    PREVIEW_ROW_LIMIT,
    build_replay_preview,
    get_car_charging_preview_update_js,
    render_car_charging_preview,
)

# ----------------------------------------------------------------------
# Synthetic fixtures — actual Predbat native replay contract
# ----------------------------------------------------------------------


def _base_fixture(minutes=120, now_minute=12 * 60):
    """Build a base with the exact field set the preview reads."""
    base = SimpleNamespace()
    midnight = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    base.midnight_utc = midnight
    base.minutes_now = int(now_minute)
    base.now_utc = midnight + timedelta(minutes=base.minutes_now)
    base.forecast_minutes = int(minutes)
    base.soc_max = 10.0
    base.reserve = 1.0
    base.car_charging_rate = [6.77]
    base.load_minutes_step = {m: 0.4 for m in range(0, minutes, 5)}
    base.load_minutes_step10 = {m: 0.5 for m in range(0, minutes, 5)}
    base.pv_forecast_minute_step = {m: 0.2 for m in range(0, minutes, 5)}
    base.pv_forecast_minute10_step = {m: 0.1 for m in range(0, minutes, 5)}
    return base


def _grid_stamp(base, relative_minute):
    """Return the timestamp string the native grid trace uses for a relative minute."""
    return (base.midnight_utc + timedelta(minutes=base.minutes_now + relative_minute)).strftime(TIME_FORMAT)


def _slot(index, start_min, duration_min=5):
    """Represent one native five-minute forecast step."""
    return SimpleNamespace(index=index, start_min=start_min, duration_min=duration_min)


def _action(index, mode="demand", ev_on=False):
    """Represent an actual solver action with explicit EV state."""
    return SimpleNamespace(
        slot_index=index,
        battery_mode=mode,
        ev_action="on" if ev_on else "off",
    )


def _pred(base, soc_kw=5.0, grid_kw=2.0, slope_per_step=0.0, last_value=None):
    """Build a finite predict_soc / predict_grid_power dict for the horizon.

    SOC keys are RELATIVE 5-min steps; grid keys are ISO timestamp strings.
    """
    predict_soc = {}
    grid = {}
    steps = list(range(0, base.forecast_minutes, 5))
    for index, minute in enumerate(steps):
        predict_soc[minute] = float(soc_kw) + float(slope_per_step) * index
        grid[_grid_stamp(base, minute)] = float(grid_kw) + float(slope_per_step) * index * 0.001
    return SimpleNamespace(
        predict_soc=predict_soc,
        predict_grid_power=grid,
        final_soc=last_value if last_value is not None else float(soc_kw) + float(slope_per_step) * (len(steps) - 1),
    )


def _slots(base):
    """Build every indexed step in the synthetic native horizon."""
    return [_slot(i, i * 5, 5) for i in range(0, base.forecast_minutes // 5)]


def _actions(base, ev_step_indices=(), modes=("demand",)):
    """Build actions matching the synthetic step indices."""
    actions = []
    for i in range(0, base.forecast_minutes // 5):
        actions.append(_action(i, mode=modes[i % len(modes)], ev_on=i in ev_step_indices))
    return actions


def _result(actions, status="feasible"):
    """Expose the minimal solver result used by the replay adapter."""
    return SimpleNamespace(actions=actions, status=status)


# ----------------------------------------------------------------------
# build_replay_preview tests — actual schema, must keep passing
# ----------------------------------------------------------------------


def test_partial_bins_and_absolute_clock_alignment():
    """A non-aligned start produces truthful absolute first/last partial timestamps aligned to half-hours."""
    base = _base_fixture(minutes=120, now_minute=12 * 60 + 5)
    pred = _pred(base, soc_kw=5.0, grid_kw=2.0)
    result = _result(_actions(base))
    rows = build_replay_preview(base, pred, result, _slots(base), scenario=0)
    assert rows, rows
    # First row spans minutes_now to the next half-hour boundary.
    first_abs_start = base.minutes_now
    first_abs_end = ((base.minutes_now // 30) + 1) * 30
    # Last row spans the last full half-hour boundary to minutes_now + horizon.
    last_abs_end = base.minutes_now + base.forecast_minutes
    assert rows[0]["start"] == (base.midnight_utc + timedelta(minutes=first_abs_start)).strftime(TIME_FORMAT)
    assert rows[0]["end"] == (base.midnight_utc + timedelta(minutes=first_abs_end)).strftime(TIME_FORMAT)
    # First row is the 25-min partial (12:05 -> 12:30); horizon is 120 min, so last bin ends at 14:05.
    assert rows[0]["minutes"] == 25
    assert rows[-1]["end"] == (base.midnight_utc + timedelta(minutes=last_abs_end)).strftime(TIME_FORMAT)
    assert rows[-1]["minutes"] == 5
    assert "demand" in rows[0]["battery_modes"]
    assert rows[0]["battery_modes"] == "demand"


def test_partial_first_bin_when_now_unaligned():
    """When now sits inside a half-hour, the first row is partial."""
    base = _base_fixture(minutes=65, now_minute=12 * 60 + 10)
    pred = _pred(base, soc_kw=5.0, grid_kw=2.0)
    result = _result(_actions(base))
    rows = build_replay_preview(base, pred, result, _slots(base), scenario=0)
    assert rows, rows
    first = rows[0]
    # 12:10 to 12:30 = 20 min partial.
    assert first["minutes"] == 20, first
    # Sum equals the full horizon.
    minutes_sum = sum(row["minutes"] for row in rows)
    assert minutes_sum == 65, minutes_sum


def test_soc_start_differs_from_end_on_slope():
    """A non-trivial SOC slope is reflected as distinct start and end percentages."""
    base = _base_fixture(minutes=60)
    pred = _pred(base, soc_kw=4.0, grid_kw=1.0, slope_per_step=0.1)
    result = _result(_actions(base))
    rows = build_replay_preview(base, pred, result, _slots(base), scenario=0)
    assert rows
    first = rows[0]
    assert first["battery_start_percent"] < first["battery_end_percent"]
    assert 0 <= first["battery_min_percent"] <= 100
    assert first["battery_start_percent"] >= first["battery_min_percent"] - 1e-9


def test_peak_w_is_upper_bound_never_averaged_down():
    """A 9.5 kW spike survives as >=9500 W; the upper bound is never floored downward."""
    base = _base_fixture(minutes=60)
    pred = _pred(base, soc_kw=5.0, grid_kw=2.0)
    pred.predict_grid_power[_grid_stamp(base, 15)] = 9.5
    result = _result(_actions(base))
    rows = build_replay_preview(base, pred, result, _slots(base), scenario=0)
    assert rows
    peak = max(row["grid_peak_w"] for row in rows)
    assert peak >= 9500, peak
    assert peak <= 9500.5 + 1e-6, peak


def test_pv10_uses_scenario_specific_maps():
    """Scenario 1 reads load_minutes_step10 / pv_forecast_minute10_step; scenario 0 reads nominal."""
    base = _base_fixture(minutes=60)
    base.load_minutes_step10 = {m: 0.9 for m in range(0, 60, 5)}
    base.pv_forecast_minute10_step = {m: 0.05 for m in range(0, 60, 5)}
    pred = _pred(base, soc_kw=5.0, grid_kw=2.0)
    result = _result(_actions(base))
    rows_nominal = build_replay_preview(base, pred, result, _slots(base), scenario=0)
    rows_pv10 = build_replay_preview(base, pred, result, _slots(base), scenario=1)
    assert rows_nominal and rows_pv10
    assert rows_pv10[0]["house_kwh"] > rows_nominal[0]["house_kwh"]
    assert rows_pv10[0]["pv_kwh"] < rows_nominal[0]["pv_kwh"]


def test_mixed_modes_preserved_in_row():
    """A mode transition splits rows instead of merging totals into a false timeline."""
    base = _base_fixture(minutes=30)
    pred = _pred(base, soc_kw=5.0, grid_kw=2.0)
    actions = [
        _action(0, "demand"),
        _action(1, "charge"),
        _action(2, "hold"),
        _action(3, "demand"),
        _action(4, "charge"),
        _action(5, "demand"),
    ]
    result = _result(actions)
    rows = build_replay_preview(base, pred, result, _slots(base), scenario=0)
    assert rows
    assert [row["battery_modes"] for row in rows] == ["demand", "charge", "hold", "demand", "charge", "demand"]
    assert [row["minutes"] for row in rows] == [5] * 6


def test_rows_follow_the_configured_battery_window():
    """Ignoring the selected window would collapse five-minute rows into half-hours."""
    for window, count in ((5, 12), (15, 4), (30, 2), (60, 1)):
        base = _base_fixture(minutes=60)
        base.plan_preview_row_minutes = window
        pred = _pred(base)
        rows = build_replay_preview(base, pred, _result(_actions(base)), _slots(base), scenario=0)
        assert len(rows) == count, (window, rows)
        assert all(row["minutes"] == window and row["battery_modes"] == "demand" for row in rows)
        assert abs(sum(row["house_kwh"] for row in rows) - 4.8) < 1e-9
        assert abs(sum(row["pv_kwh"] for row in rows) - 2.4) < 1e-9


def test_configured_rows_preserve_partial_edges_and_step_costs():
    """Subdividing a partial interval preserves its own SOC, peak, energy and cumulative cost."""
    base = _base_fixture(minutes=65, now_minute=725)
    base.plan_preview_row_minutes = 15
    pred = _pred(base, slope_per_step=0.1, last_value=6.3)
    pred.metric_time = {_grid_stamp(base, minute): 100 + minute for minute in range(0, 65, 5)}
    pred.final_metric = 165
    pred.predict_grid_power[_grid_stamp(base, 15)] = 5.5
    rows = build_replay_preview(base, pred, _result(_actions(base)), _slots(base), scenario=0)
    assert [row["minutes"] for row in rows] == [10, 15, 15, 15, 10]
    assert rows[0]["start"] == "2026-01-01T12:05:00+0000"
    assert rows[-1]["end"] == "2026-01-01T13:10:00+0000"
    assert rows[1]["battery_start_percent"] == 52
    assert rows[-1]["battery_end_percent"] == 63
    assert rows[1]["grid_peak_w"] == 5500.5
    assert abs(sum(row["cost_change"] for row in rows) - 0.65) < 1e-9
    assert abs(rows[-1]["total_cost"] + rows[-1]["cost_change"] - 1.65) < 1e-9


def test_rows_split_at_charge_target_and_ev_boundaries():
    """A row must not hide different charge targets or EV relay intents."""
    base = _base_fixture(minutes=30)
    base.plan_preview_row_minutes = 30
    base.plan_preview_charge_limits = {base.minutes_now + minute: 50 if minute < 10 else 60 for minute in range(0, 30, 5)}
    actions = [_action(index, "charge", ev_on=index >= 4) for index in range(6)]
    rows = build_replay_preview(base, _pred(base), _result(actions), _slots(base), scenario=0)
    assert [row["minutes"] for row in rows] == [10, 10, 10]
    assert [row["charge_target_percent"] for row in rows] == [50, 60, 60]
    assert [row["ev_on_minutes"] for row in rows] == [0, 0, 10]


def test_five_minute_full_horizon_is_not_truncated_by_the_preview_limit():
    """The complete 48-hour five-minute schedule must survive preview and view validation."""
    from car_charging_view import build_car_charging_plan_view

    base = _base_fixture(minutes=2880, now_minute=5)
    base.plan_preview_row_minutes = 5
    actions = _actions(base, modes=("charge", "demand"))
    rows = build_replay_preview(base, _pred(base), _result(actions), _slots(base), scenario=0)
    assert len(rows) == 576 and sum(row["minutes"] for row in rows) == 2880
    now = base.now_utc
    report = {"status": "candidate", "generated_at": now.isoformat(), "native_checks": {key: {"passed": True, "checked_minutes": 2880} for key in ("nominal", "pv10")}, "preview": {"nominal": rows, "pv10": deepcopy(rows)}}
    view = build_car_charging_plan_view({}, report, now=now)
    assert len(view["rows"]) == 576
    assert [row["state"] for row in view["rows"][:4]] == ["Chrg", "Demand", "Chrg", "Demand"]
    assert all(" + " not in row["state_text"] for row in view["rows"])


def test_invalid_preview_window_is_not_silently_replaced():
    """Invalid configuration must not fabricate a valid-looking display cadence."""
    for value in (0, -5, 10, True, "5", float("nan")):
        base = _base_fixture(minutes=30)
        base.plan_preview_row_minutes = value
        assert build_replay_preview(base, _pred(base), _result(_actions(base)), _slots(base), scenario=0) == []


def test_single_mode_mismatch_between_scenarios_is_rejected():
    """A nominal Charge and PV10 Demand cannot describe one applied interval."""
    from car_charging_view import build_car_charging_plan_view
    from tests.test_car_planning_single_table import _good_report, _native_plan

    report = _good_report()
    report["preview"]["pv10"][0]["battery_modes"] = "charge"
    view = build_car_charging_plan_view(_native_plan(), report)
    assert view.get("plan_unavailable") is True and view["rows"] == []


def test_plan_page_and_polling_api_show_single_action_intervals():
    """Both user-facing routes retain the new interval boundaries, not a mixed-mode label."""
    import asyncio
    import json
    import re
    from tests.test_car_planning_single_table import _build_page, _good_report

    for width, expected in ((5, 6), (60, 1)):
        base = _base_fixture(minutes=30 if width == 5 else 60)
        base.plan_preview_row_minutes = width
        result = _result(_actions(base, modes=("demand", "charge") if width == 5 else ("demand",)))
        report = _good_report()
        report["preview"] = {name: build_replay_preview(base, _pred(base), result, _slots(base), scenario=index) for index, name in enumerate(("nominal", "pv10"))}
        original = deepcopy(report)
        page = _build_page(report)
        request = SimpleNamespace(rel_url=SimpleNamespace(query={}))
        html = asyncio.run(page.html_plan(request))
        match = re.search(r"window\.planData\s*=\s*({.*?});", html.text)
        assert match
        page_plan = json.loads(match.group(1))
        api_plan = json.loads(asyncio.run(page.html_api_plan_data(request)).text)["plan"]
        for plan in (page_plan, api_plan):
            assert len(plan["rows"]) == expected
            assert all(row["state"] in ("Demand", "Chrg") and " + " not in row["state_text"] for row in plan["rows"])
            assert plan["rows"][0]["plan_interval_end"] == ("2026-01-01T12:05:00+0000" if width == 5 else "2026-01-01T13:00:00+0000")
        assert report == original


def test_rejects_nonfinite_grid_power():
    """A NaN or inf in the grid trace causes the whole preview to be empty."""
    base = _base_fixture(minutes=60)
    pred = _pred(base, soc_kw=5.0, grid_kw=2.0)
    pred.predict_grid_power[_grid_stamp(base, 10)] = float("nan")
    result = _result(_actions(base))
    rows = build_replay_preview(base, pred, result, _slots(base), scenario=0)
    assert rows == []
    base = _base_fixture(minutes=60)
    pred = _pred(base, soc_kw=5.0, grid_kw=2.0)
    pred.predict_grid_power[_grid_stamp(base, 10)] = float("inf")
    rows = build_replay_preview(base, pred, result, _slots(base), scenario=0)
    assert rows == []


def test_rejects_incomplete_grid_trace():
    """A missing 5-min grid stamp is treated as incomplete; nothing is filled in."""
    base = _base_fixture(minutes=60)
    pred = _pred(base, soc_kw=5.0, grid_kw=2.0)
    pred.predict_grid_power.pop(_grid_stamp(base, 25), None)
    result = _result(_actions(base))
    rows = build_replay_preview(base, pred, result, _slots(base), scenario=0)
    assert rows == []


def test_rejects_nonfeasible_status():
    """A bounded_no_plan / error / infeasible status produces no rows."""
    base = _base_fixture(minutes=60)
    pred = _pred(base, soc_kw=5.0, grid_kw=2.0)
    for status in ("bounded_no_plan", "error", "infeasible", None):
        result = SimpleNamespace(actions=_actions(base), status=status)
        assert build_replay_preview(base, pred, result, _slots(base), scenario=0) == []


def test_does_not_mutate_inputs():
    """Every input is returned untouched, even on a deliberately broken trace."""
    base = _base_fixture(minutes=60)
    pred = _pred(base, soc_kw=5.0, grid_kw=2.0)
    slots = _slots(base)
    actions = _actions(base)
    result = _result(actions)
    base_snapshot = {
        "load_minutes_step": deepcopy(base.load_minutes_step),
        "load_minutes_step10": deepcopy(base.load_minutes_step10),
        "pv_forecast_minute_step": deepcopy(base.pv_forecast_minute_step),
        "pv_forecast_minute10_step": deepcopy(base.pv_forecast_minute10_step),
    }
    pred_snapshot = {
        "predict_soc": deepcopy(pred.predict_soc),
        "predict_grid_power": deepcopy(pred.predict_grid_power),
    }
    rows = build_replay_preview(base, pred, result, slots, scenario=0)
    assert rows
    assert base.load_minutes_step == base_snapshot["load_minutes_step"]
    assert base.load_minutes_step10 == base_snapshot["load_minutes_step10"]
    assert base.pv_forecast_minute_step == base_snapshot["pv_forecast_minute_step"]
    assert base.pv_forecast_minute10_step == base_snapshot["pv_forecast_minute10_step"]
    assert pred.predict_soc == pred_snapshot["predict_soc"]
    assert pred.predict_grid_power == pred_snapshot["predict_grid_power"]


def test_total_horizon_2880_yields_97_rows():
    """A full 48h horizon with a non-zero now produces exactly 97 rows (96 half-hours + 1 partial first)."""
    base = _base_fixture(minutes=48 * 60, now_minute=10)  # 10 min after midnight → first partial = 20 min
    pred = _pred(base, soc_kw=5.0, grid_kw=2.0)
    result = _result(_actions(base))
    rows = build_replay_preview(base, pred, result, _slots(base), scenario=0)
    assert len(rows) == 97
    assert len(rows) <= PREVIEW_ROW_LIMIT


def test_row_minutes_cover_full_horizon_exactly():
    """Sum of row minutes equals the full forecast horizon (no double-counting, no gaps)."""
    base = _base_fixture(minutes=180, now_minute=12 * 60 + 15)
    pred = _pred(base, soc_kw=5.0, grid_kw=2.0)
    result = _result(_actions(base))
    rows = build_replay_preview(base, pred, result, _slots(base), scenario=0)
    assert sum(row["minutes"] for row in rows) == base.forecast_minutes


def test_house_kwh_sums_5min_kwh_directly():
    """house_kwh is the direct sum of 5-min kWh forecast values for the slices in the bin."""
    base = _base_fixture(minutes=60, now_minute=12 * 60)
    base.load_minutes_step = {m: 0.4 for m in range(0, 60, 5)}
    pred = _pred(base, soc_kw=5.0, grid_kw=2.0)
    result = _result(_actions(base))
    rows = build_replay_preview(base, pred, result, _slots(base), scenario=0)
    # Six 5-min slices of 0.4 kWh = 2.4 kWh per full 30-min bin.
    assert abs(rows[0]["house_kwh"] - 2.4) < 1e-9, rows[0]["house_kwh"]


def test_ev_reserved_uses_full_rate_over_on_minutes():
    """EV reserved kWh is the full configured charging rate times on-minutes, divided by 60."""
    base = _base_fixture(minutes=60, now_minute=12 * 60)
    pred = _pred(base, soc_kw=5.0, grid_kw=2.0)
    actions = [_action(i, mode="demand", ev_on=(i < 6)) for i in range(0, 12)]
    result = _result(actions)
    rows = build_replay_preview(base, pred, result, _slots(base), scenario=0)
    assert abs(rows[0]["ev_reserved_kwh"] - 6.77 * 30 / 60) < 1e-9, rows[0]["ev_reserved_kwh"]
    assert rows[1]["ev_reserved_kwh"] == 0.0


def test_rejects_unknown_modes():
    """A non-{demand,hold,charge} mode in any action is rejected and the preview is empty."""
    base = _base_fixture(minutes=60)
    pred = _pred(base, soc_kw=5.0, grid_kw=2.0)
    actions = _actions(base)
    actions[3].battery_mode = "unknown_mode"
    result = _result(actions)
    rows = build_replay_preview(base, pred, result, _slots(base), scenario=0)
    assert rows == []


def test_rejects_missing_soc_endpoint_nonfinal():
    """A non-final 5-min slice missing from predict_soc causes the preview to be empty."""
    base = _base_fixture(minutes=60)
    pred = _pred(base, soc_kw=5.0, grid_kw=2.0)
    pred.predict_soc.pop(25, None)
    result = _result(_actions(base))
    rows = build_replay_preview(base, pred, result, _slots(base), scenario=0)
    assert rows == []


def test_rejects_actions_count_mismatch():
    """A result with more or fewer actions than slots produces no rows."""
    base = _base_fixture(minutes=60)
    pred = _pred(base, soc_kw=5.0, grid_kw=2.0)
    result = SimpleNamespace(actions=_actions(base)[:5], status="feasible")
    rows = build_replay_preview(base, pred, result, _slots(base), scenario=0)
    assert rows == []


# ----------------------------------------------------------------------
# render_car_charging_preview tests — BANNER ONLY, no <table> or <details>
# ----------------------------------------------------------------------


def _native_diagnostic(state, peak_w=None, limit_w=None, minutes_over=None):
    """Build a cached diagnostic observation without enforcing anything."""
    attrs = {}
    if peak_w is not None:
        attrs["peak_w"] = peak_w
    if limit_w is not None:
        attrs["limit_w"] = limit_w
    if minutes_over is not None:
        attrs["minutes_over"] = minutes_over
    return {"state": state, "attributes": attrs}


def _good_report(status="candidate", uses_stale=False, native_passed=True):
    """Supply fresh paired data for banner-only assertions."""
    report = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "uses_stale_soc_assumption": uses_stale,
        "preview": {"nominal": [{"x": 1}], "pv10": [{"x": 1}]},
    }
    if native_passed:
        report["native_checks"] = {"nominal": {"passed": True}, "pv10": {"passed": True}}
    else:
        report["native_checks"] = {"nominal": {"passed": False}, "pv10": {"passed": False}}
    return report


def test_render_returns_only_banner_no_table():
    """The renderer never emits a <table> or <details>; only a banner + status text."""
    html = render_car_charging_preview(_good_report(), _native_diagnostic("ok", peak_w=5400, limit_w=6000))
    assert "<table" not in html
    assert "<details" not in html
    assert "<button" not in html
    assert "<input" not in html


def test_render_banner_labels_existing_plan_table_as_native():
    """The banner explicitly labels the existing Plan table as showing the combined candidate."""
    html = render_car_charging_preview(_good_report(), _native_diagnostic("ok", peak_w=5400, limit_w=6000))
    # Banner references the existing Plan table and labels it as the combined-candidate display.
    assert "Plan" in html
    assert "read-only" in html.lower() or "simulation" in html.lower()
    # No claim of control / commands.
    assert "not" in html.lower()


def test_render_simulation_never_executed_text_always_present():
    """The renderer always surfaces 'read-only simulation, not executed' (or equivalent)."""
    # Good report.
    html = render_car_charging_preview(_good_report(), _native_diagnostic("ok", peak_w=5400, limit_w=6000))
    assert "read-only" in html.lower()
    assert "not" in html.lower() and ("executed" in html.lower() or "control" in html.lower() or "commands" in html.lower())


def test_render_stale_soc_warning_maintained():
    """A report with uses_stale_soc_assumption labels the candidate as never valid."""
    html = render_car_charging_preview(_good_report(uses_stale=True), _native_diagnostic("ok", peak_w=5000, limit_w=6000))
    assert "stale SoC" in html
    assert "never valid" in html


def test_render_6kW_exceeded_warning_still_visible():
    """A native state of 'exceeded' over a 6 kW limit surfaces a visible red warning, table-free."""
    html = render_car_charging_preview(
        _good_report(),
        _native_diagnostic("exceeded", peak_w=7200, limit_w=6000, minutes_over=15),
    )
    assert "car-charging-preview-warning" in html
    assert "background" in html or "color" in html
    assert "Native import limit exceeded" in html
    assert "7200" in html and "6000" in html and "15" in html
    # Disclaimer text remains.
    assert "not an enforcement action" in html
    assert "not a service outage claim" in html
    # No table or details.
    assert "<table" not in html
    assert "<details" not in html


def test_render_warning_without_report_still_shown():
    """An exceeded diagnostic with no report still surfaces a visible warning, no table."""
    html = render_car_charging_preview(None, _native_diagnostic("exceeded", peak_w=8000, limit_w=6000))
    assert "car-charging-preview-warning" in html
    assert "8000" in html and "6000" in html
    assert "<table" not in html
    assert "<details" not in html


def test_render_none_inputs_returns_empty_string():
    """report=None and native_diagnostic=None returns an empty render."""
    assert render_car_charging_preview(None, None) == ""


def test_render_stale_report_aged_no_table():
    """A report older than 15 minutes is labelled stale and shows no table."""
    now = datetime.now(timezone.utc)
    report = _good_report()
    report["generated_at"] = (now - timedelta(minutes=20)).isoformat()
    html = render_car_charging_preview(report, _native_diagnostic("ok", peak_w=4000, limit_w=6000), now=now)
    assert "older than 15 minutes" in html or "stale" in html.lower()
    assert "<table" not in html


def test_render_malformed_timestamp_no_fresh_candidate_no_table():
    """A missing/malformed generated_at never presents a fresh candidate table."""
    html = render_car_charging_preview(
        {**_good_report(), "generated_at": "not-a-timestamp"},
        _native_diagnostic("ok", peak_w=4000, limit_w=6000),
    )
    assert "missing or malformed" in html or "no candidate" in html.lower()
    assert "<table" not in html


def test_render_rejected_status_no_table_label():
    """A 'native_rejected' or 'conditional_native_rejected' status labels the section as rejected and shows no table."""
    for status in ("native_rejected", "conditional_native_rejected"):
        html = render_car_charging_preview(
            {**_good_report(status=status), "native_checks": {"nominal": {"passed": False}, "pv10": {"passed": False}}},
            _native_diagnostic("ok", peak_w=5400, limit_w=6000),
        )
        assert "rejected" in html.lower()
        assert "<table" not in html


def test_render_unsupported_status_no_table():
    """An 'unsupported' or 'disabled' status shows no candidate table and surfaces the unavailability text."""
    for status in ("unsupported", "disabled", "no_plan_found", "conditional_no_plan_found", "error", "invalid_input", "departure_passed"):
        html = render_car_charging_preview(
            {"status": status, "generated_at": datetime.now(timezone.utc).isoformat()},
            _native_diagnostic("ok", peak_w=4000, limit_w=6000),
        )
        assert "<table" not in html
        assert "<details" not in html


def test_render_unpaired_preview_no_table():
    """A report whose nominal and pv10 schedules do not match has no candidate table."""
    report = _good_report()
    # Mismatched schedule lengths.
    report["preview"] = {"nominal": [{"x": 1}], "pv10": []}
    html = render_car_charging_preview(report, _native_diagnostic("ok", peak_w=5400, limit_w=6000))
    assert "<table" not in html


def test_render_native_checks_unpassed_no_table():
    """When native checks did not pass, no candidate table is shown."""
    html = render_car_charging_preview(
        _good_report(native_passed=False),
        _native_diagnostic("ok", peak_w=5400, limit_w=6000),
    )
    assert "<table" not in html
    assert "<details" not in html


def test_render_escapes_xss_payloads():
    """Every string field is HTML-escaped; the raw payload never appears in the HTML."""
    payload = "<script>alert(1)</script>"
    html = render_car_charging_preview(
        {
            "status": "candidate<script>alert(1)</script>",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "planning_reference_at": payload,
            "preview": {"nominal": [{"x": 1}], "pv10": [{"x": 1}]},
            "native_checks": {"nominal": {"passed": True}, "pv10": {"passed": True}},
        },
        _native_diagnostic(payload, peak_w=5000, limit_w=6000),
    )
    assert payload not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_render_does_not_duplicate_id():
    """The rendered HTML does not contain a duplicate 'id=' attribute value across elements."""
    import re

    html = render_car_charging_preview(_good_report(), _native_diagnostic("ok", peak_w=5400, limit_w=6000))
    ids = re.findall(r'id="([^"]+)"', html)
    assert len(ids) == len(set(ids)), ids


# ----------------------------------------------------------------------
# get_car_charging_preview_update_js — no independent poll
# ----------------------------------------------------------------------


def test_car_planning_preview_js_returns_empty_no_independent_poll():
    """The preview JS no longer polls; the parent owns the full API poll. The helper returns empty."""
    script = get_car_charging_preview_update_js()
    # No independent fetch / setInterval in this helper.
    assert "fetch(" not in script
    assert "setInterval" not in script
    assert "AbortController" not in script
    # The helper is an empty string (or at most an empty <script> wrapper) by policy.
    assert script in ("", "<script></script>")


# ----------------------------------------------------------------------
# Test registry
# ----------------------------------------------------------------------


def test_plan_forecast_hours_limits_only_visible_rows_and_totals():
    """A 24-hour plan setting must not display the complete 48-hour replay."""
    from car_charging_view import build_car_charging_plan_view
    from tests.test_car_planning_single_table import _good_report, _native_plan

    for window in (5, 15, 30, 60):
        base = _base_fixture(minutes=2880, now_minute=725)
        base.plan_preview_row_minutes = window
        base.forecast_plan_hours = 24
        pred = _pred(base, soc_kw=4, slope_per_step=0.001)
        pred.metric_time = {_grid_stamp(base, minute): 100 + minute for minute in range(0, 2880, 5)}
        pred.final_metric = 2980
        result = _result(_actions(base))
        report = _good_report()
        report["forecast_plan_hours"] = 24
        report["native_checks"] = {name: {"passed": True, "checked_minutes": 2880} for name in ("nominal", "pv10")}
        report["preview"] = {name: build_replay_preview(base, pred, result, _slots(base), scenario=index) for index, name in enumerate(("nominal", "pv10"))}
        original = deepcopy(report)
        assert sum(r["minutes"] for r in report["preview"]["nominal"]) == 2880
        view = build_car_charging_plan_view(_native_plan(), report)
        assert view["rows"][0]["time"] == "2026-01-01T12:05:00+0000"
        assert view["rows"][-1]["plan_interval_end"] == "2026-01-02T12:05:00+0000"
        assert view["totals"]["load_forecast"] == 115.2
        assert view["totals"]["pv_forecast"] == 57.6
        assert view["totals"]["total_cost"] == 15.4
        assert report == original


def test_display_horizon_cannot_hide_a_rejected_replay_tail():
    """Rows outside the visible day still belong to the independently validated replay."""
    from car_charging_view import build_car_charging_plan_view
    from tests.test_car_planning_single_table import _good_report, _native_plan

    base = _base_fixture(minutes=2880)
    base.plan_preview_row_minutes = 5
    base.forecast_plan_hours = 24
    report = _good_report()
    report["forecast_plan_hours"] = 24
    rows = build_replay_preview(base, _pred(base), _result(_actions(base)), _slots(base), scenario=0)
    report["preview"] = {"nominal": rows, "pv10": deepcopy(rows)}
    report["preview"]["pv10"][-1]["battery_modes"] = "invalid"
    view = build_car_charging_plan_view(_native_plan(), report)
    assert view["rows"] == [] and view.get("plan_unavailable") is True


def test_invalid_plan_forecast_hours_never_fabricates_a_visible_day():
    """A malformed plan horizon must not silently become a plausible 24-hour view."""
    from car_charging_view import build_car_charging_plan_view
    from tests.test_car_planning_single_table import _good_report, _native_plan

    for value in (True, 0, -24, 7, 97, "24", float("nan")):
        report = _good_report()
        report["forecast_plan_hours"] = value
        view = build_car_charging_plan_view(_native_plan(), report)
        assert view["rows"] == [] and view.get("plan_unavailable") is True


def run_car_planning_preview_tests(unused_base):
    """Run all bounded preview regressions with isolated synthetic fixtures."""
    from tests.test_car_planning_reference_display import ReferenceDisplayTests

    reference = ReferenceDisplayTests()
    tests = [
        reference.test_reference_statuses_keep_complete_native_trajectories,
        reference.test_selected_reference_survives_publication_and_receipt_projection,
        reference.test_reference_status_does_not_waive_incomplete_prediction,
        reference.test_unknown_reference_status_stays_closed,
        reference.test_missing_preview_is_not_a_hardware_confirmation_failure,
        reference.test_expired_observation_receipt_still_reports_expiry,
        test_partial_bins_and_absolute_clock_alignment,
        test_partial_first_bin_when_now_unaligned,
        test_soc_start_differs_from_end_on_slope,
        test_peak_w_is_upper_bound_never_averaged_down,
        test_pv10_uses_scenario_specific_maps,
        test_mixed_modes_preserved_in_row,
        test_rejects_nonfinite_grid_power,
        test_rejects_incomplete_grid_trace,
        test_rejects_nonfeasible_status,
        test_does_not_mutate_inputs,
        test_total_horizon_2880_yields_97_rows,
        test_rows_follow_the_configured_battery_window,
        test_configured_rows_preserve_partial_edges_and_step_costs,
        test_rows_split_at_charge_target_and_ev_boundaries,
        test_five_minute_full_horizon_is_not_truncated_by_the_preview_limit,
        test_invalid_preview_window_is_not_silently_replaced,
        test_single_mode_mismatch_between_scenarios_is_rejected,
        test_plan_page_and_polling_api_show_single_action_intervals,
        test_plan_forecast_hours_limits_only_visible_rows_and_totals,
        test_display_horizon_cannot_hide_a_rejected_replay_tail,
        test_invalid_plan_forecast_hours_never_fabricates_a_visible_day,
        test_row_minutes_cover_full_horizon_exactly,
        test_house_kwh_sums_5min_kwh_directly,
        test_ev_reserved_uses_full_rate_over_on_minutes,
        test_rejects_unknown_modes,
        test_rejects_missing_soc_endpoint_nonfinal,
        test_rejects_actions_count_mismatch,
        test_render_returns_only_banner_no_table,
        test_render_banner_labels_existing_plan_table_as_native,
        test_render_simulation_never_executed_text_always_present,
        test_render_stale_soc_warning_maintained,
        test_render_6kW_exceeded_warning_still_visible,
        test_render_warning_without_report_still_shown,
        test_render_none_inputs_returns_empty_string,
        test_render_stale_report_aged_no_table,
        test_render_malformed_timestamp_no_fresh_candidate_no_table,
        test_render_rejected_status_no_table_label,
        test_render_unsupported_status_no_table,
        test_render_unpaired_preview_no_table,
        test_render_native_checks_unpassed_no_table,
        test_render_escapes_xss_payloads,
        test_render_does_not_duplicate_id,
        test_car_planning_preview_js_returns_empty_no_independent_poll,
    ]
    for test in tests:
        test()
        print("PASS:", test.__name__)
    return False
