"""Presentation-only contracts: no control state may change with the display."""

import subprocess
from copy import deepcopy

from car_charging_view import build_car_charging_plan_view, render_car_charging_preview
from tests.test_car_planning_applied_view import _applied_report, _now
from web_helper import get_plan_renderer_js


def test_partial_visible_text_is_bounded():
    """Inspect the entire HTML tree, not merely text before the first hidden span."""
    from html.parser import HTMLParser

    from tests.test_car_planning_partial import test_real_partial_plan_preserves_target_and_power
    from unit_test import create_predbat

    class VisibleText(HTMLParser):
        """Collect only text outside inline-hidden diagnostic elements."""

        def __init__(self):
            """Initialise visibility and output without touching the report."""
            super().__init__()
            self.hidden = []
            self.text = []

        def handle_starttag(self, tag, attrs):
            """Track explicit hidden wrappers across the whole fragment."""
            if tag not in ("br", "hr", "img", "input", "meta", "link"):
                self.hidden.append("display:none" in dict(attrs).get("style", "").replace(" ", ""))

        def handle_endtag(self, tag):
            """Close the balanced markup emitted by the production banner."""
            if self.hidden:
                self.hidden.pop()

        def handle_data(self, data):
            """Retain current user-facing warnings but exclude diagnostics."""
            if not any(self.hidden):
                self.text.append(data)

    base = create_predbat()
    report = test_real_partial_plan_preserves_target_and_power(base)
    report["uses_stale_soc_assumption"] = True
    html = render_car_charging_preview(report, {"state": "exceeded"}, base.now_utc)
    parser = VisibleText()
    parser.feed(html)
    visible = " ".join(parser.text)
    assert "requested target NOT validated" in visible
    assert "EV SOC unconfirmed" in visible and "native grid limit exceeded" in visible
    assert "Partial charging plan" not in visible and "EV SOC is old" not in visible
    assert len(visible) < 320


def test_limited_replay_detail():
    """The segmented replay retains per-step native power and cumulative cost for display."""
    from types import SimpleNamespace

    from car_charging_replay import replay_discharge_limits
    from const import PREDBAT_MAX_CARS
    from tests.test_car_planning_native import _setup
    from tests.test_car_planning_preview import _grid_stamp
    from unit_test import create_predbat

    base = create_predbat()
    _setup(base, minutes=10)
    base.prediction_kernel_enable = False
    base.car_charging_soc = [0.0] * PREDBAT_MAX_CARS
    base.car_charging_limit_model = [0.0] * PREDBAT_MAX_CARS
    base.car_charging_slots = [[] for _ in range(PREDBAT_MAX_CARS)]
    result = SimpleNamespace(actions=[SimpleNamespace(battery_mode="demand", discharge_limit_kw=None)] * 2)
    slots = [SimpleNamespace(start_min=m, duration_min=5) for m in (0, 5)]
    replay = replay_discharge_limits(base, result, slots, 0)
    assert set(replay.predict_battery_power) == set(replay.predict_grid_power)
    assert len(replay.predict_battery_power) == 2
    assert replay.metric_time[_grid_stamp(base, 0)] == base.cost_today_sofar
    assert replay.metric_time[_grid_stamp(base, 5)] <= replay.final_metric


def test_replay_display_steps():
    """Five-minute flow detail is additive and does not alter replay bins."""
    from car_charging_view import build_replay_preview
    from tests.test_car_planning_preview import _actions, _base_fixture, _grid_stamp, _pred, _result, _slots

    base = _base_fixture(minutes=30)
    pred = _pred(base)
    pred.predict_battery_power = {_grid_stamp(base, m): -0.5 for m in range(0, 30, 5)}
    rows = build_replay_preview(base, pred, _result(_actions(base, modes=("charge",))), _slots(base), 0)
    assert len(rows) == 1
    assert len(rows[0]["display_steps"]) == 6
    assert rows[0]["display_steps"][0]["battery_kw"] == -0.5
    assert rows[0]["display_steps"][0]["grid_kw"] == 2.0


def test_plan_flow_labels():
    """A charging window without energy gain is Hold, not a charging claim."""
    for end, action in ((50.0, "Hold"), (50.001, "Hold"), (50.3, "Charge"), (49.0, "Discharge")):
        report = _applied_report(now=_now())
        for scenario in ("nominal", "pv10"):
            report["preview"][scenario][0].update(battery_modes="charge", battery_end_percent=end, charge_target_percent=35)
        before = deepcopy(report)
        view = build_car_charging_plan_view({}, report, _now())
        assert view["rows"][0]["plan_action"] == action
        assert view["rows"][0]["show_limit"] == "35"
        assert view["rows"][0]["plan_reason"]
        assert report == before


def test_healthy_applied_preview_is_quiet_and_invalid_warning_stays_visible():
    """Healthy copy stays out of the preview block while invalid execution alerts remain."""
    report = _applied_report(now=_now())
    html = render_car_charging_preview(report, None, _now())
    assert "current command only — future forecast under revalidation" not in html
    assert "Active plan" not in html
    assert "<details" not in html
    report["execution_valid_until"] = report["execution_observed_at"]
    html = render_car_charging_preview(report, None, _now())
    assert 'role="alert"' in html


def test_plan_table_javascript():
    """Execute real renderer functions, checking grouping, cost and input purity."""
    source = get_plan_renderer_js()
    source = source[source.index("<script>") + 8 : source.rindex("</script>")]
    script = r"""
const assert = require('node:assert/strict');
global.window = {};
const row = (n, patch={}) => ({time:`2026-11-16T00:${String(n).padStart(2,'0')}:00+0100`,
 plan_interval_end:`2026-11-16T00:${String(n+5).padStart(2,'0')}:00+0100`,
 plan_action:'Hold', plan_reason:'Target reached', state:'Chrg', state_text:'Hold', show_limit:'35',
 import_rate:8.1, export_rate:0, pv_forecast:0.01, load_forecast:0.03, car_charging:0,
 soc_percent:40, soc_end_percent:40, soc_change:0, cost_change:0.02, total_cost:2+n/5*0.02,
 plan_ev_on_minutes:0, ...patch});
const data = {rows:[row(0),row(5),row(10),row(15,{import_rate:11.2})],
 plan_execution:true,currency_symbols:['€','c'],num_cars:1,recorded_cost:2};
const before = JSON.stringify(data);
const grouped = groupPlanRows(data.rows);
assert.equal(grouped.length,2);
assert.equal(grouped[0].plan_interval_end,data.rows[2].plan_interval_end);
assert.ok(Math.abs(grouped[0].cost_change-0.06)<1e-9);
assert.ok(Math.abs(grouped[0].load_forecast-0.09)<1e-9);
for (const patch of [{show_limit:'40'},{plan_action:'Charge'},{plan_ev_on_minutes:5},
 {plan_reason:'Other'},{plan_phase:'current'},{export_rate:1}]) {
 assert.equal(groupPlanRows([row(0),row(5,patch)]).length,2);
}
assert.equal(groupPlanRows([row(0,{plan_ev_on_minutes:5}),row(5,{plan_ev_on_minutes:30})]).length,2);
assert.equal(groupPlanRows([row(0),row(10)]).length,2);
const html = renderPlanTable(data,{},false,false,false);
assert.ok(html.includes('Target %'));
assert.ok(html.includes('Reason'));
assert.ok(html.includes('Recorded so far'));
assert.ok(html.includes('Forecast cost'));
assert.ok(html.includes('Future forecast'));
assert.ok(html.includes('UTC+01:00'));
assert.ok(html.includes('<details'));
assert.ok(!html.includes('NaN'));
assert.equal(JSON.stringify(data),before);
const empty = renderPlanTable({...data,rows:[],execution_blocked:'expired'}, {},false,false,false);
assert.ok(!empty.includes('Forecast cost'));
console.log('PASS: real plan renderer grouping, boundaries, cost, timezone and purity');
"""
    prelude = "global.window = {}; global.document = {createElement: () => ({set textContent(v){this.innerHTML=String(v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}})};\n"
    result = subprocess.run(["node", "-e", prelude + source + script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def run_plan_table_clarity_tests():
    """Run clarity regressions through the existing applied-view test group."""
    from tests.test_plan_compact_strategy import CompactStrategyTests

    failed = False
    for test in (
        test_partial_visible_text_is_bounded,
        test_limited_replay_detail,
        test_replay_display_steps,
        test_plan_flow_labels,
        test_healthy_applied_preview_is_quiet_and_invalid_warning_stays_visible,
        test_plan_table_javascript,
        CompactStrategyTests().test_compact_strategy_contract,
    ):
        try:
            test()
            print("PASS: " + test.__name__)
        except Exception as exc:
            failed = True
            print("ERROR: {}: {}".format(test.__name__, exc))
    return failed
