"""Execute the actual compact renderer's read-only strategy and layout contracts."""

import ast
from pathlib import Path
import subprocess
import unittest


class CompactStrategyTests(unittest.TestCase):
    """Separate stable control strategy from five-minute battery-flow direction."""

    def test_compact_strategy_contract(self):
        """Preserve all flow samples, tariff boundaries, cost and source objects."""
        path = Path(__file__).parents[1] / "web_helper.py"
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_plan_renderer_js")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        source = namespace["get_plan_renderer_js"]()
        source = source[source.index("<script>") + 8 : source.rindex("</script>")]
        assertions = r"""
const assert = require('node:assert/strict');
const row = (minute, patch={}) => ({time:`2026-11-16T11:${String(minute).padStart(2,'0')}:00+0100`,
 plan_interval_end:`2026-11-16T11:${String(minute+5).padStart(2,'0')}:00+0100`,
 state:'Demand', plan_action:'Discharge', plan_reason:'Battery supplies demand / export',
 plan_phase:'forecast',show_limit:'',import_rate:19.1,export_rate:0,plan_ev_on_minutes:0,
 pv_forecast:0.1,load_forecast:0.2,car_charging:0,cost_change:0.02,
 soc_percent:70-minute/10,soc_end_percent:69.5-minute/10,soc_change:-0.5,
 plan_steps:[{battery_kw:1,grid_kw:0}],...patch});
const solar=row(5,{plan_action:'Charge',plan_reason:'Solar surplus stored',
 plan_steps:[{battery_kw:-0.8,grid_kw:0}]});
const rows=[row(0),solar,row(10),row(15),row(20),row(25),row(30)];
const before=JSON.stringify(rows);
const grouped=groupCompactPlanRows(rows);
assert.equal(grouped.length,2);
assert.equal(grouped[0].plan_action,'Self-use');
assert.equal(grouped[0].plan_members.length,6);
assert.equal(grouped[0].plan_members[1].plan_action,'Charge');
assert.equal(grouped[0].plan_members[1].plan_reason,'Solar surplus stored');
assert.ok(Math.abs(grouped[0].cost_change-0.12)<1e-9);
assert.ok(Math.abs(grouped[0].load_forecast-1.2)<1e-9);
assert.equal(grouped[0].soc_end_percent,rows[5].soc_end_percent);
assert.equal(JSON.stringify(rows),before);
for(const patch of [{import_rate:8.1},{export_rate:1},{state:'Chrg',show_limit:'35'},
 {plan_ev_on_minutes:5},{plan_phase:'current'},{show_limit:'40'}]){
 assert.equal(groupCompactPlanRows([row(0),row(5,patch)]).length,2);
}
assert.equal(groupCompactPlanRows([row(0),row(10)]).length,2);
assert.equal(compactPlanFlow(solar),'Solar → battery');
assert.equal(compactPlanFlow(row(0)),'Battery → demand');
assert.equal(compactPlanFlow({...row(0),plan_steps:[{battery_kw:0,grid_kw:3}]}),'Grid → demand');
assert.equal(compactPlanFlow({...solar,plan_steps:[{battery_kw:-1,grid_kw:3}]}),'Battery gains · grid imports');
assert.equal(compactPlanFlow(grouped[0]),'Variable self-use');
assert.equal(groupCompactPlanRows([row(0,{state:'Chrg',plan_action:'Hold',show_limit:'35'})])[0].plan_action,'Charge ceiling');
assert.equal(groupCompactPlanRows([row(0,{state:'Unexpected'})])[0].plan_action,'Discharge');
console.log('PASS: compact control strategies, half-hour slots, source labels, exact samples and purity');
"""
        result = subprocess.run(["node", "-e", "global.window={};\n" + source + assertions], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
