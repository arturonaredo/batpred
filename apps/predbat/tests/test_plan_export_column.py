"""Execute the actual JavaScript plan renderer to check zero-export columns."""

import ast
import json
from pathlib import Path
import subprocess
import unittest


def render_plan(rates, editable=False, debug=False, unavailable=False):
    """Render a minimal plan using Node and the production renderer body."""
    source = Path(__file__).parents[1] / "web_helper.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_plan_renderer_js")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    script = namespace["get_plan_renderer_js"]()
    start = script.index("    function renderPlanTable(")
    end = script.index("\n    function ", start + 1)
    renderer = script[start:end]
    rows = [{"time": "2026-11-12T14:00:00+0100", "import_rate": 11.2, "export_rate": rate, "state": "Demand", "soc_percent": 50, "pv_forecast": 0.2, "load_forecast": 0.1, "cost_change": 0, "plan_rates_unavailable": unavailable} for rate in rates]
    plan = {"rows": rows, "currency_symbols": ["€", "c"], "num_cars": 0, "totals": {"pv_forecast": 0.6, "load_forecast": 0.3, "total_cost": 0}}
    stubs = "const window = {}; const escapeAttr = String; const formatTimeDisplay = String; const scenarioTitle = () => ''; const reasonTitleAttr = () => ''; const renderRateCell = (rate,color,kind) => `<td id=${kind}>${rate}</td>`; const renderTimeCell = () => '<td></td>'; const renderStateCell = () => '<td colspan=2></td>'; const renderLoadCell = () => '<td></td>'; const renderSocCell = () => '<td></td>';"
    command = stubs + renderer + "\nconsole.log(renderPlanTable(" + json.dumps(plan) + ", {}, " + json.dumps(debug) + ", " + json.dumps(editable) + ", false));"
    result = subprocess.run(["node", "-e", command], check=True, capture_output=True, text=True, timeout=20)
    if "Error rendering" in result.stdout:
        raise AssertionError(result.stdout + result.stderr)
    return result.stdout


class PlanExportColumnTests(unittest.TestCase):
    """Preserve meaningful export rates and aligned totals while removing clutter."""

    def test_all_zero_export_hidden(self):
        """Read-only zero rates have neither a heading nor body cells."""
        html = render_plan([0, 0, 0])
        self.assertNotIn("Export c", html)
        self.assertNotIn("id=export", html)
        self.assertIn("<td></td>" * 5 + "<td bgcolor=#FFFFFF><b>0.6", html)

    def test_nonzero_export_visible(self):
        """Positive and negative rates, however small, remain visible."""
        for rate in (0.01, -0.01, 0.00001):
            html = render_plan([0, rate])
            self.assertIn("Export c", html)
            self.assertEqual(html.count("id=export"), 2)

    def test_debug_zero_export_hidden(self):
        """The debug heading and cells obey the same visibility decision."""
        self.assertNotIn("Export c", render_plan([0], debug=True))

    def test_editable_zero_export_retains_controls(self):
        """Do not hide the only means of editing an otherwise zero rate."""
        self.assertIn("Export c", render_plan([0], editable=True))

    def test_unavailable_rates_not_treated_as_zero(self):
        """An unavailable rate is not evidence of a genuine zero tariff."""
        self.assertIn("Export c", render_plan([0], unavailable=True))

    def test_solar_header_explains_modeled_interval_energy(self):
        """Do not imply that cloud-adjusted buckets are measured or smooth source PV."""
        html = render_plan([0])
        self.assertIn("including configured cloud variability", html)
        self.assertIn("kWh, not kW", html)


def run_plan_export_column_tests(my_predbat):
    """Register executable renderer checks with the standard Predbat runner."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PlanExportColumnTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return not result.wasSuccessful()


if __name__ == "__main__":
    unittest.main()
