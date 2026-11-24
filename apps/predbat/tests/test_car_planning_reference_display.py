"""Reference plans must retain their paired display without inventing hardware authority."""

from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
import unittest

from car_charging_plan import _publish_report
from car_charging_view import _applied_block_reason, build_car_charging_plan_view, build_replay_preview
from tests.test_car_planning_applied_view import _applied_report
from tests.test_car_planning_preview import _actions, _base_fixture, _grid_stamp, _pred, _result, _slots


class ReferenceDisplayTests(unittest.TestCase):
    """Use the production replay builder, publisher and applied-view gates together."""

    def test_reference_statuses_keep_complete_native_trajectories(self):
        """Only the known, real native-reference provenance states join feasible."""
        base = _base_fixture(minutes=60)
        prediction = _pred(base)
        baseline = build_replay_preview(base, prediction, _result(_actions(base)), _slots(base), 0)
        self.assertTrue(baseline)
        for status in ("ordinary_native_joint_reference", "generated_low_rates_native_reference", "ordinary_native_household_plan"):
            with self.subTest(status=status):
                result = _result(_actions(base), status=status)
                before = deepcopy(result)
                rows = build_replay_preview(base, prediction, result, _slots(base), 0)
                self.assertEqual(rows, baseline)
                self.assertEqual(vars(result), vars(before))

    def test_selected_reference_survives_publication_and_receipt_projection(self):
        """Reproduce a safe selected reference whose fresh receipt lost all display rows."""
        base = _base_fixture(minutes=60)
        report = _applied_report(now=base.now_utc)
        report.pop("preview", None)
        report["hardware_confirmed"] = False
        report["execution_reason"] = "inverter_observed_not_hardware_confirmed"
        result = _result(_actions(base), status="generated_low_rates_native_reference")
        report["native_checks"] = {name: {"passed": True, "checked_minutes": 60, "_preview_rows": build_replay_preview(base, _pred(base), result, _slots(base), scenario)} for name, scenario in (("nominal", 0), ("pv10", 1))}
        owner = SimpleNamespace()
        _publish_report(owner, report, False)
        self.assertEqual(set(owner.car_charging_report.get("preview", {})), {"nominal", "pv10"})
        before = deepcopy(owner.car_charging_report)
        view = build_car_charging_plan_view({}, owner.car_charging_report, base.now_utc)
        self.assertTrue(view["plan_execution"])
        self.assertFalse(view["plan_simulation"])
        self.assertTrue(view["rows"])
        self.assertFalse(owner.car_charging_report["hardware_confirmed"])
        self.assertEqual(owner.car_charging_report, before)

    def test_reference_status_does_not_waive_incomplete_prediction(self):
        """A recognised reference still needs every native grid sample."""
        base = _base_fixture(minutes=60)
        prediction = _pred(base)
        prediction.predict_grid_power.pop(_grid_stamp(base, 25))
        result = _result(_actions(base), status="generated_low_rates_native_reference")
        self.assertEqual(build_replay_preview(base, prediction, result, _slots(base), 0), [])

    def test_unknown_reference_status_stays_closed(self):
        """Do not accept arbitrary labels as successful replay evidence."""
        base = _base_fixture(minutes=60)
        for status in ("unsafe", "native_rejected", "generated_reference_failed", "unknown", None, [], {}):
            result = _result(_actions(base), status=status)
            self.assertEqual(build_replay_preview(base, _pred(base), result, _slots(base), 0), [])

    def test_missing_preview_is_not_a_hardware_confirmation_failure(self):
        """An observation-only disclaimer must not mask the actual display failure."""
        base = _base_fixture(minutes=60)
        report = _applied_report(now=base.now_utc)
        report["execution_reason"] = "inverter_observed_not_hardware_confirmed"
        report.pop("preview")
        self.assertEqual(_applied_block_reason(report, base.now_utc), "paired replay preview missing or invalid")
        self.assertEqual(build_car_charging_plan_view({}, report, base.now_utc)["rows"], [])

    def test_expired_observation_receipt_still_reports_expiry(self):
        """The observation label cannot hide an expired receipt or revive its rows."""
        base = _base_fixture(minutes=60)
        report = _applied_report(now=base.now_utc, observed_at=base.now_utc - timedelta(minutes=6), interval_seconds=300)
        report["execution_reason"] = "inverter_observed_not_hardware_confirmed"
        self.assertEqual(_applied_block_reason(report, base.now_utc), "execution receipt expired")
        self.assertEqual(build_car_charging_plan_view({}, report, base.now_utc)["rows"], [])


if __name__ == "__main__":
    unittest.main()
