"""Economic-pruning regressions for the pure car scheduling solver.

The bounded-beam search at ``PlannerConfig(beam_width=128)`` previously kept a
single cost leader per EV-progress bucket plus the highest-energy neighbour.
Snapshots at horizon end can show ~30 distinct surviving labels and miss
trajectories whose lower stored energy at the end reveals a cheaper plan that
avoids mid-horizon peak-band imports. This module guards the new behaviour:

* the cost leader must still occupy position 0 of its own EV bucket before
  the round-robin selects it (verified against the returned order at the
  layer-by-layer budget);
* when ``preserve_battery`` is set, the head of each bucket must now be
  re-ordered so the layer-by-layer pick spans SoC tiers instead of cost-
  ranked duplicates;
* a synthetic cheap-band shift must show that, given enough history, the
  planner removes peak-band imports rather than skipping them;
* an exact short-horizon oracle agrees with the search on the cost of the
  optimal schedule for tiny (≤ 6 slot) cases.
"""

from copy import deepcopy
from dataclasses import replace
from itertools import product
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from car_charging_solver import (
    BATTERY_DEMAND,
    BATTERY_HOLD,
    BATTERY_CHARGE,
    BatteryConfig,
    EVConfig,
    PlannerConfig,
    SiteConfig,
    Slot,
    STATUS_FEASIBLE,
    _Node,
    _bucket,
    _resolve_action,
    _prune,
    plan,
)


def _battery(**overrides):
    """A small synthetic battery for fast oracle-grade tests."""
    values = dict(
        capacity_kwh=10,
        initial_kwh=5,
        reserve_kwh=1,
        max_charge_kw=3,
        max_discharge_kw=3,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        end_min_kwh=1,
        terminal_value_per_kwh=0,
    )
    values.update(overrides)
    return BatteryConfig(**values)


def _vehicle(**overrides):
    """A small synthetic EV that finishes before the demand window ends."""
    values = dict(
        battery_kwh=10,
        start_soc_kwh=0,
        stop_soc_kwh=1,
        rate_kw=2,
        efficiency=1.0,
        deadline_min=180,
        pause_battery_percent=0,
        restart_battery_percent=0,
        min_off_minutes=0,
        initially_on=False,
        initial_off_minutes=10,
    )
    values.update(overrides)
    return EVConfig(**values)


def _bucket_nodes(nodes):
    """Return the in-place bucket layout used internally by ``_prune``."""
    buckets = {}
    for node in nodes:
        buckets.setdefault(_bucket(node.ev_kwh, 0.05), []).append(node)
    return buckets


class PruneBucketHeadTests(unittest.TestCase):
    """Verify the new within-bucket ordering invariants on synthetic buckets."""

    def test_default_order_keeps_established_cost_layers(self):
        """Ordinary and apply callers do not opt into the shadow economic ordering."""
        nodes = [_Node(2, 10, False, 10, cost=0), _Node(3, 10, False, 10, cost=1), _Node(6, 10, False, 10, cost=2), _Node(9, 10, False, 10, cost=9)]
        self.assertFalse(PlannerConfig().economic_diversity)
        self.assertEqual(_prune(nodes, 4, 0.05, 0, preserve_battery=True), [nodes[0], nodes[3], nodes[1], nodes[2]])
        self.assertEqual(_prune(nodes, 4, 0.05, 0, preserve_battery=True, economic_diversity=True), [nodes[0], nodes[3], nodes[2], nodes[1]])

    def test_max_energy_anchor_outside_cost_head_survives(self):
        """Do not sacrifice the original resilience anchor when it is expensive."""
        nodes = [_Node(2 + i * 0.01, 10, False, 10, cost=i) for i in range(20)]
        high = _Node(9, 10, False, 10, cost=1000)
        kept = _prune(nodes + [high], width=3, bucket=0.05, terminal_value=0, preserve_battery=True, economic_diversity=True)
        self.assertEqual(kept[0], nodes[0])
        self.assertIn(high, kept)
        self.assertEqual(len(kept), 3)

    def test_cost_leader_still_in_returned_layer_zero(self):
        """The cheapest label of its EV bucket is selected first by the layer budget."""
        nodes = [
            _Node(battery_kwh=5.0, ev_kwh=8.0, on=True, off_minutes=0, cost=2.0),
            _Node(battery_kwh=3.0, ev_kwh=8.0, on=True, off_minutes=0, cost=1.0),
            _Node(battery_kwh=7.0, ev_kwh=8.0, on=True, off_minutes=0, cost=1.5),
            _Node(battery_kwh=4.0, ev_kwh=8.0, on=True, off_minutes=0, cost=4.0),
        ]
        kept = _prune(nodes, width=4, bucket=0.05, terminal_value=0.0, preserve_battery=True, economic_diversity=True)
        # Layer 0 across all buckets must include the cost leader (ranking 1.0)
        leader = min(nodes, key=lambda n: (n.cost, n.switches))
        self.assertIn(leader, kept)
        self.assertEqual(kept[0], leader, "cost leader must be picked first across buckets")

    def test_head_includes_max_battery_anchor(self):
        """Even when the cost leader has middling energy, max-battery stays in the head."""
        nodes = [
            _Node(battery_kwh=2.0, ev_kwh=10.0, on=True, off_minutes=0, cost=0.5),
            _Node(battery_kwh=8.0, ev_kwh=10.0, on=True, off_minutes=0, cost=2.0),
            _Node(battery_kwh=4.0, ev_kwh=10.0, on=True, off_minutes=0, cost=1.0),
            _Node(battery_kwh=6.0, ev_kwh=10.0, on=True, off_minutes=0, cost=1.5),
        ]
        kept = _prune(nodes, width=8, bucket=0.05, terminal_value=0.0, preserve_battery=True, economic_diversity=True)
        batteries = sorted(n.battery_kwh for n in kept)
        self.assertIn(8.0, batteries, "max battery anchor should survive in the kept set")

    def test_preserve_battery_false_preserves_ranking_order(self):
        """When hold is allowed the original ranking order is the layer order."""
        nodes = [
            _Node(battery_kwh=2.0, ev_kwh=10.0, on=True, off_minutes=0, cost=0.5),
            _Node(battery_kwh=8.0, ev_kwh=10.0, on=True, off_minutes=0, cost=2.0),
            _Node(battery_kwh=4.0, ev_kwh=10.0, on=True, off_minutes=0, cost=1.0),
        ]
        kept = _prune(nodes, width=8, bucket=0.05, terminal_value=0.0, preserve_battery=False)
        # Cost-sorted order preserved
        costs = [n.cost for n in kept]
        self.assertEqual(costs, sorted(costs))

    def test_does_not_exceed_width(self):
        """The kept list never grows past ``width``."""
        nodes = [_Node(battery_kwh=float(i), ev_kwh=10.0, on=True, off_minutes=0, cost=float(i)) for i in range(50)]
        pruned = _prune(nodes, width=8, bucket=0.05, terminal_value=0.0, preserve_battery=True, economic_diversity=True)
        self.assertEqual(len(pruned), 8)

    def test_layer_two_picks_different_battery_tier(self):
        """With ``width > 1`` and a multi-bucket input, layer 1/2 should not duplicate the cost leader."""
        # Three distinct EV buckets; first picks cheap, second picks high-energy.
        nodes = [
            _Node(battery_kwh=2.0, ev_kwh=10.0, on=True, off_minutes=0, cost=0.5),
            _Node(battery_kwh=4.0, ev_kwh=10.0, on=True, off_minutes=0, cost=1.5),
            _Node(battery_kwh=6.0, ev_kwh=10.0, on=True, off_minutes=0, cost=2.5),
            _Node(battery_kwh=9.0, ev_kwh=12.0, on=True, off_minutes=0, cost=0.6),
            _Node(battery_kwh=8.0, ev_kwh=12.0, on=True, off_minutes=0, cost=1.6),
        ]
        kept = _prune(nodes, width=4, bucket=0.05, terminal_value=0.0, preserve_battery=True, economic_diversity=True)
        battery_set = {(round(n.battery_kwh, 4), round(n.ev_kwh, 4)) for n in kept}
        # At least one of the layer picks must be a different SoC than the leader of its bucket.
        leader_ev10 = min((n for n in nodes if n.ev_kwh == 10.0), key=lambda n: (n.cost, n.switches))
        self.assertIn((round(leader_ev10.battery_kwh, 4), 10.0), battery_set)
        # Layer 1-2 should include the max-battery variant for the first bucket.
        self.assertTrue(any(round(n.battery_kwh, 4) == 6.0 and n.ev_kwh == 10.0 for n in kept))


class EconomicSearchTests(unittest.TestCase):
    """Real search comparisons with the synthetic inputs."""

    @staticmethod
    def _paired_band_inputs(intervening_peak):
        """Use one low-SOC battery, EV target/deadline and cap for paired band cases."""
        slots = []
        for index in range(72):
            if intervening_peak:
                price = 11.2 if index < 12 or 36 <= index < 48 else 19.1 if index < 36 else 8.1
                load_kw = 1.2 if index < 12 or 36 <= index < 48 else 1.8 if index < 36 else 0.3
            else:
                price = 11.2 if index < 48 else 8.1
                load_kw = 1.2 if index < 48 else 0.3
            slots.append(Slot(index, index * 5, 5, load_kw * 5 / 60, 0, price, 0))
        ev = _vehicle(
            battery_kwh=40,
            start_soc_kwh=0.5,
            stop_soc_kwh=2.5,
            rate_kw=6.77,
            efficiency=0.92,
            deadline_min=360,
            availability=tuple(range(48, 72)),
            pause_battery_percent=0,
            restart_battery_percent=0,
            min_off_minutes=10,
        )
        battery = _battery(
            capacity_kwh=32.154,
            initial_kwh=3.215,
            reserve_kwh=3.215,
            max_charge_kw=3,
            max_discharge_kw=3.7,
            charge_efficiency=0.95,
            discharge_efficiency=0.97,
            end_min_kwh=4.0,
            cycle_cost_per_kwh=0.2,
            terminal_value_per_kwh=0.1,
        )
        site = SiteConfig(import_cap_kw=7.5, export_cap_kw=0, load_port_cap_kw=16)
        config = PlannerConfig(beam_width=128, allowed_battery_modes=(BATTERY_DEMAND, BATTERY_CHARGE), economic_diversity=True)
        return tuple(slots), ev, battery, site, config

    @staticmethod
    def _plan_without_early_charge(slots, ev, battery, site, config, cutoff_minute):
        """Solve the matched counterfactual with only pre-cutoff grid-charge transitions removed."""
        original_resolve = _resolve_action

        def resolve_without_early_charge(slot, ev_on, mode, node, battery_config, ev_config, site_config):
            if slot.start_min < cutoff_minute and mode == BATTERY_CHARGE:
                return None
            return original_resolve(slot, ev_on, mode, node, battery_config, ev_config, site_config)

        with patch("car_charging_solver._resolve_action", side_effect=resolve_without_early_charge):
            return plan(slots, ev, battery, site, config)

    def _assert_paired_constraints(self, result, ev, battery, site):
        """Check the paired economic results meet unchanged terminal and power gates."""
        self.assertEqual(result.status, STATUS_FEASIBLE)
        self.assertTrue(result.diagnostics.ev_stop_target_met)
        self.assertGreaterEqual(result.diagnostics.final_ev_soc_kwh, ev.stop_soc_kwh - 1e-9)
        self.assertGreaterEqual(result.diagnostics.final_battery_soc_kwh, battery.end_min_kwh - 1e-9)
        self.assertGreaterEqual(min(result.battery_soc_trace), battery.reserve_kwh - 1e-9)
        peak_import_kw = max(segment.grid_import_kw for action in result.actions for segment in action.segments)
        self.assertLessEqual(peak_import_kw, site.import_cap_kw + 1e-9)

    def test_bounded_search_matches_positive_cost_oracle_and_uses_cheap_tail(self):
        """An actual price transition moves required imports into the cheaper slots."""
        slots = tuple(Slot(i, i * 5, 5, 0.03, 0, 19.1 if i < 3 else 8.1, 0) for i in range(6))
        ev = _vehicle(rate_kw=6.77, stop_soc_kwh=0.8, deadline_min=30, pause_battery_percent=15, restart_battery_percent=25)
        battery = _battery(capacity_kwh=2, initial_kwh=0.9, reserve_kwh=0.2, end_min_kwh=0.2, max_charge_kw=5, max_discharge_kw=8)
        site = SiteConfig(6, 0, 16)
        modes = (BATTERY_DEMAND, BATTERY_CHARGE)
        result = plan(slots, ev, battery, site, PlannerConfig(beam_width=128, allowed_battery_modes=modes, economic_diversity=True))
        oracle = self._oracle_search(slots, ev, battery, site, modes)
        self.assertEqual(result.status, STATUS_FEASIBLE)
        self.assertGreater(oracle, 0)
        self.assertAlmostEqual(result.total_cost, oracle, places=6)
        self.assertEqual(sum(result.import_kwh_trace[:3]), 0)
        self.assertGreater(sum(result.import_kwh_trace[3:]), 0)

    def test_exact_oracle_matches_beam_none_on_short_horizon(self):
        """A 6-slot example: search and brute-force produce the same cost."""
        slots = tuple(Slot(i, i * 5, 5, 1.0 * 5 / 60, 0.0, 10.0 if i < 3 else 2.0, 0.0) for i in range(6))
        ev = _vehicle()
        battery = _battery()
        site = SiteConfig(import_cap_kw=6.0, export_cap_kw=None, load_port_cap_kw=10.0)
        result = plan(slots, ev, battery, site, PlannerConfig(beam_width=None))
        self.assertEqual(result.status, STATUS_FEASIBLE)
        oracle = self._oracle_search(slots, ev, battery, site)
        self.assertIsNotNone(oracle)
        self.assertAlmostEqual(result.total_cost, oracle, places=6)

    def test_cheap_band_steers_plan_to_lower_cost(self):
        """A trailing cheap band steers the plan to push demand into the cheap period.

        With a peak price throughout, the only feasible EV charging is to pre-
        charge the battery at peak prices. Once the prices flip to cheap, the
        planner must use cheap imports even though the peak-only horizon had
        no choice; the total cost must drop as a result.
        """
        slots_peak = tuple(Slot(i, i * 5, 5, 0.0, 0.0, 10.0, 0.0) for i in range(12))
        slots_cheap = tuple(Slot(i, i * 5, 5, 0.0, 0.0, 2.0, 0.0) for i in range(12))
        # Already above the restart threshold; small EV target delivery.
        ev = _vehicle(start_soc_kwh=0.5, stop_soc_kwh=2, deadline_min=60, pause_battery_percent=15, restart_battery_percent=25, min_off_minutes=10)
        # Battery well above pause_battery_percent (15% of capacity = 1.5) so the
        # solver can keep the EV relay on once started.
        battery = _battery(capacity_kwh=10, initial_kwh=3, reserve_kwh=0.5, end_min_kwh=0.5)
        site = SiteConfig(import_cap_kw=6.0, export_cap_kw=None, load_port_cap_kw=None)
        peak = plan(slots_peak, ev, battery, site, PlannerConfig(beam_width=128, allowed_battery_modes=(BATTERY_DEMAND, BATTERY_CHARGE), economic_diversity=True))
        cheap = plan(slots_cheap, ev, battery, site, PlannerConfig(beam_width=128, allowed_battery_modes=(BATTERY_DEMAND, BATTERY_CHARGE), economic_diversity=True))
        self.assertEqual(peak.status, STATUS_FEASIBLE)
        self.assertEqual(cheap.status, STATUS_FEASIBLE)
        # Cheap-band weighted cost is at most half of peak-band weighted cost.
        self.assertLess(cheap.total_cost, peak.total_cost / 2)

    def test_search_removes_peak_band_imports(self):
        """A plan made cheaper by cheap-band pricing must show fewer grid imports than the peak-only version."""
        # Construct a workload that requires grid imports during the EV window.
        slots_peak = tuple(Slot(i, i * 5, 5, 0.0, 0.0, 10.0, 0.0) for i in range(12))
        slots_cheap = tuple(Slot(i, i * 5, 5, 0.0, 0.0, 2.0, 0.0) for i in range(12))
        # Big EV target that requires sustained grid imports.
        ev = _vehicle(start_soc_kwh=0.0, stop_soc_kwh=2.0, deadline_min=60, pause_battery_percent=15, restart_battery_percent=25, min_off_minutes=10)
        battery = _battery(capacity_kwh=2, initial_kwh=0.5, reserve_kwh=0.2, end_min_kwh=0.2, max_charge_kw=2, max_discharge_kw=2)
        site = SiteConfig(import_cap_kw=6.0, export_cap_kw=None, load_port_cap_kw=None)
        peak = plan(slots_peak, ev, battery, site, PlannerConfig(beam_width=128, allowed_battery_modes=(BATTERY_DEMAND, BATTERY_CHARGE), economic_diversity=True))
        cheap = plan(slots_cheap, ev, battery, site, PlannerConfig(beam_width=128, allowed_battery_modes=(BATTERY_DEMAND, BATTERY_CHARGE), economic_diversity=True))
        self.assertEqual(peak.status, STATUS_FEASIBLE)
        self.assertEqual(cheap.status, STATUS_FEASIBLE)
        # Cheap-band weighted cost is at most half of peak-band weighted cost.
        self.assertLess(cheap.total_cost, peak.total_cost / 2)

    def test_waits_for_cheap_band_without_a_profitable_peak(self):
        """An 8.1-cent overnight band beats storing energy at 11.2 when no peak intervenes."""
        slots, ev, battery, site, config = self._paired_band_inputs(intervening_peak=False)
        result = plan(slots, ev, battery, site, config)
        no_precharge = self._plan_without_early_charge(slots, ev, battery, site, config, cutoff_minute=240)

        self._assert_paired_constraints(result, ev, battery, site)
        self._assert_paired_constraints(no_precharge, ev, battery, site)
        self.assertAlmostEqual(result.total_cost, no_precharge.total_cost, places=6)
        self.assertEqual(sum(action.battery_charge_kwh for action in result.actions[:48]), 0)
        self.assertGreater(sum(action.battery_charge_kwh for action in result.actions[48:]), 0)

    def test_peak_avoiding_charge_then_cheap_band_recharge(self):
        """Buying at 11.2 beats 19.1 peak imports; 8.1 restores the terminal SOC."""
        slots, ev, battery, site, config = self._paired_band_inputs(intervening_peak=True)
        result = plan(slots, ev, battery, site, config)
        wait_for_cheap = self._plan_without_early_charge(slots, ev, battery, site, config, cutoff_minute=240)

        self._assert_paired_constraints(result, ev, battery, site)
        self._assert_paired_constraints(wait_for_cheap, ev, battery, site)
        self.assertLess(result.total_cost, wait_for_cheap.total_cost)
        self.assertGreater(sum(action.battery_charge_kwh for action in result.actions[:12]), 0)
        self.assertGreater(sum(action.battery_discharge_kwh for action in result.actions[12:36]), 0)
        self.assertGreater(sum(action.battery_charge_kwh for action in result.actions[48:]), 0)

    def test_apply_six_kw_waits_for_cheap_band_without_a_profitable_peak(self):
        """Apply ordering with grid support waits when early storage has no price benefit."""
        slots, ev, battery, site, config = self._paired_band_inputs(intervening_peak=False)
        site = replace(site, import_cap_kw=6, charge_import_cap_kw=6)
        config = replace(config, economic_diversity=False)
        result = plan(slots, ev, battery, site, config)
        wait = self._plan_without_early_charge(slots, ev, battery, site, config, cutoff_minute=240)

        self._assert_paired_constraints(result, ev, battery, site)
        self._assert_paired_constraints(wait, ev, battery, site)
        self.assertAlmostEqual(result.total_cost, wait.total_cost, places=6)
        self.assertEqual(sum(action.battery_charge_kwh for action in result.actions[:48]), 0)
        self.assertGreater(sum(action.battery_charge_kwh for action in result.actions[48:]), 0)

    def test_apply_six_kw_prepares_for_peak_then_recharges_overnight(self):
        """The live apply ordering can buy before a peak without filling all expensive slots."""
        slots, ev, battery, site, config = self._paired_band_inputs(intervening_peak=True)
        site = replace(site, import_cap_kw=6, charge_import_cap_kw=6)
        config = replace(config, economic_diversity=False)
        result = plan(slots, ev, battery, site, config)
        wait = self._plan_without_early_charge(slots, ev, battery, site, config, cutoff_minute=240)

        self._assert_paired_constraints(result, ev, battery, site)
        self._assert_paired_constraints(wait, ev, battery, site)
        self.assertLess(result.total_cost, wait.total_cost)
        self.assertGreater(sum(action.battery_charge_kwh for action in result.actions[:12]), 0)
        self.assertGreater(sum(action.battery_discharge_kwh for action in result.actions[12:36]), 0)
        self.assertGreater(sum(action.battery_charge_kwh for action in result.actions[48:]), 0)

    def test_apply_six_kw_feasible_initial_demand_can_cost_more_than_charging_now(self):
        """Deferring the last cheap step is feasible but loses a real peak-avoidance saving."""
        slots, ev, battery, site, config = self._paired_band_inputs(intervening_peak=True)
        slots = tuple(replace(slot, import_price=19.1) if 1 <= index < 12 else slot for index, slot in enumerate(slots))
        site = replace(site, import_cap_kw=6, charge_import_cap_kw=6)
        config = replace(config, economic_diversity=False)
        ordinary = plan(slots, ev, battery, site, config)
        demand_first = plan(slots, ev, battery, site, replace(config, initial_battery_mode=BATTERY_DEMAND))

        self._assert_paired_constraints(ordinary, ev, battery, site)
        self._assert_paired_constraints(demand_first, ev, battery, site)
        self.assertEqual(ordinary.actions[0].battery_mode, BATTERY_CHARGE)
        self.assertEqual(demand_first.actions[0].battery_mode, BATTERY_DEMAND)
        self.assertLess(ordinary.total_cost, demand_first.total_cost)
        self.assertAlmostEqual(ordinary.diagnostics.final_ev_soc_kwh, demand_first.diagnostics.final_ev_soc_kwh)
        self.assertAlmostEqual(ordinary.diagnostics.final_battery_soc_kwh, demand_first.diagnostics.final_battery_soc_kwh)

    @staticmethod
    def _oracle_search(slots, ev, battery, site, modes=(BATTERY_DEMAND, BATTERY_HOLD, BATTERY_CHARGE)):
        """Enumerate choices without pruning, using the same gated segmented physics."""
        n = len(slots)
        if n > 6:
            return None

        best_cost = None
        for choices in product(product(modes, (False, True)), repeat=n):
            node = _Node(battery.initial_kwh, ev.start_soc_kwh, ev.initially_on, ev.initial_off_minutes)
            cost = 0.0
            ok = True
            for slot, (mode, turn_on) in zip(slots, choices):
                action = _resolve_action(slot, turn_on, mode, node, battery, ev, site)
                if action is None:
                    ok = False
                    break
                on = action.ev_on_minutes >= slot.duration_min - 1e-9 and action.ev_on_minutes > 0
                off = slot.duration_min - action.ev_on_minutes if action.ev_on_minutes else (0 if node.on else node.off_minutes) + slot.duration_min
                node = _Node(action.battery_soc_kwh_end, action.ev_soc_kwh_end, on, 0 if on else min(off, ev.min_off_minutes))
                cost += action.slot_cost
            if not ok:
                continue
            if node.ev_kwh < ev.stop_soc_kwh - 1e-9 or node.battery_kwh < battery.end_min_kwh - 1e-9:
                continue
            ranking = cost - battery.terminal_value_per_kwh * node.battery_kwh
            if best_cost is None or ranking < best_cost:
                best_cost = ranking
        return best_cost


def test_generated_reference_search_budget_includes_unrestricted_and_best_continuous(base):
    """Limit expensive low-rate replays to configured, 512, and the two cheapest continuous plans."""
    from car_charging_plan import _search_joint_cost_candidates

    horizon = base.forecast_minutes
    slots = [Slot(index, index * 5, 5, 0.1, 0, 10, 0) for index in range(7)]
    options = {"beam": 128, "mode": "apply", "ev_charging_priority": "cost", "control_model": "deye_solarman"}
    schedules = {}
    checks_by_label = {}
    windows = ([], [], [])

    def make_candidate(label, on_index):
        """Create a unique EV reservation without depending on solver physics."""
        actions = tuple(
            SimpleNamespace(
                slot_index=slot.index,
                battery_mode="demand",
                ev_action="on" if slot.index == on_index else "off",
                discharge_limit_kw=None,
            )
            for slot in slots
        )
        result = SimpleNamespace(status="feasible", total_cost=label, actions=actions, label=label)
        schedules[label] = tuple((action.slot_index, action.ev_action) for action in actions)
        return result

    def safe_checks(cost):
        """Return complete paired evidence with a known native cost for ordering."""
        return {
            name: {
                "passed": True,
                "checked_minutes": horizon,
                "horizon_energy_cost": cost,
                "battery_cycle_kwh": 0,
                "final_battery_kwh": 5,
                "final_ev_kwh": 1,
                "validated_ev_target_kwh": 1,
            }
            for name in ("nominal", "pv10")
        }

    configured = make_candidate("configured", 0)
    duplicate_configured = make_candidate("continuous_duplicate", 0)
    continuous = [make_candidate("continuous_{}".format(index), index + 1) for index in range(5)]
    unrestricted = make_candidate("unrestricted_512", 6)
    checks_by_label["configured"] = safe_checks(60)
    checks_by_label[duplicate_configured.label] = safe_checks(60)
    checks_by_label["unrestricted_512"] = safe_checks(55)
    continuous_costs = (50, 40, 30, 20, 10)
    continuous_attempts = []
    for result, cost in zip(continuous, continuous_costs):
        checks = safe_checks(cost)
        checks_by_label[result.label] = checks
        continuous_attempts.append(
            {
                "_candidate": (result, slots, windows, checks, options, "continuous_block"),
                "solver_status": "feasible",
            }
        )

    refined_checks = safe_checks(55)
    checks_by_label[unrestricted.label] = refined_checks
    duplicate_checks = checks_by_label[duplicate_configured.label]
    failed_reference_checks = {name: {"passed": False, "checked_minutes": horizon, "reasons": ["site_import_exceeded"]} for name in ("nominal", "pv10")}
    reference_calls = []
    generated_calls = []
    ordinary_reference = SimpleNamespace(status="ordinary_reference", actions=())
    continuous_attempts.insert(
        0,
        {
            "_candidate": (duplicate_configured, slots, windows, duplicate_checks, options, "continuous_block"),
            "solver_status": "feasible",
            "test_label": "duplicate_configured",
        },
    )

    def replay_candidate(_base, _trial, result, _slots):
        """Return the per-candidate paired evidence used to rank continuous schedules."""
        return windows, checks_by_label[result.label]

    def replay_ordinary(_base, _trial, _snapshot, _candidate, _slots, ev_windows):
        """Prove that every candidate still receives its ordinary native reference replay."""
        reference_calls.append(deepcopy(ev_windows))
        return ordinary_reference, (deepcopy(ev_windows), [], []), deepcopy(failed_reference_checks)

    def replay_generated(_base, _trial, schedule, _checks, _slots):
        """Accept the first configured reference, then prove the selected 512/continuous sources still run."""
        fixed_schedule = tuple(schedule)
        generated_calls.append(fixed_schedule)
        if fixed_schedule == schedules["configured"]:
            result = SimpleNamespace(
                status="test_generated_safe_reference",
                actions=tuple(SimpleNamespace(slot_index=index, battery_mode="demand", ev_action=ev_action, discharge_limit_kw=None) for index, ev_action in fixed_schedule),
            )
            return {
                "status": "paired_safe",
                "result": result,
                "slots": _slots,
                "windows": windows,
                "checks": safe_checks(100),
                "costs": {"nominal": 100, "pv10": 100},
                "provenance": {"test": "configured-safe"},
            }
        return {"status": "unsafe", "reason": "test_generated_reference_unsafe", "provenance": {}}

    with (
        patch("car_charging_plan.build_candidate", return_value=(unrestricted, slots)),
        patch("car_charging_plan._replay_candidate", side_effect=replay_candidate),
        patch("car_charging_household._replay_native_joint_plan", side_effect=replay_ordinary),
        patch("car_charging_household._generate_low_rates_native_reference", side_effect=replay_generated),
        patch("car_charging_execute.economic_input_fingerprint", return_value={"test": "same-input"}),
    ):
        selected, attempts = _search_joint_cost_candidates(
            base,
            options,
            {"plan_valid": True, "charge_windows": [], "charge_limits": []},
            (configured, slots),
            continuous_attempts,
        )

    expected_schedules = {
        schedules["configured"],
        schedules["unrestricted_512"],
        schedules["continuous_3"],
        schedules["continuous_4"],
    }
    assert selected["source"] == "configured_beam", selected
    assert selected["economic_comparison"]["positive_improvement"] is True
    assert len(reference_calls) == 8, "ordinary native reference replay must remain unbudgeted"
    assert len(generated_calls) <= 4, generated_calls
    assert len(generated_calls) == 4, generated_calls
    assert set(generated_calls) == expected_schedules, generated_calls
    duplicate_attempt = next(attempt for attempt in attempts if attempt.get("test_label") == "duplicate_configured")
    assert duplicate_attempt["generated_reference_passed"] is True, duplicate_attempt
    budgeted = [attempt for attempt in attempts if attempt.get("generated_reference_status") == "bounded_generated_reference_schedule_budget"]
    assert len(budgeted) == 3, attempts
    assert all(attempt["economic_rejection"] == "bounded_generated_reference_schedule_budget" for attempt in budgeted)


def test_native_reference_alternative_reports_selected_plan_switches(base):
    """The selected native alternative, not its rejected proposal, owns candidate switch metadata."""
    from car_charging_plan import _search_joint_cost_candidates

    slots = [Slot(index, index * 5, 5, 0.1, 0, 10, 0) for index in range(3)]
    candidate = SimpleNamespace(
        status="feasible",
        total_cost=0,
        actions=tuple(
            SimpleNamespace(
                slot_index=index,
                battery_mode="charge" if index in (0, 2) else "demand",
                ev_action="off",
                discharge_limit_kw=None,
            )
            for index in range(3)
        ),
    )
    reference = SimpleNamespace(
        status="ordinary_native_joint_reference",
        actions=tuple(SimpleNamespace(slot_index=index, battery_mode="demand", ev_action="off", discharge_limit_kw=None) for index in range(3)),
    )
    checks = {
        name: {
            "passed": True,
            "checked_minutes": base.forecast_minutes,
            "reasons": [],
            "horizon_energy_cost": 10,
            "battery_cycle_kwh": 0,
            "final_battery_kwh": base.soc_kw,
            "final_ev_kwh": 0,
            "validated_ev_target_kwh": 0,
        }
        for name in ("nominal", "pv10")
    }
    windows = ([], [], [])
    options = {"beam": 512, "mode": "apply", "ev_charging_priority": "cost", "control_model": "deye_solarman"}

    with (
        patch("car_charging_plan._replay_candidate", return_value=(windows, checks)),
        patch("car_charging_household._replay_native_joint_plan", return_value=(reference, windows, checks)),
    ):
        selected, _attempts = _search_joint_cost_candidates(
            base,
            options,
            {"plan_valid": True, "charge_windows": [], "charge_limits": []},
            (candidate, slots),
            [],
        )

    comparison = selected["economic_comparison"]
    assert selected["source"] == "native_reference_alternative"
    assert comparison["proposed_candidate_battery_mode_switches"] == 2
    assert comparison["candidate_battery_mode_switches"] == 0
    assert comparison["reference_battery_mode_switches"] == 0
    assert comparison["positive_improvement"] is False
    assert "generated_reference_schedule_cap_4" in comparison["search_scope"]


def run_car_planning_economic_tests(base):
    """Run the economic regressions through Predbat's registered harness."""
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(test) for test in (PruneBucketHeadTests, EconomicSearchTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return True
    test_generated_reference_search_budget_includes_unrestricted_and_best_continuous(base)
    test_native_reference_alternative_reports_selected_plan_switches(base)
    return False


if __name__ == "__main__":
    unittest.main(verbosity=2)
