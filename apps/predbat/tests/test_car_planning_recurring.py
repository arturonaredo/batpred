"""Standing EV intent through real Storage methods and the native control seams."""

from tests.car_planning_fixtures import car_settings

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from car_charging_recurring import next_departure, resolve_recurring_session
from tests.test_car_planning_actual import _set_entity
from tests.test_car_planning_session import session_inputs


class RecurringStorage:
    """Async Storage contract with detached data, deliberately without a public data dict."""

    def __init__(self):
        """Keep saved records private so only the production contract can persist them."""
        self.records = {}
        self.fail_save = False

    async def load(self, module, filename):
        """Return a detached durable record across fresh runtime instances."""
        return deepcopy(self.records.get((module, filename)))

    async def load_checked(self, module, filename):
        """Match strict reads; the filesystem integration case separately exercises corruption."""
        return await self.load(module, filename)

    async def save(self, module, filename, data, **kwargs):
        """Require explicit JSON/no-expiry persistence and model rejected writes."""
        assert (module, filename) == ("car_charging", "recurring_session")
        assert kwargs == {"format": "json", "expiry": None}
        if self.fail_save:
            return False
        self.records[(module, filename)] = deepcopy(data)
        return True


def recurring_inputs(base):
    """Use the actual template fixture with a standing policy, not a renewed confirmation."""
    fixture = session_inputs(base)
    settings = car_settings(base)
    settings.pop("confirmed_session")
    settings.pop("session")
    settings.update(apply_verified=False, execution_authorised=True, ev_charging_priority="asap")
    settings["recurring_session"] = {"enabled": True, "policy_authorised_at": (base.now_utc - timedelta(minutes=30)).isoformat(), "energy_entity": "sensor.test_ev_energy", "power_entity": "sensor.test_ev_power"}
    base.args["timezone"] = "UTC"
    base.car_charging_plan_time = [(base.now_utc + timedelta(minutes=30)).astimezone(timezone.utc).strftime("%H:%M:%S")]
    store = RecurringStorage()
    base.components = SimpleNamespace(get_component=lambda name: store if name == "storage" else None)
    return fixture, store


def test_recurring_native_ready_and_dst(base):
    """Daily deadlines use the chosen native clock and real 23/25-hour calendar days."""
    recurring_inputs(base)
    base.args["timezone"] = "Europe/Madrid"
    base.car_charging_plan_time = ["07:30:00"]
    zone = ZoneInfo("Europe/Madrid")
    for date, hours in (("2026-03-28T07:30:00+01:00", 23), ("2026-10-24T07:30:00+02:00", 25)):
        now = datetime.fromisoformat(date)
        deadline = next_departure(base, now)
        assert (deadline - now.astimezone(timezone.utc)).total_seconds() == hours * 3600
        assert deadline.astimezone(zone).strftime("%H:%M:%S") == "07:30:00"
    base.car_charging_plan_time = ["08:15:00"]
    assert next_departure(base, now).astimezone(zone).strftime("%H:%M:%S") == "08:15:00"


def test_recurring_stored_baseline_and_real_timestamps(base):
    """An idle old energy timestamp is valid; timestamps and user config remain untouched."""
    (_, writes, _, _), store = recurring_inputs(base)
    before = deepcopy(base.args)
    entities = deepcopy(base.ha_interface.dummy_items)
    first = resolve_recurring_session(base, now=base.now_utc)
    assert first["status"] == "ready", first
    assert first["observation"]["ev_soc_is_estimate"]
    assert first["observation"]["soc_observation_age_minutes"] == 720
    assert "operator_confirmed_at" not in first["observation"]
    assert base.args == before and base.ha_interface.dummy_items == entities and not writes
    saved = store.records[("car_charging", "recurring_session")]["session"]
    assert saved["meter_baseline"] == 100
    later = base.now_utc + timedelta(minutes=20)
    _set_entity(base, "sensor.test_ev_energy", "101", last_updated=later, last_changed=later, attributes={"unit_of_measurement": "kWh"})
    value = resolve_recurring_session(base, now=later)
    assert value["identity"] == first["identity"] and abs(value["observation"]["ev_soc"] - base.car_charging_loss) < 1e-9, value
    assert store.records[("car_charging", "recurring_session")]["session"]["meter_baseline"] == 100


def test_recurring_absence_and_reconnection(base):
    """Every absent entry point avoids SOC/meter and a real return needs no operator edit."""
    recurring_inputs(base)
    settings = car_settings(base)
    sink = cache_sink()
    entity = settings["connection_entity"]
    old = cache_event(sink, entity, "connected", base.now_utc - timedelta(hours=1))
    base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
    first = resolve_recurring_session(base, now=base.now_utc)
    assert first["status"] == "ready"
    later = base.now_utc + timedelta(minutes=1)
    old = cache_event(sink, entity, "disconnected", later, old)
    base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
    getter = base.get_state_wrapper

    def guard(*args, **kwargs):
        """Make a forbidden unavailable-vehicle energy read a hard assertion."""
        entity = kwargs.get("entity_id", args[0] if args else None)
        assert entity not in (settings["soc_entity"], "sensor.test_ev_energy", "sensor.test_ev_power"), entity
        return getter(*args, **kwargs)

    with patch.object(base, "get_state_wrapper", side_effect=guard):
        assert resolve_recurring_session(base, now=base.now_utc)["reason"] == "connection_not_confirmed"
    later = base.now_utc + timedelta(minutes=2)
    cache_event(sink, entity, "connected", later, old)
    base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
    second = resolve_recurring_session(base, now=later)
    assert second["status"] == "ready" and second["identity"] != first["identity"], second
    assert settings["recurring_session"]["policy_authorised_at"] == first["observation"]["policy_authorised_at"]


def test_recurring_counter_failure_survives_restart_and_rollover(base):
    """Neither restart, clock advancement nor plausible-looking data erases a contradiction."""
    _, store = recurring_inputs(base)
    assert resolve_recurring_session(base, now=base.now_utc)["status"] == "ready"
    _set_entity(base, "sensor.test_ev_energy", "99", attributes={"unit_of_measurement": "kWh"})
    value = resolve_recurring_session(base, now=base.now_utc)
    assert value["reason"] == "recurring_meter_contradiction", value
    _set_entity(base, "sensor.test_ev_energy", "100", attributes={"unit_of_measurement": "kWh"})
    restored = RecurringStorage()
    restored.records = deepcopy(store.records)
    base.components = SimpleNamespace(get_component=lambda name: restored)
    for hours in (1, 25, 49):
        assert resolve_recurring_session(base, now=base.now_utc + timedelta(hours=hours))["reason"] == "recurring_meter_contradiction"


def test_recurring_storage_and_unauthorised_gates(base):
    """Missing/failed/corrupt storage and a disabled policy cannot grant authority."""
    _, store = recurring_inputs(base)
    store.fail_save = True
    assert resolve_recurring_session(base, now=base.now_utc)["reason"] == "recurring_storage_write_failed"
    store.fail_save = False
    store.records[("car_charging", "recurring_session")] = {"version": 1}
    assert resolve_recurring_session(base, now=base.now_utc)["reason"] == "recurring_storage_corrupt"
    car_settings(base)["recurring_session"]["enabled"] = False
    with patch.object(base, "get_state_wrapper", side_effect=AssertionError("Disabled policy read a sensor")):
        assert resolve_recurring_session(base)["status"] == "disabled"
    car_settings(base)["recurring_session"]["enabled"] = True
    car_settings(base)["execution_authorised"] = False
    assert resolve_recurring_session(base, now=base.now_utc)["reason"] == "recurring_not_authorised"


def test_recurring_rollover_preserves_energy_and_evidence(base):
    """Calendar rollover changes only the deadline, not the physical meter/SOC baseline."""
    _, store = recurring_inputs(base)
    first = resolve_recurring_session(base, now=base.now_utc)
    later = base.now_utc + timedelta(minutes=20)
    _set_entity(base, "sensor.test_ev_energy", "101", last_updated=later, last_changed=later, attributes={"unit_of_measurement": "kWh"})
    progress = resolve_recurring_session(base, now=later)
    rollover = datetime.fromisoformat(first["session"]["departure"]) + timedelta(minutes=1)
    current = resolve_recurring_session(base, now=rollover)
    assert current["status"] == "ready" and current["identity"] == first["identity"], current
    assert current["session"]["departure"] != first["session"]["departure"]
    assert current["observation"]["ev_soc"] == progress["observation"]["ev_soc"]
    assert current["latest_deadline"]["met"] is None
    assert current["latest_deadline"]["last_evidence"]["shortfall_kwh"] > 0
    assert store.records[("car_charging", "recurring_session")]["session"]["meter_baseline"] == 100


def test_recurring_native_planning_and_execution(base):
    """The real candidate, paired replay, dispatch and lease all consume standing evidence."""
    from car_charging_execute import execute_car_charging_plan, publish_car_charging_slot
    from car_charging_plan import evaluate_car_charging

    (_, writes, _, _), _ = recurring_inputs(base)
    report = evaluate_car_charging(base, publish=False)
    assert report["status"] == "candidate", report
    assert all(check["passed"] for check in report["native_checks"].values()) and not writes
    assert execute_car_charging_plan(base)[0] == "Charging", base.car_charging_report
    assert base.car_charging_report["status"] == "applied"
    assert not any(service.startswith("switch") for service, _ in writes)
    assert publish_car_charging_slot(base) is True
    assert base.dashboard_values["binary_sensor." + base.prefix + "_car_charging_slot"]["state"] == "on"


def test_recurring_fetch_does_not_read_vehicle_soc(base):
    """The upstream native fetch cannot bypass combined availability-first SOC ownership."""
    recurring_inputs(base)
    settings = car_settings(base)
    _set_entity(base, settings["home_entity"], "not_home")
    original = base.get_arg

    def guard(name, *args, **kwargs):
        """Reject the indirect SOC read before it can reach even the HA cache."""
        assert name != "car_charging_soc", "Native fetch bypassed availability"
        return original(name, *args, **kwargs)

    with patch.object(base, "get_arg", side_effect=guard):
        base.fetch_sensor_data_cars()


def test_recurring_invalid_measurements_never_become_baselines(base):
    """Missing, future, non-finite, wrongly scaled or invalid observations fail closed."""
    for fault in ("future_home", "future_soc", "future_meter", "invalid_soc", "nan_meter", "power_units", "overpower", "missing_permission"):
        recurring_inputs(base)
        settings = car_settings(base)
        if fault == "future_home":
            _set_entity(base, settings["home_entity"], "home", last_updated=base.now_utc + timedelta(seconds=1))
        elif fault == "future_soc":
            _set_entity(base, settings["soc_entity"], 0, last_updated=base.now_utc + timedelta(seconds=1), attributes={"retrievalstatus": "VALID"})
        elif fault == "future_meter":
            _set_entity(base, "sensor.test_ev_energy", "100", last_updated=base.now_utc + timedelta(seconds=1), attributes={"unit_of_measurement": "kWh"})
        elif fault == "invalid_soc":
            _set_entity(base, settings["soc_entity"], 0, attributes={"retrievalstatus": "INVALID"})
        elif fault == "nan_meter":
            _set_entity(base, "sensor.test_ev_energy", "nan", attributes={"unit_of_measurement": "kWh"})
        elif fault == "power_units":
            _set_entity(base, "sensor.test_ev_power", "0", attributes={"unit_of_measurement": "A"})
        elif fault == "overpower":
            _set_entity(base, "sensor.test_ev_power", "99", attributes={"unit_of_measurement": "kW"})
        else:
            base.ha_interface.dummy_items.pop(settings["permission_entity"])
        value = resolve_recurring_session(base, now=base.now_utc)
        assert value["status"] == "blocked", (fault, value)


def test_recurring_stale_full_is_not_observed_completion(base):
    """A retained full reading after a physical gap cannot claim a newly met target."""
    recurring_inputs(base)
    settings = car_settings(base)
    _set_entity(base, settings["soc_entity"], base.car_charging_limit[0], last_changed=base.now_utc - timedelta(hours=12), last_updated=base.now_utc - timedelta(hours=12), attributes={"retrievalstatus": "VALID"})
    value = resolve_recurring_session(base, now=base.now_utc)
    assert value["reason"] == "recurring_stale_full_soc", value


def test_recurring_cannot_escape_legacy_contradiction(base):
    """Changing policy type is not a shortcut around an unresolved SOC contradiction."""
    _, store = recurring_inputs(base)
    store.records[("car_charging", "confirmed_session_guard")] = {"invalid": True}
    assert resolve_recurring_session(base, now=base.now_utc)["reason"] == "recurring_legacy_session_requires_reconciliation"


def test_recurring_absent_deadline_is_not_hidden(base):
    """A missed/unverifiable deadline remains visible even when the EV is absent at rollover."""
    recurring_inputs(base)
    first = resolve_recurring_session(base, now=base.now_utc)
    settings = car_settings(base)
    _set_entity(base, settings["home_entity"], "not_home")
    later = datetime.fromisoformat(first["session"]["departure"]) + timedelta(minutes=1)
    value = resolve_recurring_session(base, now=later)
    assert value["reason"] == "ev_not_home"
    assert value["latest_deadline"]["departure"] == first["session"]["departure"]
    assert value["latest_deadline"]["met"] is None and value["latest_deadline"]["unavailable"] is True


def cache_sink():
    """Expose the real HA cache writer and bulk path without network or database writers."""
    from ha import HAInterface

    sink = SimpleNamespace(state_data={}, db_enable=False, db_primary=False, ha_key="test", log=lambda *args: None, shared_attribute_lists={})
    sink.share_long_lists = HAInterface.share_long_lists.__get__(sink)
    sink.update_state_item = HAInterface.update_state_item.__get__(sink)
    sink.update_states = HAInterface.update_states.__get__(sink)
    return sink


def cache_event(sink, entity, state, stamp, old=None):
    """Deliver a genuine raw WS-shaped state event and keep exact observation timestamps."""
    payload = {"entity_id": entity, "state": state, "last_changed": stamp.isoformat(), "last_updated": stamp.isoformat(), "attributes": {"retrievalstatus": "VALID"}}
    sink.update_state_item(payload, entity, True, old_state=old)
    return payload


def test_real_cache_force_update_and_gaps(base):
    """Exercise actual cache code, not mocks that implement the intended continuity themselves."""
    from car_charging_policy import household_reason

    session_inputs(base)
    settings = car_settings(base)
    sink = cache_sink()
    entity = settings["home_entity"]
    old = cache_event(sink, entity, "home", base.now_utc - timedelta(hours=1))
    anchor = old["last_changed"]
    for seconds in (30, 20, 10):
        old = cache_event(sink, entity, "home", base.now_utc - timedelta(seconds=seconds), old)
        assert sink.state_data[entity]["value_last_changed"] == anchor
        assert sink.state_data[entity]["last_changed"] == old["last_changed"]
        assert sink.state_data[entity]["last_updated"] == old["last_updated"]
        base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
        assert household_reason(base) is None
    for key in ("state", "last_changed", "last_updated"):
        previous = deepcopy(old)
        previous[key] = "not_home" if key == "state" else (base.now_utc - timedelta(minutes=1)).isoformat()
        cache_event(sink, entity, "home", base.now_utc - timedelta(seconds=5), previous)
        base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
        assert household_reason(base) == "new_vehicle_session_required", key
    # A new runtime has no earlier event chain; it must not inherit an old anchor.
    cold = cache_sink()
    cache_event(cold, entity, "home", base.now_utc)
    base.ha_interface.dummy_items[entity] = deepcopy(cold.state_data[entity])
    assert household_reason(base) == "new_vehicle_session_required"


def test_real_bulk_preserves_only_identical_present_entities(base):
    """Use update_states itself: preserve a proven anchor, discard absent entities and gaps."""
    recurring_inputs(base)
    sink = cache_sink()
    old = cache_event(sink, "device_tracker.present", "home", base.now_utc - timedelta(hours=1))
    anchor = old["last_changed"]
    new = cache_event(sink, "device_tracker.present", "home", base.now_utc, old)
    cache_event(sink, "sensor.removed", "connected", base.now_utc)
    sink.api_call = lambda path: [new]
    sink.update_states()
    assert list(sink.state_data) == ["device_tracker.present"]
    assert sink.state_data["device_tracker.present"]["value_last_changed"] == anchor
    gap = {**new, "last_changed": (base.now_utc + timedelta(seconds=1)).isoformat(), "last_updated": (base.now_utc + timedelta(seconds=1)).isoformat()}
    sink.api_call = lambda path: [gap]
    sink.update_states()
    assert sink.state_data["device_tracker.present"]["value_last_changed"] is None


def test_recurring_observes_between_cycle_unplug_and_charging_states(base):
    """Reconnection between planning calls creates a new session; eligible changes do not."""
    recurring_inputs(base)
    settings = car_settings(base)
    settings["connection_entity_states"] = ["8", "1", "5"]
    sink = cache_sink()
    entity = settings["connection_entity"]
    old = cache_event(sink, entity, "8", base.now_utc - timedelta(minutes=10))
    base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
    first = resolve_recurring_session(base, now=base.now_utc)
    assert first["status"] == "ready", first
    for index, state in enumerate(("1", "5", "3", "8", "1"), 1):
        now = base.now_utc + timedelta(seconds=index)
        old = cache_event(sink, entity, state, now, old)
        if index in (2, 5):
            base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
            value = resolve_recurring_session(base, now=now)
            assert value["status"] == "ready", value
            assert (value["identity"] == first["identity"]) is (index == 2), value


def test_recurring_cold_connection_cache_recovers_proven_recorder_continuity(base):
    """A cold cache keeps the same EV session only when Recorder proves eligible continuity."""
    _, store = recurring_inputs(base)
    settings = car_settings(base)
    settings["connection_entity_states"] = ["8", "4"]
    settings["soc_observed_at_entity"] = None
    settings["connection_observed_at_entity"] = None
    base.car_charging_plan_time = [(base.now_utc + timedelta(hours=8)).strftime("%H:%M:%S")]
    entity = settings["connection_entity"]
    checked = base.now_utc - timedelta(minutes=10)
    _set_entity(base, entity, "8", last_changed=checked, last_updated=checked, attributes={"retrievalstatus": "VALID"})
    first = resolve_recurring_session(base, now=base.now_utc)
    assert first["status"] == "ready", first
    before = deepcopy(store.records[("car_charging", "recurring_session")]["session"])

    # A restarted HA cache contains only this later state; Recorder retains the
    # earlier eligible 8 -> 4 transition and the latest state publication.
    transition = base.now_utc + timedelta(minutes=45)
    now = base.now_utc + timedelta(hours=1)
    _set_entity(
        base,
        entity,
        "4",
        last_changed=transition,
        last_updated=transition,
        attributes={"retrievalstatus": "VALID", "value_history": [{"state": "4", "at": transition.isoformat()}], "value_last_changed": transition.isoformat()},
    )
    rows = [{"entity_id": entity, "state": state, "last_changed": stamp.isoformat(), "last_updated": stamp.isoformat()} for state, stamp in (("8", checked), ("4", checked + timedelta(milliseconds=141)), ("4", transition))]
    restored = RecurringStorage()
    restored.records = deepcopy(store.records)
    base.components = SimpleNamespace(get_component=lambda name: restored if name == "storage" else None)
    with patch.object(base.ha_interface, "get_history_window", return_value=[rows], create=True) as reader:
        current = resolve_recurring_session(base, now=now)

    assert current["status"] == "ready", current
    assert current["identity"] == first["identity"]
    retained = restored.records[("car_charging", "recurring_session")]["session"]
    for name in ("started_at", "initial_kwh", "meter_baseline", "anchor_kwh", "anchor_energy", "anchor_soc_at", "last_vehicle_value"):
        assert retained[name] == before[name], name
    assert retained["connection_checked_at"] == transition.isoformat()
    assert retained["identity"] == before["identity"]
    assert reader.call_count == 1
    assert reader.call_args.args == (entity, checked, now)


def test_recurring_cold_connection_cache_fails_closed_without_complete_history(base):
    """Missing, malformed, unknown or unavailable Recorder rows cannot mint a fresh EV baseline."""
    for fault in ("missing", "truncated", "stale", "unknown", "unavailable", "future", "reversed", "wrong_entity", "cache_conflict", "exception", "unplug"):
        _, store = recurring_inputs(base)
        settings = car_settings(base)
        settings["connection_entity_states"] = ["8", "4"]
        settings["soc_observed_at_entity"] = None
        settings["connection_observed_at_entity"] = None
        base.car_charging_plan_time = [(base.now_utc + timedelta(hours=8)).strftime("%H:%M:%S")]
        entity = settings["connection_entity"]
        checked = base.now_utc - timedelta(minutes=10)
        _set_entity(base, entity, "8", last_changed=checked, last_updated=checked, attributes={"retrievalstatus": "VALID"})
        first = resolve_recurring_session(base, now=base.now_utc)
        assert first["status"] == "ready", (fault, first)
        before = deepcopy(store.records[("car_charging", "recurring_session")]["session"])

        transition = base.now_utc + timedelta(minutes=45)
        now = base.now_utc + timedelta(hours=1)
        _set_entity(
            base,
            entity,
            "4",
            last_changed=transition,
            last_updated=transition,
            attributes={"retrievalstatus": "VALID", "value_history": [{"state": "4", "at": transition.isoformat()}], "value_last_changed": transition.isoformat()},
        )
        rows = [{"entity_id": entity, "state": state, "last_changed": stamp.isoformat(), "last_updated": stamp.isoformat()} for state, stamp in (("8", checked), ("4", checked + timedelta(milliseconds=141)), ("4", transition))]
        result_rows = [deepcopy(row) for row in rows]
        if fault == "missing":
            history = None
        elif fault == "truncated":
            history = [result_rows[1:]]
        elif fault == "stale":
            history = [result_rows[:-1]]
        elif fault in ("unknown", "unavailable"):
            result_rows[1]["state"] = fault
            history = [result_rows]
        elif fault == "future":
            result_rows[-1]["last_updated"] = (now + timedelta(seconds=1)).isoformat()
            history = [result_rows]
        elif fault == "reversed":
            history = [list(reversed(result_rows))]
        elif fault == "wrong_entity":
            result_rows[1]["entity_id"] = "sensor.other"
            history = [result_rows]
        elif fault == "cache_conflict":
            _set_entity(
                base,
                entity,
                "4",
                last_changed=transition,
                last_updated=transition,
                attributes={
                    "retrievalstatus": "VALID",
                    "value_history": [
                        {"state": "3", "at": (base.now_utc + timedelta(minutes=30)).isoformat()},
                        {"state": "4", "at": transition.isoformat()},
                    ],
                    "value_last_changed": transition.isoformat(),
                },
            )
            history = [result_rows]
        elif fault == "unplug":
            result_rows[1]["state"] = "3"
            _set_entity(base, settings["relay_entity"], "on", last_changed=now, last_updated=now)
            history = [result_rows]
        else:
            history = [result_rows]

        restored = RecurringStorage()
        restored.records = deepcopy(store.records)
        base.components = SimpleNamespace(get_component=lambda name: restored if name == "storage" else None)
        with patch.object(
            base.ha_interface,
            "get_history_window",
            return_value=history,
            side_effect=RuntimeError("Recorder unavailable") if fault == "exception" else None,
            create=True,
        ):
            result = resolve_recurring_session(base, now=now)

        expected = "recurring_new_baseline_requires_off" if fault == "unplug" else "connection_not_confirmed"
        assert result["status"] == "blocked" and result["reason"] == expected, (fault, result)
        assert restored.records[("car_charging", "recurring_session")]["session"] == before, fault


def test_recurring_baseline_is_not_reset_by_cache_heartbeat(base):
    """The standing resolver must consume the same transition evidence as dated sessions."""
    recurring_inputs(base)
    sink = cache_sink()
    entity = car_settings(base)["home_entity"]
    old = cache_event(sink, entity, "home", base.now_utc - timedelta(hours=1))
    base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
    first = resolve_recurring_session(base, now=base.now_utc)
    for minute in (1, 2, 3):
        now = base.now_utc + timedelta(minutes=minute)
        old = cache_event(sink, entity, "home", now, old)
        base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
        current = resolve_recurring_session(base, now=now)
        assert current["identity"] == first["identity"], current


def test_recurring_meter_stall_ignores_attribute_freshness(base):
    """Repeated timestamps/attribute reports cannot pretend cumulative energy advanced."""
    recurring_inputs(base)
    assert resolve_recurring_session(base, now=base.now_utc)["status"] == "ready"
    settings = car_settings(base)
    _set_entity(base, settings["relay_entity"], "on", last_changed=base.now_utc)
    later = base.now_utc + timedelta(minutes=11)
    _set_entity(base, "sensor.test_ev_energy", "100", last_updated=later, attributes={"unit_of_measurement": "kWh"})
    value = resolve_recurring_session(base, now=later)
    assert value["reason"] == "recurring_meter_stopped", value


def test_recurring_changed_soc_anchor_persists_after_measurement_ages(base):
    """A validated physical change corrects the estimate; aging cannot revert that correction."""
    recurring_inputs(base)
    assert resolve_recurring_session(base, now=base.now_utc)["status"] == "ready"
    later = base.now_utc + timedelta(minutes=20)
    _set_entity(base, "sensor.test_ev_energy", "101", last_updated=later, last_changed=later, attributes={"unit_of_measurement": "kWh"})
    value = base.car_charging_loss - 0.1
    _set_entity(base, car_settings(base)["soc_entity"], value, last_changed=later, last_updated=later, attributes={"retrievalstatus": "VALID"})
    corrected = resolve_recurring_session(base, now=later)
    assert corrected["status"] == "ready" and corrected["observation"]["ev_soc"] == value, corrected
    aged = resolve_recurring_session(base, now=later + timedelta(minutes=10))
    assert aged["status"] == "ready" and aged["observation"]["ev_soc"] == value, aged
    assert datetime.fromisoformat(aged["observation"]["soc_observed_at"]) == later


def test_recurring_actual_storage_persistence_and_corruption(base):
    """Exercise the installed filesystem backend, including its permissive legacy load path."""
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from ha import run_async
    from storage import StorageLocalFiles, StorageComponent

    recurring_inputs(base)
    with TemporaryDirectory(prefix="predbat-recurring-storage-") as directory:
        backend = StorageLocalFiles(directory, lambda *args: None)
        component = SimpleNamespace(backend=backend)
        for name in ("save", "load_checked"):
            setattr(component, name, getattr(StorageComponent, name).__get__(component))
        base.components = SimpleNamespace(get_component=lambda name: component)
        first = resolve_recurring_session(base, now=base.now_utc)
        assert first["status"] == "ready", first
        component.backend = StorageLocalFiles(directory, lambda *args: None)
        resumed = resolve_recurring_session(base, now=base.now_utc + timedelta(minutes=1))
        assert resumed["identity"] == first["identity"]
        cache = Path(directory) / "cache"
        data = cache / "car_charging_recurring_session.json"
        meta = cache / "car_charging_recurring_session.meta"
        data.write_text("{broken")
        assert run_async(component.backend.load("car_charging", "recurring_session")) is None
        assert resolve_recurring_session(base, now=base.now_utc + timedelta(minutes=2))["status"] == "blocked"
        assert data.read_text() == "{broken", "Corruption was silently replaced by a new authority record"
        meta.unlink()
        assert resolve_recurring_session(base, now=base.now_utc + timedelta(minutes=3))["status"] == "blocked"


def test_recurring_current_target_change_invalidates_staged_authority(base):
    """Changing the native ready time cannot extend a previously staged EV lease."""
    from car_charging_execute import execute_car_charging_plan
    from car_charging_plan import evaluate_car_charging

    (_, writes, _, _), _ = recurring_inputs(base)
    assert evaluate_car_charging(base, publish=False)["status"] == "candidate"
    base.car_charging_plan_time = [(base.now_utc + timedelta(minutes=45)).astimezone(timezone.utc).strftime("%H:%M:%S")]
    assert execute_car_charging_plan(base) is None and not writes
    assert base.car_charging_report["execution_reason"] == "configuration_changed", base.car_charging_report


def test_recurring_native_electrical_gates_remain_authoritative(base):
    """Standing intent does not override permission, reserve/restart, cooldown or read-only."""
    from car_charging_execute import execute_car_charging_plan
    from car_charging_plan import evaluate_car_charging

    for fault in ("read_only", "permission", "lease", "restart", "cooldown", "support"):
        (_, writes, _, _), _ = recurring_inputs(base)
        settings = car_settings(base)
        assert evaluate_car_charging(base, publish=False)["status"] == "candidate"
        if fault == "read_only":
            base.set_read_only = True
        elif fault == "permission":
            _set_entity(base, settings["permission_entity"], "off")
        elif fault == "lease":
            settings["ev_executor_lease_verified"] = False
        elif fault == "restart":
            base.soc_kw = base.soc_max * 0.20
        elif fault == "cooldown":
            _set_entity(base, settings["relay_entity"], "off", last_changed=base.now_utc - timedelta(seconds=599))
        else:
            _set_entity(base, "switch.test_support", "off")
        assert execute_car_charging_plan(base) is None, (fault, base.car_charging_report)
        assert not writes and base.car_charging_execution is None, (fault, writes)


def test_recurring_native_fetch_capability_reset_keeps_bounded_lease(base):
    """Reloading an unsupported native feature is not a new effective control configuration."""
    from car_charging_execute import execute_car_charging_plan, execution_fingerprint, publish_car_charging_slot
    from car_charging_plan import evaluate_car_charging

    (inverter, writes, _, _), _ = recurring_inputs(base)
    assert not inverter.inv_support_charge_freeze and not base.set_charge_freeze
    assert evaluate_car_charging(base, publish=False)["status"] == "candidate"
    execute_car_charging_plan(base)
    assert base.car_charging_execution is not None
    _set_entity(base, car_settings(base)["relay_entity"], "on")
    receipt = deepcopy(base.car_charging_execution)
    before = execution_fingerprint(base)
    count = len(writes)
    # fetch_config_options restores the user's True; fetch_inverter_data later
    # forces it back to False for this real DeyeSolarman capability. Neither is
    # a user edit. publish_car_plan runs BETWEEN these two native fetch stages.
    base.set_charge_freeze = True
    assert execution_fingerprint(base) == before
    assert publish_car_charging_slot(base) is True
    assert base.dashboard_values["binary_sensor." + base.prefix + "_car_charging_slot"]["state"] == "on"
    assert base.car_charging_execution == receipt and len(writes) == count
    base.set_charge_freeze = False
    assert execution_fingerprint(base) == before
    # A real hardware capability change must still revoke authority.
    inverter.inv_support_charge_freeze = True
    publish_car_charging_slot(base)
    assert base.car_charging_execution is None
    assert base.dashboard_values["binary_sensor." + base.prefix + "_car_charging_slot"]["state"] == "off"
    assert len(writes) == count


def test_recurring_effective_control_changes_still_revoke(base):
    """The effective-feature fix cannot hide actual policy, target, reserve or power changes."""
    from car_charging_execute import execute_car_charging_plan, publish_car_charging_slot
    from car_charging_plan import evaluate_car_charging

    for field, value in (("set_read_only", True), ("set_charge_low_power", True), ("set_discharge_during_charge", False), ("reserve", 4.0), ("car_charging_rate", [3.0]), ("car_charging_limit", [2.0])):
        (_, writes, _, _), _ = recurring_inputs(base)
        assert evaluate_car_charging(base, publish=False)["status"] == "candidate"
        execute_car_charging_plan(base)
        assert base.car_charging_execution is not None
        _set_entity(base, car_settings(base)["relay_entity"], "on")
        count = len(writes)
        base.set_charge_freeze = True
        setattr(base, field, value)
        publish_car_charging_slot(base)
        assert base.car_charging_execution is None, field
        assert base.dashboard_values["binary_sensor." + base.prefix + "_car_charging_slot"]["state"] == "off", field
        assert len(writes) == count


def stopped_recurring_inputs(base):
    """Reproduce the stopped session and the user's exact-event acknowledgement."""
    fixture, store = recurring_inputs(base)
    first = resolve_recurring_session(base, now=base.now_utc)
    settings = car_settings(base)
    stopped = base.now_utc + timedelta(minutes=1)
    _set_entity(base, settings["connection_entity"], "4", last_changed=stopped, last_updated=stopped, attributes={"retrievalstatus": "VALID"})
    assert resolve_recurring_session(base, now=stopped)["reason"] == "connection_not_confirmed"
    assert not store.records[("car_charging", "recurring_session")]["active"]
    base.now_utc += timedelta(minutes=2)
    settings["connection_error_acknowledgement"] = {
        "state": "4",
        "observed_at": stopped.isoformat(),
        "acknowledged_at": base.now_utc.isoformat(),
        "departure": first["session"]["departure"],
        "session_identity": first["identity"],
    }
    return fixture, store, first


def test_recurring_acknowledged_stop_preserves_baseline(base):
    """An explicitly reconciled interruption retains its original SOC and meter evidence."""
    (_, writes, _, _), store, first = stopped_recurring_inputs(base)
    settings = car_settings(base)
    # Cold-start tracker timestamps can advance without a value transition. The
    # operator has just confirmed this same stopped physical session, not new SOC.
    _set_entity(base, settings["home_entity"], "home", last_updated=base.now_utc, last_changed=base.now_utc)
    _set_entity(base, "sensor.test_ev_energy", "100.02", last_updated=base.now_utc, attributes={"unit_of_measurement": "kWh"})
    before = deepcopy(settings)
    value = resolve_recurring_session(base, now=base.now_utc)
    assert value["status"] == "ready" and value["identity"] == first["identity"], value
    assert abs(value["observation"]["ev_metered_ac_kwh"] - 0.02) < 1e-9
    assert value["observation"]["soc_observed_at"] == first["observation"]["soc_observed_at"]
    assert value["observation"]["connection_error_acknowledged"] is True
    assert store.records[("car_charging", "recurring_session")]["session"]["meter_baseline"] == 100
    assert store.records[("car_charging", "recurring_session")]["active"]
    assert settings == before and "4" not in settings["connection_entity_states"] and not writes
    # Genuine normal charging follows without losing the recovered energy baseline.
    later = base.now_utc + timedelta(minutes=1)
    sink = cache_sink()
    old = cache_event(sink, settings["connection_entity"], "4", datetime.fromisoformat(before["connection_error_acknowledgement"]["observed_at"]))
    cache_event(sink, settings["connection_entity"], "connected", later, old)
    base.ha_interface.dummy_items[settings["connection_entity"]] = deepcopy(sink.state_data[settings["connection_entity"]])
    resumed = resolve_recurring_session(base, now=later)
    assert resumed["status"] == "ready" and resumed["identity"] == first["identity"], resumed
    assert resumed["observation"]["connection_error_acknowledged"] is False


def test_recurring_stop_acknowledgement_scope(base):
    """Other events, sessions, deadlines, absences and contradictions never inherit consent."""
    for fault in ("missing", "identity", "departure", "future", "republished", "away", "new_home", "invalid", "unplug_history", "invalid_retrieval"):
        _, store, _ = stopped_recurring_inputs(base)
        settings = car_settings(base)
        ack = settings["connection_error_acknowledgement"]
        if fault == "missing":
            settings.pop("connection_error_acknowledgement")
        elif fault == "identity":
            ack["session_identity"] = "other-session"
        elif fault == "departure":
            base.car_charging_plan_time = ["23:59:00"]
        elif fault == "future":
            ack["acknowledged_at"] = (base.now_utc + timedelta(seconds=1)).isoformat()
        elif fault == "republished":
            _set_entity(base, settings["connection_entity"], "4", last_updated=base.now_utc, attributes={"retrievalstatus": "VALID"})
        elif fault == "away":
            _set_entity(base, settings["home_entity"], "not_home")
        elif fault == "new_home":
            base.now_utc += timedelta(seconds=1)
            _set_entity(base, settings["home_entity"], "home", last_changed=base.now_utc, last_updated=base.now_utc)
        elif fault == "invalid":
            store.records[("car_charging", "recurring_session")]["invalid"] = "recurring_meter_contradiction"
        elif fault == "unplug_history":
            sink = cache_sink()
            observed = datetime.fromisoformat(ack["observed_at"])
            old = cache_event(sink, settings["connection_entity"], "disconnected", observed - timedelta(seconds=1))
            cache_event(sink, settings["connection_entity"], "4", observed, old)
            base.ha_interface.dummy_items[settings["connection_entity"]] = deepcopy(sink.state_data[settings["connection_entity"]])
        else:
            _set_entity(base, settings["connection_entity"], "4", last_changed=datetime.fromisoformat(ack["observed_at"]), last_updated=datetime.fromisoformat(ack["observed_at"]), attributes={"retrievalstatus": "INVALID"})
        getter = base.get_state_wrapper
        forbidden_reads = []

        def guard(*args, **kwargs):
            """Reject SOC or meter reads until same-session availability is proved."""
            entity = kwargs.get("entity_id", args[0] if args else None)
            if entity in (settings["soc_entity"], "sensor.test_ev_energy", "sensor.test_ev_power"):
                forbidden_reads.append(entity)
                raise AssertionError(entity)
            return getter(*args, **kwargs)

        with patch.object(base, "get_state_wrapper", side_effect=guard):
            assert resolve_recurring_session(base, now=base.now_utc)["status"] == "blocked", fault
        assert not forbidden_reads, (fault, forbidden_reads)


def test_recurring_acknowledged_stop_native_lease(base):
    """Exact-event acknowledgement reaches the native executor, not a competing relay writer."""
    from car_charging_execute import execute_car_charging_plan, publish_car_charging_slot
    from car_charging_plan import evaluate_car_charging

    (_, writes, _, _), _, _ = stopped_recurring_inputs(base)
    # The two-minute interruption leaves only five complete charging slots in
    # this deliberately short fixture; retain a feasible synthetic target.
    base.car_charging_limit[0] = 2.0
    report = evaluate_car_charging(base, publish=False)
    assert report["status"] == "candidate", report
    assert execute_car_charging_plan(base)[0] == "Charging", base.car_charging_report
    assert publish_car_charging_slot(base) is True
    assert base.dashboard_values["binary_sensor." + base.prefix + "_car_charging_slot"]["state"] == "on"
    assert not any(service.startswith("switch") for service, _ in writes)
    count = len(writes)
    _set_entity(base, car_settings(base)["connection_entity"], "4", last_updated=base.now_utc, attributes={"retrievalstatus": "VALID"})
    publish_car_charging_slot(base)
    assert base.dashboard_values["binary_sensor." + base.prefix + "_car_charging_slot"]["state"] == "off"
    assert base.car_charging_execution is None and len(writes) == count


def test_recurring_stop_acknowledgement_keeps_electrical_guards(base):
    """Reconciliation does not waive electrical limits, manual policy or meter failures."""
    from car_charging_execute import execute_car_charging_plan
    from car_charging_plan import evaluate_car_charging

    for fault in ("permission", "restart", "cooldown", "read_only", "manual_soc", "meter", "support"):
        (_, writes, _, _), store, _ = stopped_recurring_inputs(base)
        settings = car_settings(base)
        base.car_charging_limit[0] = 2.0
        assert evaluate_car_charging(base, publish=False)["status"] == "candidate"
        if fault == "permission":
            _set_entity(base, settings["permission_entity"], "off")
        elif fault == "restart":
            base.soc_kw = base.soc_max * 0.20
        elif fault == "cooldown":
            _set_entity(base, settings["relay_entity"], "off", last_changed=base.now_utc - timedelta(seconds=599))
        elif fault == "read_only":
            base.set_read_only = True
        elif fault == "manual_soc":
            base.car_charging_manual_soc[0] = True
        elif fault == "meter":
            _set_entity(base, "sensor.test_ev_energy", "99", attributes={"unit_of_measurement": "kWh"})
        else:
            _set_entity(base, "switch.test_support", "off")
        assert execute_car_charging_plan(base) is None, (fault, base.car_charging_report)
        assert not writes and base.car_charging_execution is None, fault
        if fault == "meter":
            assert store.records[("car_charging", "recurring_session")]["invalid"] == "recurring_meter_contradiction"


def handoff_inputs(base, next_ev=True):
    """Stage a paired-safe EV run with a first-slot domestic target boundary."""
    from car_charging_execute import command_fingerprint, execute_car_charging_plan
    from car_charging_plan import evaluate_car_charging

    fixture, _ = recurring_inputs(base)
    # Make the native candidate stop battery charging after the first five-minute
    # cheap slot while the solver keeps the EV delivery ON across that boundary.
    base.rate_import.update({base.minutes_now + minute: 1 if minute < 5 else 10 for minute in range(base.forecast_minutes)})
    assert evaluate_car_charging(base, publish=False)["status"] == "candidate"
    pending = base.car_charging_pending
    # The target transition comes from this accepted paired-replay candidate,
    # rather than a post-validation mutation of its commands.
    first = pending["commands"][0]
    assert first["ev_on"] and pending["commands"][1]["ev_on"]
    assert first["target_soc"] != pending["commands"][1]["target_soc"]
    if not next_ev:
        # An OFF next step is a separate deliberately injected cancellation case.
        pending["commands"][1]["ev_on"] = False
        pending["command_fingerprint"] = command_fingerprint(pending["commands"])
    execute_car_charging_plan(base)
    assert base.car_charging_execution is not None, base.car_charging_report
    _set_entity(base, car_settings(base)["relay_entity"], "on")
    return fixture


def test_recurring_domestic_boundary_keeps_only_bounded_ev_handoff(base):
    """An already-ON EV survives domestic target-boundary latency, never an off slot or stale plan."""
    from car_charging_execute import publish_car_charging_slot

    _, writes, _, _ = handoff_inputs(base)
    receipt = deepcopy(base.car_charging_execution)
    boundary = base.midnight_utc + timedelta(minutes=receipt["command"]["end"])
    count = len(writes)
    # Renew just before the boundary, then publish during the ordinary new-plan
    # fetch. Neither action dispatches a new inverter command or extends forever.
    for stamp, renew in ((boundary - timedelta(seconds=15), True), (boundary + timedelta(seconds=2), False), (boundary + timedelta(seconds=29), True)):
        with patch("car_charging_plan._observation_clock", return_value=stamp):
            publish_car_charging_slot(base, renew=renew)
        slot = base.dashboard_values["binary_sensor." + base.prefix + "_car_charging_slot"]
        assert slot["state"] == "on", (stamp, base.car_charging_report)
        assert datetime.fromisoformat(slot["attributes"]["valid_until"]) <= boundary + timedelta(seconds=30)
        assert len(writes) == count
    with patch("car_charging_plan._observation_clock", return_value=boundary + timedelta(seconds=30)):
        publish_car_charging_slot(base, renew=True)
    assert base.car_charging_execution is None
    assert base.dashboard_values["binary_sensor." + base.prefix + "_car_charging_slot"]["state"] == "off"
    assert len(writes) == count
    handoff_inputs(base, next_ev=False)
    receipt = deepcopy(base.car_charging_execution)
    boundary = base.midnight_utc + timedelta(minutes=receipt["command"]["end"])
    with patch("car_charging_plan._observation_clock", return_value=boundary):
        publish_car_charging_slot(base, renew=True)
    assert base.car_charging_execution is None


def test_recurring_handoff_cannot_waive_independent_guards(base):
    """The brief overlap still rejects permissions, reserves, actual controls and target changes."""
    from car_charging_execute import publish_car_charging_slot

    for fault in ("permission", "reserve", "target", "controls", "deadline", "tampered_end"):
        _, writes, sources, _ = handoff_inputs(base)
        receipt = base.car_charging_execution
        boundary = base.midnight_utc + timedelta(minutes=receipt["command"]["end"])
        count = len(writes)
        if fault == "permission":
            _set_entity(base, car_settings(base)["permission_entity"], "off")
        elif fault == "reserve":
            base.soc_kw = base.soc_max * 0.15
        elif fault == "target":
            base.car_charging_limit[0] = 0
        elif fault == "controls":
            _set_entity(base, sources[0], "Disabled")
        elif fault == "deadline":
            base.car_charging_plan_time = [boundary.strftime("%H:%M:%S")]
        else:
            receipt["ev_handoff_until"] = (boundary + timedelta(seconds=31)).isoformat()
        with patch("car_charging_plan._observation_clock", return_value=boundary + timedelta(seconds=1)):
            publish_car_charging_slot(base, renew=True)
        assert base.car_charging_execution is None, fault
        assert len(writes) == count, fault


def test_recurring_acknowledgement_survives_proven_cold_home_heartbeat(base):
    """Recorder continuity reconciles a cold cache without inventing fresh SOC or consent."""
    (_, writes, _, _), store, first = stopped_recurring_inputs(base)
    settings = car_settings(base)
    acknowledged_at = base.now_utc
    assert resolve_recurring_session(base, now=base.now_utc)["status"] == "ready"
    for minute in (10, 20):
        # Each newly built HA cache only knows the last force-update publication.
        now = acknowledged_at + timedelta(minutes=minute)
        _set_entity(base, settings["home_entity"], "home", last_changed=now, last_updated=now)
        history = [[{"entity_id": settings["home_entity"], "state": "home", "last_changed": stamp.isoformat(), "last_updated": stamp.isoformat()} for stamp in (acknowledged_at, now)]]
        with patch.object(base.ha_interface, "get_history_window", return_value=history, create=True) as reader:
            value = resolve_recurring_session(base, now=now)
            assert value["status"] == "ready" and value["identity"] == first["identity"], value
            assert value["observation"]["soc_observed_at"] == first["observation"]["soc_observed_at"]
            assert store.records[("car_charging", "recurring_session")]["session"]["meter_baseline"] == 100
            assert reader.call_count == 1 and reader.call_args.args[0] == settings["home_entity"]
            assert resolve_recurring_session(base, now=now + timedelta(seconds=1))["status"] == "ready"
            assert reader.call_count == 1
    assert settings["connection_error_acknowledgement"]["acknowledged_at"] == acknowledged_at.isoformat()
    assert not writes


def test_recurring_cold_home_history_fails_closed_before_soc(base):
    """Missing, truncated, contradictory or wrong-entity history cannot preserve consent."""
    for fault in ("missing", "away", "unknown", "truncated", "stale", "future", "reversed", "other_entity", "exception"):
        _, _, _ = stopped_recurring_inputs(base)
        base.car_charging_home_history_check = None
        settings = car_settings(base)
        accepted = base.now_utc
        now = accepted + timedelta(minutes=10)
        _set_entity(base, settings["home_entity"], "home", last_changed=now, last_updated=now)
        rows = [{"entity_id": settings["home_entity"], "state": "home", "last_changed": stamp.isoformat(), "last_updated": stamp.isoformat()} for stamp in (accepted, accepted + timedelta(minutes=5), now)]
        if fault == "missing":
            rows = []
        elif fault in ("away", "unknown"):
            rows[1]["state"] = "not_home" if fault == "away" else "unavailable"
        elif fault == "truncated":
            rows = rows[1:]
        elif fault == "stale":
            rows = rows[:-1]
        elif fault == "future":
            rows[-1]["last_updated"] = (now + timedelta(seconds=1)).isoformat()
        elif fault == "reversed":
            rows = list(reversed(rows))
        elif fault == "other_entity":
            rows[1]["entity_id"] = "device_tracker.other"
        getter = base.get_state_wrapper
        forbidden_reads = []

        def guard(*args, **kwargs):
            """Track every forbidden read even when the public resolver catches exceptions."""
            entity = kwargs.get("entity_id", args[0] if args else None)
            if entity in (settings["soc_entity"], "sensor.test_ev_energy", "sensor.test_ev_power"):
                forbidden_reads.append(entity)
            return getter(*args, **kwargs)

        with patch.object(base.ha_interface, "get_history_window", return_value=[rows], side_effect=RuntimeError("offline") if fault == "exception" else None, create=True) as reader:
            with patch.object(base, "get_state_wrapper", side_effect=guard):
                assert resolve_recurring_session(base, now=now)["status"] == "blocked", fault
                assert resolve_recurring_session(base, now=now + timedelta(seconds=1))["status"] == "blocked", fault
            assert reader.call_count == 1, fault
        assert not forbidden_reads, (fault, forbidden_reads)


def test_recurring_meter_batches_use_energy_interval_not_poll_interval(base):
    """Five-minute meter batches remain valid under fifteen-second lease revalidation."""
    _, store = recurring_inputs(base)
    start = base.now_utc
    first = resolve_recurring_session(base, now=start)
    for second in range(15, 601, 15):
        now = start + timedelta(seconds=second)
        if second in (300, 600):
            _set_entity(base, "sensor.test_ev_energy", str(100 + second / 600), last_changed=now, last_updated=now, attributes={"unit_of_measurement": "kWh"})
        result = resolve_recurring_session(base, now=now)
        assert result["status"] == "ready" and result["identity"] == first["identity"], result
    assert abs(result["observation"]["ev_metered_ac_kwh"] - 1) < 1e-9
    assert store.records[("car_charging", "recurring_session")]["session"]["meter_baseline"] == 100
    # A genuinely impossible jump is still sticky, and cannot corrupt the last
    # valid meter anchor before storing the contradiction for later diagnosis.
    before = deepcopy(store.records[("car_charging", "recurring_session")]["session"])
    now = start + timedelta(seconds=615)
    _set_entity(base, "sensor.test_ev_energy", "105", last_changed=now, last_updated=now, attributes={"unit_of_measurement": "kWh"})
    assert resolve_recurring_session(base, now=now)["reason"] == "recurring_meter_contradiction"
    after = store.records[("car_charging", "recurring_session")]["session"]
    assert after["last_energy"] == before["last_energy"] and after["last_energy_progress_at"] == before["last_energy_progress_at"]


def test_recurring_delayed_meter_uses_genuine_publication_time(base):
    """Delayed cache consumption cannot compress two valid meter intervals into seconds."""
    _, store = recurring_inputs(base)
    start = base.now_utc
    assert resolve_recurring_session(base, now=start)["status"] == "ready"
    for minute, observed in ((5, 9), (10, 10)):
        source_time = start + timedelta(minutes=minute)
        _set_entity(base, "sensor.test_ev_energy", str(100 + minute / 10), last_changed=source_time, last_updated=source_time, attributes={"unit_of_measurement": "kWh"})
        result = resolve_recurring_session(base, now=start + timedelta(minutes=observed))
        assert result["status"] == "ready", result
        assert datetime.fromisoformat(store.records[("car_charging", "recurring_session")]["session"]["last_energy_progress_at"]) == source_time


def test_recurring_connection_only_ignores_gps_and_preserves_meter(base):
    """Omitting the optional tracker uses the real plug session, never a fabricated home state."""
    _, store = recurring_inputs(base)
    settings = car_settings(base)
    tracker = settings.pop("home_entity")
    _set_entity(base, tracker, "not_home")
    sink = cache_sink()
    entity = settings["connection_entity"]
    old = cache_event(sink, entity, "connected", base.now_utc - timedelta(hours=1))
    base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
    getter = base.get_state_wrapper
    tracker_reads = []

    def guard(*args, **kwargs):
        """Record forbidden GPS reads even when the resolver catches the exception."""
        entity = kwargs.get("entity_id", args[0] if args else None)
        if entity == tracker:
            tracker_reads.append(entity)
            raise AssertionError("Connection-only policy must not consult GPS")
        return getter(*args, **kwargs)

    with patch.object(base, "get_state_wrapper", side_effect=guard):
        first = resolve_recurring_session(base, now=base.now_utc)
        assert first["status"] == "ready", first
        later = base.now_utc + timedelta(minutes=5)
        _set_entity(base, tracker, "unavailable", last_changed=later, last_updated=later)
        _set_entity(base, "sensor.test_ev_energy", "100.5", last_changed=later, last_updated=later, attributes={"unit_of_measurement": "kWh"})
        value = resolve_recurring_session(base, now=later)
        assert value["status"] == "ready" and value["identity"] == first["identity"], value
        assert abs(value["observation"]["ev_metered_ac_kwh"] - 0.5) < 1e-9
    assert not tracker_reads
    assert store.records[("car_charging", "recurring_session")]["session"]["meter_baseline"] == 100
    old = cache_event(sink, entity, "disconnected", later, old)
    base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
    assert resolve_recurring_session(base, now=later)["reason"] == "connection_not_confirmed"
    later += timedelta(minutes=1)
    cache_event(sink, entity, "connected", later, old)
    base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
    reconnected = resolve_recurring_session(base, now=later)
    assert reconnected["status"] == "ready" and reconnected["identity"] != first["identity"], reconnected
    assert store.records[("car_charging", "recurring_session")]["session"]["meter_baseline"] == 100.5


def test_recurring_connection_only_keeps_availability_and_fault_guards(base):
    """Missing GPS never waives plug, consent, timestamps, metering or error observations."""
    for fault in ("unplugged", "unknown", "error", "invalid", "future", "permission", "meter"):
        _, store = recurring_inputs(base)
        settings = car_settings(base)
        settings.pop("home_entity")
        assert resolve_recurring_session(base, now=base.now_utc)["status"] == "ready"
        if fault in ("unplugged", "unknown", "error"):
            _set_entity(base, settings["connection_entity"], {"unplugged": "disconnected", "unknown": "unavailable", "error": "4"}[fault], attributes={"retrievalstatus": "VALID"})
        elif fault == "invalid":
            _set_entity(base, settings["connection_entity"], "connected", attributes={"retrievalstatus": "INVALID"})
        elif fault == "future":
            _set_entity(base, settings["connection_entity"], "connected", last_updated=base.now_utc + timedelta(seconds=1), attributes={"retrievalstatus": "VALID"})
        elif fault == "permission":
            _set_entity(base, settings["permission_entity"], "off")
        else:
            store.records[("car_charging", "recurring_session")]["invalid"] = "recurring_meter_contradiction"
        getter = base.get_state_wrapper
        forbidden_reads = []

        def guard(*args, **kwargs):
            """Observe availability-first validation before any vehicle energy read."""
            entity = kwargs.get("entity_id", args[0] if args else None)
            if entity in (settings["soc_entity"], "sensor.test_ev_energy", "sensor.test_ev_power"):
                forbidden_reads.append(entity)
            return getter(*args, **kwargs)

        with patch.object(base, "get_state_wrapper", side_effect=guard):
            result = resolve_recurring_session(base, now=base.now_utc)
        assert result["status"] == "blocked" and not forbidden_reads, (fault, result, forbidden_reads)


def test_recurring_connection_only_native_lease_and_revocation(base):
    """Connection-only authority still passes native replay and loses its lease on unplug."""
    from car_charging_execute import execute_car_charging_plan, publish_car_charging_slot
    from car_charging_plan import evaluate_car_charging

    for fault in ("connection", "permission", "error"):
        recurring_inputs(base)
        settings = car_settings(base)
        tracker = settings.pop("home_entity")
        _set_entity(base, tracker, "not_home")
        candidate = evaluate_car_charging(base, publish=False)
        assert candidate["status"] == "candidate", candidate
        execute_car_charging_plan(base)
        assert base.car_charging_execution is not None, base.car_charging_report
        _set_entity(base, settings["relay_entity"], "on")
        publish_car_charging_slot(base)
        assert base.dashboard_values["binary_sensor." + base.prefix + "_car_charging_slot"]["state"] == "on"
        entity = settings["permission_entity"] if fault == "permission" else settings["connection_entity"]
        _set_entity(base, entity, {"permission": "off", "connection": "disconnected", "error": "4"}[fault], attributes={"retrievalstatus": "VALID"})
        publish_car_charging_slot(base)
        assert base.car_charging_execution is None, fault
        assert base.dashboard_values["binary_sensor." + base.prefix + "_car_charging_slot"]["state"] == "off", fault


def test_recurring_removing_home_requires_explicit_policy_reconciliation(base):
    """Changing presence policy cannot silently discard a previous session or latched fault."""
    _, store = recurring_inputs(base)
    assert resolve_recurring_session(base, now=base.now_utc)["status"] == "ready"
    before = deepcopy(store.records)
    car_settings(base).pop("home_entity")
    assert resolve_recurring_session(base, now=base.now_utc)["reason"] == "recurring_policy_changed_requires_reconciliation"
    assert store.records == before


def test_recurring_connection_only_acknowledgement_keeps_exact_event_scope(base):
    """An optional absent tracker does not broaden an explicitly acknowledged stop event."""
    _, store = recurring_inputs(base)
    settings = car_settings(base)
    settings.pop("home_entity")
    first = resolve_recurring_session(base, now=base.now_utc)
    assert first["status"] == "ready", first
    stopped = base.now_utc + timedelta(minutes=1)
    _set_entity(base, settings["connection_entity"], "4", last_changed=stopped, last_updated=stopped, attributes={"retrievalstatus": "VALID"})
    assert resolve_recurring_session(base, now=stopped)["reason"] == "connection_not_confirmed"
    now = stopped + timedelta(minutes=1)
    settings["connection_error_acknowledgement"] = {"state": "4", "observed_at": stopped.isoformat(), "acknowledged_at": now.isoformat(), "departure": first["session"]["departure"], "session_identity": first["identity"]}
    value = resolve_recurring_session(base, now=now)
    assert value["status"] == "ready" and value["identity"] == first["identity"], value
    assert store.records[("car_charging", "recurring_session")]["session"]["meter_baseline"] == 100
    _set_entity(base, settings["connection_entity"], "4", last_changed=stopped, last_updated=now, attributes={"retrievalstatus": "VALID"})
    assert resolve_recurring_session(base, now=now)["reason"] == "connection_not_confirmed"


def test_recurring_telemetry_gap_keeps_meter_baseline_and_late_batch(base):
    """A brief unavailable report cannot rebase pending energy into an impossible new session."""
    for observe_gap in (False, True):
        _, store = recurring_inputs(base)
        settings = car_settings(base)
        settings.pop("home_entity")
        settings["connection_entity_states"] = ["connected", "charging"]
        base.car_charging_rate = [7]
        base.car_charging_plan_time = [(base.now_utc + timedelta(hours=2)).strftime("%H:%M:%S")]
        sink = cache_sink()
        entity = settings["connection_entity"]
        start = base.now_utc
        old = cache_event(sink, entity, "connected", start - timedelta(minutes=1))
        base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
        first = resolve_recurring_session(base, now=start)
        assert first["status"] == "ready", first
        _set_entity(base, settings["relay_entity"], "on", last_changed=start + timedelta(minutes=35), last_updated=start + timedelta(minutes=35))
        old = cache_event(sink, entity, "charging", start + timedelta(minutes=36), old)
        base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
        published = start + timedelta(minutes=38, seconds=23)
        _set_entity(base, "sensor.test_ev_energy", "100.25", last_changed=published, last_updated=published, attributes={"unit_of_measurement": "kWh"})
        assert resolve_recurring_session(base, now=published + timedelta(seconds=1))["status"] == "ready"
        before = deepcopy(store.records[("car_charging", "recurring_session")]["session"])
        gap = start + timedelta(minutes=40, seconds=18)
        old = cache_event(sink, entity, "unavailable", gap, old)
        base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
        if observe_gap:
            value = resolve_recurring_session(base, now=gap)
            assert value["status"] == "blocked" and value["reason"] == "connection_not_confirmed", value
            retained = store.records[("car_charging", "recurring_session")]["session"]
            assert retained == before
        old = cache_event(sink, entity, "charging", gap + timedelta(seconds=1), old)
        base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
        resumed = resolve_recurring_session(base, now=gap + timedelta(seconds=9))
        assert resumed["status"] == "ready" and resumed["identity"] == first["identity"], resumed
        published = start + timedelta(minutes=41, seconds=10)
        _set_entity(base, "sensor.test_ev_energy", "100.45", last_changed=published, last_updated=published, attributes={"unit_of_measurement": "kWh"})
        resumed = resolve_recurring_session(base, now=published + timedelta(seconds=1))
        assert resumed["status"] == "ready" and resumed["identity"] == first["identity"], resumed
        record = store.records[("car_charging", "recurring_session")]
        assert record["invalid"] == "" and record["session"]["meter_baseline"] == 100
        assert abs(resumed["observation"]["ev_metered_ac_kwh"] - 0.45) < 1e-9
        assert record["session"]["last_energy_progress_at"] == published.isoformat()


def test_recurring_gap_does_not_hide_a_real_unplug_or_retained_fault(base):
    """Unknown telemetry is not permission and cannot hide explicit disconnection or faults."""
    for fault in ("unplug", "retained_fault"):
        _, store = recurring_inputs(base)
        settings = car_settings(base)
        settings.pop("home_entity")
        sink = cache_sink()
        entity = settings["connection_entity"]
        old = cache_event(sink, entity, "connected", base.now_utc - timedelta(minutes=1))
        base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
        first = resolve_recurring_session(base, now=base.now_utc)
        assert first["status"] == "ready"
        _set_entity(base, settings["relay_entity"], "on", last_changed=base.now_utc)
        stamp = base.now_utc + timedelta(seconds=5)
        old = cache_event(sink, entity, "unavailable", stamp, old)
        if fault == "unplug":
            old = cache_event(sink, entity, "disconnected", stamp + timedelta(seconds=1), old)
        else:
            store.records[("car_charging", "recurring_session")]["invalid"] = "recurring_meter_contradiction"
        old = cache_event(sink, entity, "connected", stamp + timedelta(seconds=2), old)
        base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
        result = resolve_recurring_session(base, now=stamp + timedelta(seconds=3))
        assert result["reason"] == ("recurring_new_baseline_requires_off" if fault == "unplug" else "recurring_meter_contradiction"), result


def test_recurring_unavailable_connection_revokes_lease_without_losing_energy(base):
    """A telemetry gap withdraws live authority even when its energy evidence is retained."""
    from car_charging_execute import execute_car_charging_plan, publish_car_charging_slot
    from car_charging_plan import evaluate_car_charging

    recurring_inputs(base)
    settings = car_settings(base)
    settings.pop("home_entity")
    assert evaluate_car_charging(base, publish=False)["status"] == "candidate"
    execute_car_charging_plan(base)
    assert base.car_charging_execution is not None
    first = resolve_recurring_session(base, now=base.now_utc)
    _set_entity(base, settings["relay_entity"], "on")
    _set_entity(base, settings["connection_entity"], "unavailable", attributes={})
    publish_car_charging_slot(base)
    assert base.car_charging_execution is None
    assert base.dashboard_values["binary_sensor." + base.prefix + "_car_charging_slot"]["state"] == "off"
    assert first["status"] == "ready"


def test_recurring_allowlisted_stop_state_is_log_warning_only(base):
    """An explicitly eligible power-interruption status preserves the session without an error."""
    _, store = recurring_inputs(base)
    settings = car_settings(base)
    settings.pop("home_entity")
    settings["connection_warning_states"] = ["INTERRUPTED"]
    sink = cache_sink()
    entity = settings["connection_entity"]
    old = cache_event(sink, entity, "connected", base.now_utc - timedelta(minutes=1))
    base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
    first = resolve_recurring_session(base, now=base.now_utc)
    assert first["status"] == "ready", first
    before = deepcopy(store.records[("car_charging", "recurring_session")]["session"])
    stopped = base.now_utc + timedelta(seconds=1)
    cache_event(sink, entity, "INTERRUPTED", stopped, old)
    base.ha_interface.dummy_items[entity] = deepcopy(sink.state_data[entity])
    with patch.object(base, "log") as log, patch.object(base, "record_status") as status:
        for seconds in (1, 2):
            value = resolve_recurring_session(base, now=stopped + timedelta(seconds=seconds))
            assert value["status"] == "ready" and value["identity"] == first["identity"], value
        assert log.call_count == 1 and log.call_args.args[0].startswith("Warn: EV configured connection warning")
        status.assert_not_called()
    retained = store.records[("car_charging", "recurring_session")]
    assert retained["invalid"] == "" and retained["session"]["meter_baseline"] == before["meter_baseline"]


def arrival_baseline_inputs(base, prior_cycle=False):
    """Reproduce independently delayed arrival fields without delivering any AC energy."""
    fixture, store = recurring_inputs(base)
    settings = car_settings(base)
    settings.pop("home_entity")
    base.car_charging_battery_size = [20]
    base.car_charging_limit = [20]
    arrival = base.now_utc - timedelta(seconds=8)
    old_soc = arrival - timedelta(minutes=16)
    _set_entity(base, settings["connection_entity"], "connected", last_changed=arrival, last_updated=arrival, attributes={"retrievalstatus": "VALID"})
    _set_entity(base, settings["soc_entity"], 16, last_changed=old_soc, last_updated=old_soc, attributes={"retrievalstatus": "VALID"})
    clocks = {key: settings.pop(key) for key in ("soc_observed_at_entity", "connection_observed_at_entity")}
    if prior_cycle:
        _set_entity(base, settings["relay_entity"], "off", last_changed=base.now_utc - timedelta(seconds=1), last_updated=base.now_utc)
        _set_entity(base, "sensor.test_ev_energy", "101", last_changed=base.now_utc - timedelta(seconds=1), last_updated=base.now_utc, attributes={"unit_of_measurement": "kWh"})
    first = resolve_recurring_session(base, now=base.now_utc)
    assert first["status"] == "ready", first
    # Model a valid baseline retained by an earlier runtime, then supply real field clocks.
    settings.update(clocks)
    _set_entity(base, clocks["soc_observed_at_entity"], old_soc.isoformat())
    _set_entity(base, clocks["connection_observed_at_entity"], arrival.isoformat())
    return fixture, store, first


def test_recurring_delayed_lower_arrival_soc_before_delivery(base):
    """A post-arrival lower reading corrects a pre-arrival baseline before energising."""
    (_, writes, _, _), store, first = arrival_baseline_inputs(base)
    settings = car_settings(base)
    before = deepcopy(store.records[("car_charging", "recurring_session")]["session"])
    observed = base.now_utc + timedelta(seconds=107)
    _set_entity(base, settings["soc_entity"], 12.6, last_changed=observed, last_updated=observed, attributes={"retrievalstatus": "VALID"})
    _set_entity(base, settings["soc_observed_at_entity"], (base.now_utc - timedelta(seconds=7)).isoformat(), last_changed=observed, last_updated=observed)
    current = resolve_recurring_session(base, now=observed)
    assert current["status"] == "ready", current
    assert current["identity"] == first["identity"] and current["observation"]["ev_soc"] == 12.6
    saved = store.records[("car_charging", "recurring_session")]
    assert saved["invalid"] == "" and saved["session"]["initial_kwh"] == saved["session"]["anchor_kwh"] == 12.6
    for key in ("identity", "started_at", "meter_baseline", "last_energy", "anchor_energy", "home_transition"):
        assert saved["session"][key] == before[key], key
    assert not writes
    # A restart-like storage read and a later real energy batch retain the corrected anchor.
    restored = RecurringStorage()
    restored.records = deepcopy(store.records)
    base.components = SimpleNamespace(get_component=lambda name: restored)
    later = observed + timedelta(minutes=5)
    _set_entity(base, "sensor.test_ev_energy", "100.5", last_changed=later, last_updated=later, attributes={"unit_of_measurement": "kWh"})
    progress = resolve_recurring_session(base, now=later)
    assert progress["status"] == "ready", progress
    assert abs(progress["observation"]["ev_soc"] - (12.6 + 0.5 * base.car_charging_loss)) < 1e-9


def test_recurring_arrival_correction_cannot_hide_delivery_or_newer_baseline(base):
    """A lower reading remains a fault without complete evidence of an idle stale baseline."""
    for fault in ("relay_on", "relay_cycled", "power", "energy", "post_arrival_baseline", "stale_update", "higher_soc", "invalid_quality"):
        _, store, _ = arrival_baseline_inputs(base)
        settings = car_settings(base)
        started = base.now_utc
        observed = started + timedelta(seconds=107)
        value, quality = 12.6, "VALID"
        if fault == "relay_on":
            _set_entity(base, settings["relay_entity"], "on", last_changed=started + timedelta(seconds=1), last_updated=observed)
        elif fault == "relay_cycled":
            _set_entity(base, settings["relay_entity"], "off", last_changed=started + timedelta(seconds=30), last_updated=observed)
        elif fault == "power":
            _set_entity(base, "sensor.test_ev_power", "0.1", last_changed=observed, last_updated=observed, attributes={"unit_of_measurement": "kW"})
        elif fault == "energy":
            _set_entity(base, "sensor.test_ev_energy", "100.1", last_changed=observed, last_updated=observed, attributes={"unit_of_measurement": "kWh"})
        elif fault == "post_arrival_baseline":
            record = store.records[("car_charging", "recurring_session")]
            record["session"]["anchor_soc_at"] = started.isoformat()
        elif fault == "higher_soc":
            value = 19
        elif fault == "invalid_quality":
            quality = "INVALID"
        _set_entity(base, settings["soc_entity"], value, last_changed=observed, last_updated=observed, attributes={"retrievalstatus": quality})
        now = observed + timedelta(seconds=181) if fault == "stale_update" else observed
        current = resolve_recurring_session(base, now=now)
        assert current["status"] == "blocked", (fault, current)
        assert store.records[("car_charging", "recurring_session")]["session"]["initial_kwh"] == 16, fault


def test_recurring_arrival_correction_does_not_clear_retained_fault(base):
    """A persisted SOC fault still requires evidence-backed reconciliation, not a lower sample."""
    _, store, _ = arrival_baseline_inputs(base)
    settings = car_settings(base)
    store.records[("car_charging", "recurring_session")]["invalid"] = "recurring_soc_contradiction"
    before = deepcopy(store.records)
    observed = base.now_utc + timedelta(seconds=107)
    _set_entity(base, settings["soc_entity"], 12.6, last_changed=observed, last_updated=observed, attributes={"retrievalstatus": "VALID"})
    assert resolve_recurring_session(base, now=observed)["reason"] == "recurring_soc_contradiction"
    assert store.records == before


def test_recurring_arrival_correction_rejects_cycle_before_first_baseline(base):
    """An unchanged counter since a late baseline cannot hide delivery since connection."""
    _, store, _ = arrival_baseline_inputs(base, prior_cycle=True)
    settings = car_settings(base)
    observed = base.now_utc + timedelta(seconds=107)
    _set_entity(base, settings["soc_entity"], 12.6, last_changed=observed, last_updated=observed, attributes={"retrievalstatus": "VALID"})
    _set_entity(base, settings["soc_observed_at_entity"], observed.isoformat(), last_changed=observed, last_updated=observed)
    current = resolve_recurring_session(base, now=observed)
    assert current.get("reason") == "recurring_soc_contradiction", current
    assert store.records[("car_charging", "recurring_session")]["session"]["initial_kwh"] == 16


def test_recurring_arrival_correction_waits_for_physical_clock(base):
    """A fresh HA publication cannot turn an older physical reading into arrival evidence."""
    _, store, _ = arrival_baseline_inputs(base)
    settings = car_settings(base)
    observed = base.now_utc + timedelta(seconds=107)
    _set_entity(base, settings["soc_entity"], 12.6, last_changed=observed, last_updated=observed, attributes={"retrievalstatus": "VALID"})
    # The clock still describes the old sample, including the template update race.
    pending = resolve_recurring_session(base, now=observed)
    assert pending["status"] == "blocked" and pending["reason"] == "recurring_arrival_soc_pending", pending
    saved = store.records[("car_charging", "recurring_session")]
    assert saved["invalid"] == "" and saved["session"]["initial_kwh"] == 16
    _set_entity(base, settings["soc_observed_at_entity"], (base.now_utc - timedelta(seconds=7)).isoformat(), last_changed=observed, last_updated=observed)
    current = resolve_recurring_session(base, now=observed)
    assert current["status"] == "ready" and current["observation"]["ev_soc"] == 12.6, current


def test_recurring_new_arrival_waits_before_granting_energy_authority(base):
    """Let the coherent arrival fields settle before an old positive SOC can start charging."""
    (_, writes, _, _), store, _ = arrival_baseline_inputs(base)
    store.records.clear()
    current = resolve_recurring_session(base, now=base.now_utc)
    assert current["status"] == "blocked" and current["reason"] == "recurring_arrival_soc_pending", current
    assert not store.records[("car_charging", "recurring_session")]["active"] and not writes
    observed = base.now_utc + timedelta(seconds=107)
    settings = car_settings(base)
    _set_entity(base, settings["soc_entity"], 12.6, last_changed=observed, last_updated=observed, attributes={"retrievalstatus": "VALID"})
    _set_entity(base, settings["soc_observed_at_entity"], (base.now_utc - timedelta(seconds=7)).isoformat(), last_changed=observed, last_updated=observed)
    ready = resolve_recurring_session(base, now=observed)
    assert ready["status"] == "ready" and ready["observation"]["ev_soc"] == 12.6 and not writes, ready


def test_recurring_arrival_clock_failures_preserve_evidence(base):
    """Missing, naive, future or physically stale times cannot rewrite arrival evidence."""
    for fault in ("missing_mapping", "unavailable", "naive", "future", "stale"):
        _, store, _ = arrival_baseline_inputs(base)
        settings = car_settings(base)
        observed = base.now_utc + timedelta(seconds=107)
        _set_entity(base, settings["soc_entity"], 12.6, last_changed=observed, last_updated=observed, attributes={"retrievalstatus": "VALID"})
        entity = settings["soc_observed_at_entity"]
        if fault == "missing_mapping":
            settings.pop("soc_observed_at_entity")
        elif fault == "unavailable":
            _set_entity(base, entity, "unavailable")
        elif fault == "naive":
            _set_entity(base, entity, observed.replace(tzinfo=None).isoformat())
        elif fault == "future":
            _set_entity(base, entity, (observed + timedelta(seconds=1)).isoformat())
        else:
            _set_entity(base, settings["connection_observed_at_entity"], (base.now_utc - timedelta(hours=1)).isoformat())
            _set_entity(base, entity, (base.now_utc - timedelta(minutes=59)).isoformat())
        current = resolve_recurring_session(base, now=observed)
        assert current.get("reason") == "recurring_arrival_soc_pending", (fault, current)
        saved = store.records[("car_charging", "recurring_session")]
        assert saved["invalid"] == "" and saved["session"]["initial_kwh"] == 16 and saved["session"]["meter_baseline"] == 100, fault


def test_recurring_stable_preconnection_soc_is_estimated_not_a_deadlock(base):
    """A change-only vehicle can retain its idle pre-plug reading after the settling interval."""
    _, store, _ = arrival_baseline_inputs(base)
    store.records.clear()
    later = base.now_utc + timedelta(seconds=181)
    ready = resolve_recurring_session(base, now=later)
    assert ready["status"] == "ready" and ready["observation"]["ev_soc"] == 16, ready
    assert ready["observation"]["ev_soc_is_estimate"] and ready["observation"]["soc_observation_age_minutes"] > 16


def test_recurring_preconnection_baseline_rejects_physical_relay_cycle(base):
    """A connection publication arriving late cannot hide a post-plug relay cycle."""
    _, store, _ = arrival_baseline_inputs(base, prior_cycle=True)
    settings = car_settings(base)
    store.records.clear()
    connected = base.now_utc - timedelta(seconds=38)
    _set_entity(base, settings["connection_observed_at_entity"], connected.isoformat())
    _set_entity(base, settings["relay_entity"], "off", last_changed=base.now_utc - timedelta(seconds=20), last_updated=base.now_utc)
    later = base.now_utc + timedelta(seconds=181)
    current = resolve_recurring_session(base, now=later)
    assert current.get("reason") == "recurring_arrival_soc_pending", current
    assert not store.records[("car_charging", "recurring_session")]["active"]


def test_recurring_recent_lower_preplug_sample_is_only_an_estimate(base):
    """A lower idle arrival estimate never becomes fictitious measured completion."""
    _, store, _ = arrival_baseline_inputs(base)
    settings = car_settings(base)
    observed = base.now_utc + timedelta(seconds=107)
    _set_entity(base, settings["soc_entity"], 12.6, last_changed=observed, last_updated=observed, attributes={"retrievalstatus": "VALID"})
    _set_entity(base, settings["soc_observed_at_entity"], (base.now_utc - timedelta(seconds=9)).isoformat(), last_changed=observed, last_updated=observed)
    ready = resolve_recurring_session(base, now=observed)
    assert ready["status"] == "ready" and ready["observation"]["ev_soc"] == 12.6, ready
    assert ready["observation"]["ev_soc_is_estimate"] and not ready["observation"]["observed_target_met"]
    assert store.records[("car_charging", "recurring_session")]["session"]["meter_baseline"] == 100


def run_car_planning_recurring_tests(base):
    """Run isolated standing-policy tests through the normal Predbat registry."""
    from unit_test import create_predbat
    from tests.test_car_planning_vehicle_policy import test_vehicle_policy_contract
    from tests.test_car_planning_template import shared_template_available, check_unavailable_template

    if not shared_template_available():
        check_unavailable_template(base)
        return False

    for test in (
        test_recurring_native_ready_and_dst,
        test_recurring_stored_baseline_and_real_timestamps,
        test_recurring_absence_and_reconnection,
        test_recurring_counter_failure_survives_restart_and_rollover,
        test_recurring_storage_and_unauthorised_gates,
        test_recurring_rollover_preserves_energy_and_evidence,
        test_recurring_fetch_does_not_read_vehicle_soc,
        test_recurring_invalid_measurements_never_become_baselines,
        test_recurring_stale_full_is_not_observed_completion,
        test_recurring_cannot_escape_legacy_contradiction,
        test_recurring_absent_deadline_is_not_hidden,
        test_real_cache_force_update_and_gaps,
        test_real_bulk_preserves_only_identical_present_entities,
        test_recurring_observes_between_cycle_unplug_and_charging_states,
        test_recurring_cold_connection_cache_recovers_proven_recorder_continuity,
        test_recurring_cold_connection_cache_fails_closed_without_complete_history,
        test_recurring_baseline_is_not_reset_by_cache_heartbeat,
        test_recurring_meter_stall_ignores_attribute_freshness,
        test_recurring_changed_soc_anchor_persists_after_measurement_ages,
        test_recurring_actual_storage_persistence_and_corruption,
        test_recurring_native_planning_and_execution,
        test_recurring_current_target_change_invalidates_staged_authority,
        test_recurring_native_electrical_gates_remain_authoritative,
        test_recurring_native_fetch_capability_reset_keeps_bounded_lease,
        test_recurring_effective_control_changes_still_revoke,
        test_recurring_acknowledged_stop_preserves_baseline,
        test_recurring_stop_acknowledgement_scope,
        test_recurring_acknowledged_stop_native_lease,
        test_recurring_stop_acknowledgement_keeps_electrical_guards,
        test_recurring_domestic_boundary_keeps_only_bounded_ev_handoff,
        test_recurring_handoff_cannot_waive_independent_guards,
        test_recurring_acknowledgement_survives_proven_cold_home_heartbeat,
        test_recurring_cold_home_history_fails_closed_before_soc,
        test_recurring_meter_batches_use_energy_interval_not_poll_interval,
        test_recurring_delayed_meter_uses_genuine_publication_time,
        test_recurring_connection_only_ignores_gps_and_preserves_meter,
        test_recurring_connection_only_keeps_availability_and_fault_guards,
        test_recurring_connection_only_native_lease_and_revocation,
        test_recurring_removing_home_requires_explicit_policy_reconciliation,
        test_recurring_connection_only_acknowledgement_keeps_exact_event_scope,
        test_recurring_telemetry_gap_keeps_meter_baseline_and_late_batch,
        test_recurring_gap_does_not_hide_a_real_unplug_or_retained_fault,
        test_recurring_unavailable_connection_revokes_lease_without_losing_energy,
        test_recurring_allowlisted_stop_state_is_log_warning_only,
        test_recurring_delayed_lower_arrival_soc_before_delivery,
        test_recurring_arrival_correction_cannot_hide_delivery_or_newer_baseline,
        test_recurring_arrival_correction_does_not_clear_retained_fault,
        test_recurring_arrival_correction_rejects_cycle_before_first_baseline,
        test_recurring_arrival_correction_waits_for_physical_clock,
        test_recurring_new_arrival_waits_before_granting_energy_authority,
        test_recurring_arrival_clock_failures_preserve_evidence,
        test_recurring_stable_preconnection_soc_is_estimated_not_a_deadlock,
        test_recurring_preconnection_baseline_rejects_physical_relay_cycle,
        test_recurring_recent_lower_preplug_sample_is_only_an_estimate,
        test_vehicle_policy_contract,
    ):
        fresh = create_predbat()
        with patch("car_charging_plan._observation_clock", side_effect=lambda: fresh.now_utc):
            test(fresh)
    return False
