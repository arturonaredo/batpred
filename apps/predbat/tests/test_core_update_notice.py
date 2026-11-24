# -----------------------------------------------------------------------------
# Predbat Home Battery System
# Copyright Trefor Southwell 2026 - All Rights Reserved
# This application maybe used for personal use only and not for commercial use
# -----------------------------------------------------------------------------

"""Regression tests for truthful Home Assistant core-update metadata."""

import unittest
from unittest.mock import MagicMock

from userinterface import UserInterface


class CoreUpdateNoticeTests(unittest.TestCase):
    """Exercise actual update publication without Home Assistant or downloads."""

    def publish(self, available, installed="v9.8.1", latest="v9.8.0", automatic=False, in_progress=False):
        """Publish one discovery result using a minimal, transport-free owner."""
        owner = MagicMock()
        owner.config_index = {
            "version": {
                "type": "update",
                "entity": "update.predbat_version",
                "friendly_name": "Predbat Core Update",
                "title": "Predbat",
                "installed_version": installed,
                "entity_picture": "",
                "release_url": "https://example.invalid/release",
                "value": None,
            }
        }
        owner.user_config_item_enabled.return_value = True
        owner.releases = {"latest": latest, "latest_body": "Synthetic release notes"}
        owner.get_arg.return_value = automatic
        UserInterface.expose_config(owner, "version", available, force=True, in_progress=in_progress)
        owner.set_state_wrapper.assert_called_once()
        return owner, owner.set_state_wrapper.call_args.kwargs

    def test_ahead_of_stable_is_not_an_update(self):
        """A newer running build must respect discovery's no-update decision."""
        _, result = self.publish(False)
        self.assertEqual(result["state"], "off")
        self.assertEqual(result["attributes"]["installed_version"], "v9.8.1")
        self.assertEqual(result["attributes"]["latest_version"], "v9.8.0")

    def test_manual_updates_do_not_claim_automatic_updates(self):
        """The published automatic flag follows the real disabled setting."""
        owner, result = self.publish(True, installed="v9.7.0")
        self.assertEqual(result["state"], "on")
        self.assertIs(result["attributes"]["auto_update"], False)
        owner.get_arg.assert_called_once_with("auto_update", False)

    def test_enabled_automatic_setting_is_reported(self):
        """Do not hide a genuinely enabled automatic setting."""
        _, result = self.publish(True, installed="v9.7.0", automatic=True)
        self.assertIs(result["attributes"]["auto_update"], True)

    def test_matching_release_has_no_update(self):
        """An equal release remains up to date."""
        _, result = self.publish(False, latest="v9.8.1")
        self.assertEqual(result["state"], "off")

    def test_unknown_discovery_does_not_offer_install(self):
        """Failed discovery must preserve its explicit unavailable decision."""
        _, result = self.publish(False, latest="Unknown")
        self.assertEqual(result["state"], "off")

    def test_beta_or_custom_build_uses_discovery_decision(self):
        """No ad hoc version parser should reinterpret a valid discovery result."""
        for installed in ("v9.8.1-beta", "main", "custom"):
            with self.subTest(installed=installed):
                _, result = self.publish(False, installed=installed)
                self.assertEqual(result["state"], "off")

    def test_progress_is_retained_without_any_control_service(self):
        """Reporting progress must not trigger configuration or download actions."""
        owner, result = self.publish(True, in_progress=True)
        self.assertIs(result["attributes"]["in_progress"], True)
        self.assertEqual(set(name for name, _, _ in owner.mock_calls), {"user_config_item_enabled", "get_arg", "set_state_wrapper"})


if __name__ == "__main__":
    unittest.main()
