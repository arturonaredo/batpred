# -----------------------------------------------------------------------------
# Predbat Home Battery System
# Copyright Trefor Southwell 2026 - All Rights Reserved
# This application maybe used for personal use only and not for commercial use
# -----------------------------------------------------------------------------
# fmt off
# pylint: disable=consider-using-f-string
# pylint: disable=line-too-long

"""Pure, opt-in evening load forecast adjustment candidate."""

from collections.abc import Mapping
from math import floor, isfinite
from numbers import Real

from const import PREDICT_STEP
from utils import dp4

EVENING_START_MINUTE = 19 * 60
LATEST_CHEAP_START = 27 * 60
REFERENCE_EVENING_MINUTES = 5 * 60
EVENING_RATE_LIMIT_MINOR = 10.0
EVENING_ADJUSTMENT_MAX_KWH = 2.0


def apply_evening_load_adjustment(
    load_steps,
    minutes_now,
    rate_import,
    rate_import_base,
    rate_import_replicated,
    rate_import_base_replicated,
    enabled,
    adjustment_kwh,
):
    """Return a copied load-step forecast with a bounded fixed evening addition.

    A five-hour interval receives up to the configured amount; shorter intervals
    receive proportionately less, and longer intervals respect the same cap. Only
    the remaining portion from 19:00 until the first verified cheap rate (possibly
    after midnight) is adjusted. Any tariff uncertainty before that boundary, or no
    boundary by 03:00, leaves the baseline unchanged. No input is mutated.
    """
    if enabled is not True:
        return load_steps
    if isinstance(adjustment_kwh, bool) or not isinstance(adjustment_kwh, Real):
        return load_steps
    if not isfinite(adjustment_kwh) or adjustment_kwh <= 0 or adjustment_kwh > EVENING_ADJUSTMENT_MAX_KWH:
        return load_steps
    if isinstance(minutes_now, bool) or not isinstance(minutes_now, int):
        return load_steps
    if not isinstance(load_steps, Mapping):
        return load_steps
    if not isinstance(rate_import, Mapping) or not isinstance(rate_import_base, Mapping):
        return load_steps
    if not isinstance(rate_import_replicated, Mapping) or not isinstance(rate_import_base_replicated, Mapping):
        return load_steps

    start = max(minutes_now, EVENING_START_MINUTE)
    deadline = None
    for minute in range(EVENING_START_MINUTE, LATEST_CHEAP_START + 1):
        if minute in rate_import_replicated or minute in rate_import_base_replicated:
            return load_steps
        live_rate = rate_import.get(minute)
        base_rate = rate_import_base.get(minute)
        if any(isinstance(rate, bool) or not isinstance(rate, Real) or not isfinite(rate) for rate in (live_rate, base_rate)):
            return load_steps
        if live_rate != base_rate:
            return load_steps
        if base_rate < EVENING_RATE_LIMIT_MINOR:
            deadline = minute
            break
    if deadline is None or deadline <= start:
        return load_steps

    denominator = max(REFERENCE_EVENING_MINUTES, deadline - EVENING_START_MINUTE)
    adjusted = dict(load_steps)
    for step_start, load_kwh in load_steps.items():
        if isinstance(step_start, bool) or not isinstance(step_start, int) or step_start < 0:
            continue
        if isinstance(load_kwh, bool) or not isinstance(load_kwh, Real) or not isfinite(load_kwh):
            continue

        minute = minutes_now + step_start
        overlap = min(minute + PREDICT_STEP, deadline) - max(minute, start)
        if overlap > 0:
            # Floor to the forecast's 0.1 Wh resolution, ensuring rounding cannot
            # exceed the configured per-evening energy bound.
            adjustment_units = floor(adjustment_kwh * overlap * 10000 / denominator)
            if adjustment_units > 0:
                adjusted[step_start] = dp4(load_kwh + adjustment_units / 10000.0)

    return adjusted
