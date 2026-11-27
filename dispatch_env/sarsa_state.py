"""Compact four-feature state representation for tabular SARSA."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Tuple

from dispatch_env.types import AmbulanceStatus
from map_env.city_map import CITY_CENTER_COVERAGE_RADIUS_KM


StateKey = Tuple[int, int, int, int, int]
STATE_ENCODER_VERSION = "sarsa-state-meta-v2"

_DISTANCE_BUCKETS = (5.0, 12.0)
_DEFAULT_CENTER_COORD = (18.0, 16.0)


def _number(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Observation has a non-numeric {context}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Observation has a non-finite {context}")
    return result


def encode_sarsa_state(observation: Mapping[str, Any]) -> StateKey:
    """Encode fleet size, nearest distance, urgency, queue size, and center coverage."""
    if not isinstance(observation, Mapping):
        raise TypeError("observation must be a mapping")
    ambulances = observation.get("ambulances")
    call_slots = observation.get("call_slots")
    if not isinstance(ambulances, Mapping) or not isinstance(call_slots, Sequence):
        raise ValueError("Observation must contain ambulance and call-slot collections")

    idle_ambulances = []
    center_coord = observation.get("city_center_coord", _DEFAULT_CENTER_COORD)
    if (
        not isinstance(center_coord, Sequence)
        or isinstance(center_coord, (str, bytes))
        or len(center_coord) != 2
    ):
        raise ValueError("Observation has an invalid city center coordinate")
    center_x = _number(center_coord[0], "city center coord[0]")
    center_y = _number(center_coord[1], "city center coord[1]")
    idle_center_count = 0
    for ambulance in ambulances.values():
        if not isinstance(ambulance, Mapping):
            raise ValueError("Observation ambulance values must be mappings")
        status = ambulance.get("status")
        status_value = status.value if isinstance(status, AmbulanceStatus) else status
        if (
            status_value == AmbulanceStatus.IDLE.value
            and ambulance.get("current_call_id") is None
        ):
            coord = ambulance.get("coord")
            try:
                if isinstance(coord, (str, bytes)) or len(coord) != 2:
                    raise ValueError
                x = _number(coord[0], "ambulance coord[0]")
                y = _number(coord[1], "ambulance coord[1]")
            except (TypeError, IndexError, ValueError) as exc:
                raise ValueError("Observation has an invalid ambulance coordinate") from exc
            idle_ambulances.append((x, y))
            if math.hypot(x - center_x, y - center_y) <= CITY_CENTER_COVERAGE_RADIUS_KM:
                idle_center_count += 1

    pending_calls = [call for call in call_slots if call is not None]
    if any(not isinstance(call, Mapping) for call in pending_calls):
        raise ValueError("Observation call slots must contain mappings or None")

    if idle_ambulances and pending_calls:
        nearest_distance = math.inf
        for call in pending_calls:
            coord = call.get("coord")
            try:
                if isinstance(coord, (str, bytes)) or len(coord) != 2:
                    raise ValueError
                call_x = _number(coord[0], "call coord[0]")
                call_y = _number(coord[1], "call coord[1]")
                distance = min(
                    math.hypot(x - call_x, y - call_y)
                    for x, y in idle_ambulances
                )
                if distance < nearest_distance:
                    nearest_distance = distance
            except (TypeError, IndexError, ValueError) as exc:
                raise ValueError("Observation has an invalid call coordinate") from exc
        distance_bucket = next(
            (
                bucket
                for bucket, upper_bound in enumerate(_DISTANCE_BUCKETS)
                if nearest_distance <= upper_bound
            ),
            len(_DISTANCE_BUCKETS),
        )
        urgent_call = min(
            pending_calls,
            key=lambda call: (
                -int(call.get("severity", 1)),
                float(call.get("deadline", float("inf"))),
                float(call.get("created_time", 0.0)),
            ),
        )
        failure_time = _number(urgent_call.get("failure_time"), "call failure_time")
        wait_time = _number(urgent_call.get("wait_time"), "call wait_time")
        if failure_time <= 0.0:
            raise ValueError("Observation call failure_time must be positive")
        remaining = failure_time - wait_time
        urgency_bucket = int(remaining <= 0.5 * failure_time)
    else:
        distance_bucket = 2
        urgency_bucket = 1

    idle_count = min(len(idle_ambulances), 6)
    pending_bucket = min(len(pending_calls), 2)
    return (
        idle_count,
        distance_bucket,
        urgency_bucket,
        pending_bucket,
        idle_center_count,
    )


__all__ = ["STATE_ENCODER_VERSION", "StateKey", "encode_sarsa_state"]
