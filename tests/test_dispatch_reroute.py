import unittest

from dispatch_env import AmbulanceDispatchEnv
from dispatch_env.ambulance import Ambulance
from dispatch_env.call import EmergencyCall
from dispatch_env.types import AmbulanceStatus, CallStatus


class FakeCityMap:
    def __init__(self, etas):
        self.etas = etas

    def route_from_road_to_road(
        self,
        start_edge,
        start_dist,
        dest_edge,
        dest_dist,
        by_travel_time=True,
    ):
        return {
            "edges_path": [start_edge, dest_edge],
            "total_time_minutes": self.etas[(start_edge, dest_edge)],
        }


def _call(call_id, edge, severity, created_time, failure_time):
    return EmergencyCall(
        call_id=call_id,
        edge=edge,
        distance_from_start=1.0,
        coord=(10.0, 10.0),
        severity=severity,
        created_time=created_time,
        failure_time=failure_time,
        status=CallStatus.PENDING,
    )


def _environment(current_eta, new_eta, replacement_eta):
    env = AmbulanceDispatchEnv(seed=42)
    env.current_minute = 5
    current_call = _call(1, (10, 11), severity=1, created_time=0, failure_time=60)
    current_call.status = CallStatus.WAITING_FOR_AMBULANCE
    current_call.assigned_ambulance_id = 1
    new_call = _call(2, (20, 21), severity=1, created_time=5, failure_time=40)

    busy = Ambulance(
        ambulance_id=1,
        current_edge=(1, 2),
        distance_from_start=1.0,
        status=AmbulanceStatus.EN_ROUTE_TO_CALL,
        current_call_id=1,
        target_type="call",
        target_id=1,
        target_edge=current_call.edge,
        target_dist=current_call.distance_from_start,
        route_edges=[(1, 2), current_call.edge],
    )
    replacement = Ambulance(
        ambulance_id=2,
        current_edge=(3, 4),
        distance_from_start=1.0,
        status=AmbulanceStatus.IDLE,
    )
    env.ambulances = {1: busy, 2: replacement}
    env.calls = {1: current_call, 2: new_call}
    env.city_map = FakeCityMap(
        {
            ((1, 2), current_call.edge): current_eta,
            ((1, 2), new_call.edge): new_eta,
            ((3, 4), current_call.edge): replacement_eta,
        }
    )
    return env, current_call, new_call


class TestPendingCallRerouting(unittest.TestCase):
    def test_reroutes_for_a_call_at_least_25_percent_closer(self):
        env, current_call, new_call = _environment(
            current_eta=80.0,
            new_eta=30.0,
            replacement_eta=20.0,
        )

        forecasts = env._reroute_to_pending_calls()

        self.assertEqual(len(forecasts), 1)
        self.assertEqual(current_call.assigned_ambulance_id, 2)
        self.assertEqual(new_call.assigned_ambulance_id, 1)
        self.assertEqual(env.ambulances[1].reroute_count, 1)

    def test_higher_priority_reroute_requires_a_timely_replacement(self):
        env, current_call, new_call = _environment(
            current_eta=30.0,
            new_eta=29.0,
            replacement_eta=20.0,
        )
        new_call.severity = 3
        new_call.failure_time = 30.0
        new_call.created_time = 5.0

        forecasts = env._reroute_to_pending_calls()

        self.assertEqual(len(forecasts), 1)
        self.assertEqual(current_call.assigned_ambulance_id, 2)
        self.assertEqual(new_call.assigned_ambulance_id, 1)

    def test_rejects_reroute_if_old_call_cannot_be_covered_on_time(self):
        env, current_call, new_call = _environment(
            current_eta=30.0,
            new_eta=29.0,
            replacement_eta=60.0,
        )
        new_call.severity = 3
        new_call.failure_time = 30.0
        new_call.created_time = 5.0

        forecasts = env._reroute_to_pending_calls()

        self.assertEqual(forecasts, [])
        self.assertEqual(current_call.assigned_ambulance_id, 1)
        self.assertIsNone(new_call.assigned_ambulance_id)

    def test_does_not_reroute_for_a_small_eta_improvement(self):
        env, current_call, new_call = _environment(
            current_eta=80.0,
            new_eta=70.0,
            replacement_eta=20.0,
        )

        self.assertEqual(env._reroute_to_pending_calls(), [])
        self.assertEqual(current_call.assigned_ambulance_id, 1)
        self.assertIsNone(new_call.assigned_ambulance_id)


if __name__ == "__main__":
    unittest.main()
