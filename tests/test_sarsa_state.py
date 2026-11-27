import copy
import pickle
import unittest

import numpy as np

from dispatch_env import AmbulanceDispatchEnv
from dispatch_env.sarsa_state import encode_sarsa_state
from dispatch_env.types import AmbulanceStatus


def _observation():
    return {
        "current_minute": 10,
        "ambulances": {
            ambulance_id: {
                "status": AmbulanceStatus.IDLE.value,
                "current_call_id": None,
                "coord": (10.0 + ambulance_id, 10.0),
            }
            for ambulance_id in range(1, 7)
        },
        "call_slots": [
            {
                "call_id": 41,
                "severity": 3,
                "coord": (14.0, 10.0),
                "wait_time": 8.0,
                "failure_time": 30.0,
            },
            None,
            None,
        ],
        "action_mask": np.ones(229, dtype=bool),
    }


class TestSarsaStateEncoder(unittest.TestCase):
    def test_state_is_exactly_five_discrete_features(self):
        state = encode_sarsa_state(_observation())

        self.assertEqual(len(state), 5)
        self.assertTrue(all(isinstance(feature, int) for feature in state))
        idle_count, distance_bucket, urgency_bucket, pending_bucket, center_idle_count = state
        self.assertIn(idle_count, range(7))
        self.assertIn(distance_bucket, range(3))
        self.assertIn(urgency_bucket, range(2))
        self.assertIn(pending_bucket, range(3))
        self.assertIn(center_idle_count, range(7))

    def test_possible_state_count_is_bounded_by_882(self):
        states = {
            (idle, distance, urgency, pending, center_idle)
            for idle in range(7)
            for distance in range(3)
            for urgency in range(2)
            for pending in range(3)
            for center_idle in range(7)
        }

        self.assertEqual(len(states), 882)

    def test_raw_minute_and_call_id_do_not_change_state(self):
        first = _observation()
        second = copy.deepcopy(first)
        second["current_minute"] = 211
        second["call_slots"][0]["call_id"] = 9999

        self.assertEqual(encode_sarsa_state(first), encode_sarsa_state(second))

    def test_idle_coverage_counts_ambulances_inside_center_radius(self):
        state = encode_sarsa_state(_observation())

        self.assertEqual(state[4], 4)

    def test_small_coordinate_changes_within_same_bucket_do_not_change_state(self):
        first = _observation()
        second = copy.deepcopy(first)
        second["call_slots"][0]["coord"] = (14.1, 10.1)

        self.assertEqual(encode_sarsa_state(first), encode_sarsa_state(second))

    def test_queue_and_urgency_are_discretized(self):
        observation = _observation()
        normal = encode_sarsa_state(observation)
        observation["call_slots"][0]["wait_time"] = 20.0
        urgent = encode_sarsa_state(observation)
        self.assertEqual(normal[2], 0)
        self.assertEqual(urgent[2], 1)

        observation["call_slots"][1] = {
            "call_id": 42,
            "severity": 1,
            "coord": (20.0, 10.0),
            "wait_time": 0.0,
            "failure_time": 60.0,
        }
        self.assertEqual(encode_sarsa_state(observation)[3], 2)

    def test_no_idle_vehicle_uses_far_distance_bucket(self):
        observation = _observation()
        for ambulance in observation["ambulances"].values():
            ambulance["status"] = AmbulanceStatus.EN_ROUTE_TO_CALL.value

        state = encode_sarsa_state(observation)

        self.assertEqual(state[0], 0)
        self.assertEqual(state[1], 2)

    def test_encoding_is_deterministic_hashable_and_does_not_mutate_input(self):
        observation = _observation()
        before = pickle.dumps(observation, protocol=pickle.HIGHEST_PROTOCOL)

        state = encode_sarsa_state(observation)

        self.assertEqual(state, encode_sarsa_state(observation))
        self.assertIsInstance(hash(state), int)
        self.assertEqual(
            pickle.dumps(observation, protocol=pickle.HIGHEST_PROTOCOL),
            before,
        )

    def test_accepts_real_environment_observation(self):
        env = AmbulanceDispatchEnv(seed=42)
        observation = env.reset(seed=42)

        state = encode_sarsa_state(observation)

        self.assertEqual(len(state), 5)
        self.assertTrue(all(isinstance(feature, int) for feature in state))


if __name__ == "__main__":
    unittest.main()
