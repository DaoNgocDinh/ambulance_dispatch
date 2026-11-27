import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dispatch_env import AmbulanceDispatchEnv, SarsaAgent
from dispatch_env.call import EmergencyCall
from dispatch_env.sarsa import (
    nearest_available_action_id,
    resolve_meta_action,
    sarsa_action_mask,
)
from dispatch_env.sarsa_state import (
    STATE_ENCODER_VERSION,
    encode_sarsa_state,
)
from dispatch_env.types import AmbulanceStatus, CallStatus


def make_observation(
    minute=10,
    episode=1,
    action_mask=(True, True, False),
    last_step_reward=0.0,
):
    return {
        "episode": episode,
        "current_minute": minute,
        "edge_congestion_vector": [1.0, 1.0],
        "edges_by_id": {},
        "ambulances": {
            1: {
                "status": AmbulanceStatus.IDLE.value,
                "coord": (10.0, 10.0),
            }
        },
        "active_calls": [],
        "call_slots": [None, None, None],
        "action_mask": np.asarray(action_mask, dtype=bool),
        "last_step_reward": last_step_reward,
    }


class FakeEnv:
    """A small API-only environment double; it exposes no dispatch internals."""

    def __init__(self, transitions, reset_reward=0.0):
        self.transitions = transitions
        self.reset_reward = reset_reward
        self.calls = []
        self.step_index = 0

    def reset(self, seed=None):
        self.calls.append(("reset", seed))
        self.step_index = 0
        observation = make_observation(last_step_reward=self.reset_reward)
        return observation

    def step(self, action):
        self.calls.append(("step", action))
        next_observation, reward, done, info = self.transitions[self.step_index]
        self.step_index += 1
        return copy.deepcopy(next_observation), reward, done, copy.deepcopy(info)


class TestSarsaAgent(unittest.TestCase):
    def assert_nested_state_key_uses_tuples(self, value):
        if isinstance(value, tuple):
            for item in value:
                self.assert_nested_state_key_uses_tuples(item)
        else:
            self.assertNotIsInstance(value, list)

    def test_action_always_respects_action_mask(self):
        agent = SarsaAgent(epsilon=1.0, epsilon_min=0.0, seed=7)
        state = encode_sarsa_state(make_observation())
        mask = np.asarray([False, True, False, False, False, False], dtype=bool)

        for _ in range(50):
            self.assertEqual(agent.choose_action(state, mask), 1)

    def test_empty_action_mask_raises_instead_of_falling_back(self):
        agent = SarsaAgent(seed=7)
        with self.assertRaisesRegex(ValueError, "no valid action"):
            agent.choose_action(
                encode_sarsa_state(make_observation()),
                np.asarray([False, False, False, False], dtype=bool),
            )

    def test_sarsa_mask_blocks_noop_when_idle_ambulance_and_call_exist(self):
        observation = make_observation(action_mask=(True, True, True))
        observation["call_slots"][0] = {
            "call_id": 7,
            "severity": 2,
            "coord": (11.0, 10.0),
            "wait_time": 0.0,
            "failure_time": 40.0,
        }

        mask = sarsa_action_mask(observation)

        self.assertEqual(mask.tolist(), [True, True, True, False, False, False])

    def test_sarsa_mask_allows_only_noop_when_no_dispatch_is_possible(self):
        observation = make_observation()

        self.assertEqual(
            sarsa_action_mask(observation).tolist(),
            [False, False, False, False, False, True],
        )

    def test_sarsa_mask_allows_noop_when_all_ambulances_are_busy(self):
        observation = make_observation()
        observation["call_slots"][0] = {
            "call_id": 7,
            "severity": 2,
            "coord": (11.0, 10.0),
            "wait_time": 0.0,
            "failure_time": 40.0,
        }
        observation["ambulances"][1]["status"] = "Đang đến cuộc gọi"

        self.assertEqual(
            sarsa_action_mask(observation).tolist(),
            [False, False, False, False, False, True],
        )

    def test_meta_actions_resolve_to_nearest_second_nearest_and_least_driven(self):
        observation = make_observation(action_mask=(True, True, True))
        observation["call_slots"][0] = {
            "call_id": 7,
            "severity": 2,
            "coord": (11.0, 10.0),
            "deadline": 40.0,
            "wait_time": 0.0,
            "failure_time": 40.0,
        }
        observation["ambulances"] = {
            1: {"status": "Rảnh", "coord": (10.0, 10.0), "total_distance_traveled": 10.0},
            2: {"status": "Rảnh", "coord": (12.0, 10.0), "total_distance_traveled": 2.0},
            3: {"status": "Rảnh", "coord": (15.0, 10.0), "total_distance_traveled": 5.0},
        }

        class DispatchEnv:
            @staticmethod
            def encode_action(pairs):
                return pairs[0][0]

        env = DispatchEnv()
        self.assertEqual(resolve_meta_action(env, observation, 0), (1, [(1, 7)]))
        self.assertEqual(resolve_meta_action(env, observation, 1), (2, [(2, 7)]))
        self.assertEqual(resolve_meta_action(env, observation, 2), (2, [(2, 7)]))

    def test_edge_first_is_masked_in_only_when_two_units_meet_perfect_threshold(self):
        env = AmbulanceDispatchEnv(seed=42, poisson_lambda=0.0)
        env.reset(generate_initial_calls=False)
        call_edge = (1, 18)
        env.calls[7] = EmergencyCall(
            7,
            call_edge,
            0.25,
            (18.0, 16.25),
            severity=3,
            created_time=0.0,
            status=CallStatus.PENDING,
        )
        for ambulance_id, distance in ((1, 0.1), (2, 0.8)):
            ambulance = env.ambulances[ambulance_id]
            ambulance.current_edge = call_edge
            ambulance.distance_from_start = distance
            ambulance.route_edges = [call_edge]
        for ambulance_id in range(3, 7):
            ambulance = env.ambulances[ambulance_id]
            ambulance.status = AmbulanceStatus.EN_ROUTE_TO_CALL
            ambulance.current_call_id = 100 + ambulance_id

        observation = env.get_observation()
        mask = sarsa_action_mask(observation, env)

        self.assertTrue(mask[3])
        self.assertFalse(mask[4])
        self.assertFalse(mask[5])
        self.assertEqual(
            resolve_meta_action(env, observation, 3)[1],
            [(2, 7)],
        )

    def test_load_balancing_penalty_applies_when_nearest_preserves_perfect_response(self):
        env = AmbulanceDispatchEnv(seed=42, poisson_lambda=0.0)
        env.reset(generate_initial_calls=False)
        call_edge = (1, 18)
        env.calls[7] = EmergencyCall(
            7,
            call_edge,
            0.25,
            (18.0, 16.25),
            severity=3,
            created_time=0.0,
            status=CallStatus.PENDING,
        )
        near = env.ambulances[1]
        near.current_edge = call_edge
        near.distance_from_start = 0.1
        near.route_edges = [call_edge]
        near.total_distance_traveled = 10.0
        far = env.ambulances[2]
        far.current_edge = (67, 74)
        far.distance_from_start = 0.5
        far.route_edges = [(67, 74)]
        far.total_distance_traveled = 0.0
        for ambulance_id in range(3, 7):
            ambulance = env.ambulances[ambulance_id]
            ambulance.status = AmbulanceStatus.EN_ROUTE_TO_CALL
            ambulance.current_call_id = 100 + ambulance_id

        observation = env.get_observation()
        agent = SarsaAgent()
        pairs = resolve_meta_action(env, observation, 2)[1]
        forecasts = agent._dispatch_forecasts(env, pairs)

        self.assertEqual(pairs, [(2, 7)])
        self.assertEqual(
            agent._load_balance_perfect_miss_penalty(
                env,
                observation,
                2,
                forecasts,
            ),
            -25.0,
        )

    def test_idle_reposition_meta_action_requires_a_cluster_and_no_pending_calls(self):
        env = AmbulanceDispatchEnv(seed=42, poisson_lambda=0.0)
        env.reset(generate_initial_calls=False)
        for ambulance_id in (1, 2):
            ambulance = env.ambulances[ambulance_id]
            ambulance.current_edge = (67, 74)
            ambulance.distance_from_start = 0.5
            ambulance.route_edges = [(67, 74)]

        observation = env.get_observation()
        mask = sarsa_action_mask(observation, env)

        self.assertEqual(mask.tolist(), [False, False, False, False, True, True])
        self.assertEqual(resolve_meta_action(env, observation, 4), (0, []))
        self.assertTrue(
            any(ambulance.is_repositioning for ambulance in env.ambulances.values())
        )

        observation = env.get_observation()
        self.assertEqual(
            sarsa_action_mask(observation, env).tolist(),
            [False, False, False, False, False, True],
        )

    def test_unseen_or_tied_greedy_state_uses_nearest_fallback(self):
        state = encode_sarsa_state(make_observation())
        mask = np.asarray([True, True, True, False, False, False], dtype=bool)
        agent = SarsaAgent(epsilon=0.0)

        self.assertEqual(agent.choose_action(state, mask, fallback_action=0), 0)
        agent.q_table[state] = {0: 2.0, 1: 2.0, 2: 2.0}
        self.assertEqual(agent.choose_action(state, mask, fallback_action=0), 0)

    def test_fallback_selects_nearest_singleton_dispatch(self):
        observation = make_observation(action_mask=(True, True, True))
        observation["call_slots"][0] = {
            "call_id": 7,
            "coord": (11.0, 10.0),
        }

        class DispatchEnv:
            @staticmethod
            def encode_action(pairs):
                return 1

        mask = sarsa_action_mask(observation)
        self.assertEqual(nearest_available_action_id(DispatchEnv(), observation, mask), 1)

    def test_sarsa_update_matches_hand_calculated_example(self):
        agent = SarsaAgent(alpha=0.1, gamma=0.9, epsilon=0.0, epsilon_min=0.0)
        state, next_state = ("s0",), ("s1",)
        agent.q_table[state] = {3: 1.0}
        agent.q_table[next_state] = {4: 4.0}

        updated = agent.learn(state, 3, 2.0, next_state, 4, terminal=False)

        self.assertAlmostEqual(2.0 + 0.9 * 4.0, 5.6)
        self.assertAlmostEqual(1.0 + 0.1 * (5.6 - 1.0), 1.46)
        self.assertAlmostEqual(updated, 1.46)

    def test_sarsa_update_does_not_use_max_next_action(self):
        agent = SarsaAgent(alpha=0.1, gamma=0.9, epsilon=0.0, epsilon_min=0.0)
        state, next_state = ("s0",), ("s1",)
        agent.q_table[state] = {3: 1.0}
        agent.q_table[next_state] = {4: 4.0, 5: 100.0}

        updated = agent.learn(state, 3, 2.0, next_state, 4, terminal=False)

        self.assertAlmostEqual(updated, 1.46)
        self.assertNotAlmostEqual(updated, 1.0 + 0.1 * (2.0 + 0.9 * 100.0 - 1.0))

    def test_terminal_update_does_not_bootstrap(self):
        agent = SarsaAgent(alpha=0.2, gamma=0.9, epsilon=0.0, epsilon_min=0.0)
        state, next_state = ("s0",), ("terminal",)
        agent.q_table[state] = {3: 1.0}
        agent.q_table[next_state] = {5: 100.0}

        updated = agent.learn(state, 3, 3.0, next_state, None, terminal=True)

        self.assertAlmostEqual(updated, 1.4)

    def test_nonterminal_update_bootstraps_from_next_action(self):
        agent = SarsaAgent(alpha=0.2, gamma=0.5, epsilon=0.0, epsilon_min=0.0)

        updated = agent.learn(("s0",), 1, 2.0, ("s1",), 2, terminal=False)

        self.assertAlmostEqual(updated, 1.3)

    def test_optimistic_initialization_prefers_dispatch_over_noop(self):
        agent = SarsaAgent(epsilon=0.01, epsilon_min=0.0, seed=0)
        mask = np.asarray([True, True, True, False, False, False], dtype=bool)

        self.assertEqual(agent.choose_action(("new",), mask), 0)

    def test_dispatch_and_noop_reward_shaping(self):
        agent = SarsaAgent()
        observation = make_observation()
        observation["call_slots"][0] = {
            "call_id": 7,
            "coord": (10.0, 10.0),
        }
        dispatch_reward, dispatch_shaping = agent._shaped_reward(
            observation,
            0,
            2.0,
            {"decoded_action": [(1, 7)]},
            [(0.0, 10.0)],
        )
        noop_reward, noop_shaping = agent._shaped_reward(
            observation,
            5,
            2.0,
            {"decoded_action": []},
        )

        self.assertEqual(dispatch_reward, 12.0)
        self.assertEqual(dispatch_shaping, 10.0)
        self.assertEqual(noop_reward, -48.0)
        self.assertEqual(noop_shaping, -50.0)

    def test_reward_shaping_replaces_wait_failure_and_completion_costs(self):
        agent = SarsaAgent()
        observation = make_observation()
        info = {
            "reward_breakdown": {
                "wait_penalty": 2.0,
                "failure_penalty": 30.0,
                "completion_bonus": 70.0,
            },
            "wait_minutes": 10.0,
            "failed_call_count": 1,
            "completed_calls": [
                {"severity": 1, "is_perfect": True},
                {"severity": 2, "is_perfect": False},
            ],
        }

        shaped, shaping = agent._shaped_reward(observation, 5, 38.0, info)

        self.assertAlmostEqual(shaping, 15.25)
        self.assertAlmostEqual(shaped, 53.25)

    def test_training_logs_metrics_every_500_episodes(self):
        terminal_observation = make_observation(action_mask=(True, False, False))
        env = FakeEnv(
            [
                (
                    terminal_observation,
                    1.0,
                    True,
                    {"truncated": False},
                )
            ]
        )
        agent = SarsaAgent(epsilon=1.0, epsilon_min=0.05)
        output = io.StringIO()

        with redirect_stdout(output):
            agent.train(env, num_episodes=500, seed=10)

        self.assertIn("episodes=500", output.getvalue())
        self.assertIn("avg_reward_shaped=", output.getvalue())
        self.assertIn("served=0.00/0.00 calls", output.getvalue())
        self.assertIn("distance=0.00 km", output.getvalue())
        self.assertRegex(output.getvalue(), r"no_op=\d+\.\d+%")
        reset_seeds = [call[1] for call in env.calls if call[0] == "reset"]
        self.assertEqual(len(set(reset_seeds)), 500)
        self.assertAlmostEqual(agent.epsilon, 0.05)

    def test_epsilon_schedule_reaches_minimum_at_85_percent(self):
        terminal_observation = make_observation()
        env = FakeEnv(
            [(terminal_observation, 0.0, True, {"truncated": False})]
        )
        agent = SarsaAgent(epsilon=1.0, epsilon_min=0.02)

        agent.train(env, num_episodes=100, seed=10, progress_interval=100)

        self.assertGreater(agent.episode_records[83]["epsilon"], 0.02)
        self.assertAlmostEqual(agent.episode_records[84]["epsilon"], 0.02)
        self.assertAlmostEqual(agent.epsilon_decay, 0.02 ** (1 / 85))

    def test_epsilon_greedy_and_deterministic_greedy_tie_break(self):
        state = encode_sarsa_state(make_observation())
        mask = np.asarray([True, True, True, False, False, False], dtype=bool)
        greedy = SarsaAgent(epsilon=0.0, epsilon_min=0.0, seed=10)
        greedy.q_table[state] = {0: 4.0, 1: 4.0, 2: 4.0}
        self.assertEqual(greedy.choose_action(state, mask), 0)
        self.assertEqual(greedy.choose_action(state, mask), 0)

        exploring = SarsaAgent(epsilon=1.0, epsilon_min=0.0, seed=10)
        exploring.q_table[state] = {3: 1000.0}
        choices = {exploring.choose_action(state, mask) for _ in range(100)}
        self.assertEqual(choices, {0, 1, 2})

    def test_step_reward_is_used_and_reset_reward_is_accounted_separately(self):
        initial = make_observation(action_mask=(True, False, False))
        terminal_observation = make_observation(
            minute=20, action_mask=(True, False, False)
        )
        env = FakeEnv(
            [(terminal_observation, 7.25, True, {"truncated": False})],
            reset_reward=-3.0,
        )
        agent = SarsaAgent(alpha=1.0, gamma=0.9, epsilon=0.0, epsilon_min=0.0)
        state = encode_sarsa_state(initial)

        history = agent.train(env, num_episodes=1, seed=20)

        self.assertEqual(env.calls[0][0], "reset")
        self.assertEqual(env.calls[1], ("step", 0))
        self.assertEqual(agent.q_table[state][5], 7.25)
        self.assertEqual(history, [4.25])
        self.assertEqual(agent.episode_records[0]["reset_reward"], -3.0)
        self.assertEqual(agent.episode_records[0]["step_reward_total"], 7.25)

    def test_natural_termination_is_terminal_and_not_truncation(self):
        next_observation = make_observation(minute=20)
        env = FakeEnv(
            [(next_observation, 2.0, True, {"truncated": False, "backlog": None})]
        )
        agent = SarsaAgent(alpha=1.0, gamma=0.9, epsilon=0.0, epsilon_min=0.0)
        next_state = encode_sarsa_state(next_observation)
        agent.q_table[next_state] = {3: 100.0}

        agent.train(env, num_episodes=1)

        record = agent.episode_records[0]
        self.assertTrue(record["terminated"])
        self.assertFalse(record["truncated"])
        self.assertNotIn("backlog_counts", record)
        first_state = encode_sarsa_state(make_observation())
        self.assertEqual(agent.q_table[first_state][5], 2.0)

    def test_truncation_bootstraps_and_only_records_unresolved_backlog(self):
        initial = make_observation()
        final_observation = make_observation(minute=370, action_mask=(False, True, False))
        next_state = encode_sarsa_state(final_observation)
        env = FakeEnv(
            [
                (
                    final_observation,
                    1.0,
                    True,
                    {
                        "truncated": True,
                        "backlog": {
                            "pending_calls": [1],
                            "waiting_calls": [2, 3],
                            "at_scene_calls": [],
                            "en_route_hospital_ambulances": [4],
                        },
                    },
                )
            ]
        )
        agent = SarsaAgent(alpha=1.0, gamma=0.5, epsilon=0.0, epsilon_min=0.0)
        agent.q_table[next_state] = {5: 4.0, 2: 100.0}

        agent.train(env, num_episodes=1)

        first_state = encode_sarsa_state(initial)
        self.assertEqual(agent.q_table[first_state][5], 3.0)
        record = agent.episode_records[0]
        self.assertFalse(record["terminated"])
        self.assertTrue(record["truncated"])
        self.assertEqual(
            record["backlog_counts"],
            {"pending_calls": 1, "waiting_calls": 2, "at_scene_calls": 0},
        )

    def test_training_uses_only_reset_and_step_environment_api(self):
        next_observation = make_observation(minute=20)
        env = FakeEnv(
            [(next_observation, 1.0, True, {"truncated": False})]
        )
        agent = SarsaAgent(env=env, epsilon=0.0, epsilon_min=0.0, seed=12)

        agent.train(num_episodes=1)

        self.assertEqual(env.calls[0][0], "reset")
        self.assertEqual(env.calls[1], ("step", 0))

    def test_training_calls_g2_5_encoder(self):
        env = FakeEnv(
            [
                (
                    make_observation(minute=20),
                    1.0,
                    True,
                    {"truncated": False},
                )
            ]
        )
        agent = SarsaAgent(epsilon=0.0, epsilon_min=0.0)
        with patch(
            "dispatch_env.sarsa.encode_sarsa_state",
            wraps=encode_sarsa_state,
        ) as encoder:
            agent.train(env, num_episodes=1)

        self.assertGreaterEqual(encoder.call_count, 2)

    def test_safety_limit_is_recorded_separately_from_environment_done(self):
        env = FakeEnv(
            [
                (
                    make_observation(minute=20),
                    1.0,
                    False,
                    {"truncated": False},
                )
            ]
        )
        agent = SarsaAgent(epsilon=0.0, epsilon_min=0.0)

        agent.train(env, num_episodes=1, safety_limit=1)

        record = agent.episode_records[0]
        self.assertTrue(record["safety_limited"])
        self.assertFalse(record["terminated"])
        self.assertFalse(record["truncated"])

    def test_q_table_checkpoint_round_trip(self):
        state = encode_sarsa_state(make_observation())
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 5)
        self.assertTrue(all(isinstance(item, int) for item in state))
        agent = SarsaAgent(
            alpha=0.2,
            gamma=0.8,
            epsilon=0.7,
            epsilon_decay=0.9,
            epsilon_min=0.1,
            num_episodes=13,
            seed=42,
        )
        agent.epsilon = 0.3
        agent.episodes_completed = 4
        agent.episode_seeds = [111, 222, 333, 444]
        agent.q_table[state] = {0: 1.25, 2: -5.0}
        environment_config = {"num_ambulances": 6, "max_time": 360}

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "sarsa.json"
            agent.save_checkpoint(checkpoint_path, environment_config)
            with checkpoint_path.open("r", encoding="utf-8") as checkpoint_file:
                checkpoint = json.load(checkpoint_file)
            restored = SarsaAgent.load_checkpoint(checkpoint_path)

        self.assertEqual(checkpoint["state_encoder_version"], STATE_ENCODER_VERSION)
        self.assertEqual(len(restored.q_table), len(agent.q_table))
        self.assertEqual(
            sum(len(actions) for actions in restored.q_table.values()),
            sum(len(actions) for actions in agent.q_table.values()),
        )
        self.assertEqual(set(restored.q_table), set(agent.q_table))
        self.assertEqual(restored.q_table, agent.q_table)
        for original_state, original_actions in agent.q_table.items():
            restored_actions = restored.q_table[original_state]
            self.assertEqual(set(restored_actions), set(original_actions))
            for action, original_value in original_actions.items():
                self.assertEqual(restored_actions[action], original_value)
            self.assertIsInstance(original_state, tuple)
            restored_state = next(
                key for key in restored.q_table if key == original_state
            )
            self.assert_nested_state_key_uses_tuples(restored_state)
            self.assertEqual(restored_state, original_state)
            self.assertEqual(hash(restored_state), hash(original_state))
        self.assertEqual(
            restored.initial_epsilon, agent.initial_epsilon
        )
        self.assertEqual(restored.alpha, agent.alpha)
        self.assertEqual(restored.gamma, agent.gamma)
        self.assertEqual(restored.epsilon, agent.epsilon)
        self.assertEqual(restored.epsilon_decay, agent.epsilon_decay)
        self.assertEqual(restored.epsilon_min, agent.epsilon_min)
        self.assertEqual(restored.num_episodes, agent.num_episodes)
        self.assertEqual(restored.training_seed, 42)
        self.assertEqual(restored.episodes_completed, 4)
        self.assertEqual(restored.episode_seeds, agent.episode_seeds)
        self.assertEqual(restored.environment_config, environment_config)


if __name__ == "__main__":
    unittest.main()
