import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dispatch_env import SarsaAgent
from dispatch_env.call import EmergencyCall
from dispatch_env.sarsa_evaluator import (
    EVALUATION_SEEDS,
    ENVIRONMENT_CONFIG,
    aggregate_values,
    estimate_dispatch_eta_minutes,
    evaluate_checkpoint,
)
from dispatch_env.sarsa_state import STATE_ENCODER_VERSION, encode_sarsa_state
from dispatch_env.types import AmbulanceStatus, CallStatus


def make_observation(
    minute=0,
    action_mask=(True, True, False),
    last_step_reward=0.0,
    call_slots=None,
    edges_by_id=None,
):
    return {
        "episode": 0,
        "current_minute": minute,
        "edge_congestion_vector": [1],
        "edges_by_id": {} if edges_by_id is None else edges_by_id,
        "ambulances": {
            1: {
                "status": AmbulanceStatus.IDLE.value,
                "coord": (10.0, 10.0),
                "current_edge_id": 1,
                "distance_from_start": 2.0,
                "total_distance_traveled": 0.0,
            }
        },
        "active_calls": [],
        "call_slots": [None, None, None] if call_slots is None else call_slots,
        "action_mask": np.asarray(action_mask, dtype=bool),
        "last_step_reward": last_step_reward,
    }


def zero_breakdown():
    return {
        "wait_penalty": 0.0,
        "failure_penalty": 0.0,
        "distance_penalty": 0.0,
        "completion_bonus": 0.0,
        "total": 0.0,
    }


class FakeEvaluationEnv:
    def __init__(
        self,
        config,
        seed,
        transitions=None,
        calls=None,
        initial_observation=None,
        decoded_pairs=None,
    ):
        self.config = dict(config)
        self.seed = seed
        self.transitions = (
            [(make_observation(minute=1), 0.0, True, {
                "truncated": False,
                "backlog": None,
                "reward_breakdown": zero_breakdown(),
            })]
            if transitions is None
            else transitions
        )
        self.calls = {} if calls is None else calls
        self.initial_observation = (
            make_observation() if initial_observation is None else initial_observation
        )
        self.decoded_pairs = [] if decoded_pairs is None else decoded_pairs
        self.reset_seeds = []
        self.api_calls = []
        self.decoded_action_ids = []
        self.actions = []
        self.transition_index = 0

    def reset(self, seed=None):
        self.api_calls.append("reset")
        self.reset_seeds.append(seed)
        self.transition_index = 0
        return copy.deepcopy(self.initial_observation)

    def decode_action(self, action):
        self.api_calls.append("decode_action")
        self.decoded_action_ids.append(action)
        return list(self.decoded_pairs)

    def encode_action(self, pairs):
        self.api_calls.append("encode_action")
        return 1 if pairs else 0

    def step(self, action):
        self.api_calls.append("step")
        self.actions.append(action)
        observation, reward, done, info = self.transitions[self.transition_index]
        self.transition_index += 1
        return copy.deepcopy(observation), reward, done, copy.deepcopy(info)


class TestSarsaEvaluator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint_path = Path(self.temp_dir.name) / "trained_sarsa.json"
        self.observation = make_observation()
        self.state = encode_sarsa_state(self.observation)
        self.agent = SarsaAgent(
            alpha=0.1,
            gamma=0.99,
            epsilon=0.6,
            epsilon_decay=0.995,
            epsilon_min=0.05,
            num_episodes=5,
            seed=42,
        )
        self.agent.episodes_completed = 5
        self.agent.episode_seeds = [100000 + seed for seed in range(5)]
        self.agent.q_table[self.state] = {0: 1.0, 1: 0.5, 2: -3.0, 3: -1.0}
        self.agent.save_checkpoint(self.checkpoint_path, ENVIRONMENT_CONFIG)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _factory(self, **kwargs):
        environments = []

        def create(config, seed):
            env = FakeEvaluationEnv(config, seed, **kwargs)
            environments.append(env)
            return env

        return create, environments

    def test_evaluation_preserves_q_table_and_never_calls_learn(self):
        create_env, _ = self._factory()
        with patch.object(
            SarsaAgent,
            "learn",
            side_effect=AssertionError("evaluation called learn"),
        ) as learn:
            report = evaluate_checkpoint(
                self.checkpoint_path,
                environment_factory=create_env,
            )

        learn.assert_not_called()
        verification = report["verification"]
        self.assertTrue(verification["q_table_unchanged"])
        self.assertEqual(
            verification["q_table_states_before"],
            verification["q_table_states_after"],
        )
        self.assertEqual(
            verification["q_table_entries_before"],
            verification["q_table_entries_after"],
        )
        self.assertEqual(verification["learning_updates"], 0)
        self.assertEqual(
            verification["agent_epsilon_before"],
            verification["agent_epsilon_after"],
        )

    def test_greedy_policy_respects_masked_action(self):
        initial = make_observation(
            call_slots=[
                {
                    "call_id": 12,
                    "severity": 2,
                    "coord": (11.0, 10.0),
                    "wait_time": 0.0,
                    "failure_time": 40.0,
                    "deadline": 40.0,
                },
                None,
                None,
            ],
        )
        state = encode_sarsa_state(initial)
        self.agent.q_table[state] = {0: 1.0, 1: 5.0, 2: 1000.0, 3: -1.0}
        self.agent.save_checkpoint(self.checkpoint_path, ENVIRONMENT_CONFIG)
        create_env, environments = self._factory(initial_observation=initial)

        evaluate_checkpoint(
            self.checkpoint_path,
            environment_factory=create_env,
        )

        self.assertEqual(len(environments), 10)
        self.assertTrue(all(env.actions == [1] for env in environments))

    def test_unseen_dispatchable_state_uses_nearest_fallback(self):
        initial = make_observation(
            action_mask=(True, True, False),
            call_slots=[
                {
                    "call_id": 12,
                    "severity": 2,
                    "coord": (11.0, 10.0),
                    "wait_time": 0.0,
                    "failure_time": 40.0,
                    "deadline": 40.0,
                },
                None,
                None,
            ],
        )
        create_env, environments = self._factory(initial_observation=initial)

        evaluate_checkpoint(
            self.checkpoint_path,
            environment_factory=create_env,
        )

        self.assertEqual(len(environments), 10)
        self.assertTrue(all(env.actions == [1] for env in environments))

    def test_incompatible_checkpoint_is_rejected(self):
        with self.checkpoint_path.open("r", encoding="utf-8") as file:
            original = json.load(file)
        mutations = (
            ("algorithm", "DQN"),
            ("state_encoder_version", "obsolete-encoder"),
            ("action_space_version", "obsolete-actions"),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                invalid = copy.deepcopy(original)
                invalid[field] = value
                path = Path(self.temp_dir.name) / "invalid_{}.json".format(field)
                with path.open("w", encoding="utf-8") as file:
                    json.dump(invalid, file)
                with self.assertRaises(ValueError):
                    evaluate_checkpoint(path, environment_factory=self._factory()[0])

        invalid = copy.deepcopy(original)
        invalid["training"]["environment_config"]["max_time"] += 1
        path = Path(self.temp_dir.name) / "invalid_environment.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(invalid, file)
        with self.assertRaisesRegex(ValueError, "Environment config is incompatible"):
            evaluate_checkpoint(path, environment_factory=self._factory()[0])

    def test_ten_fixed_seeds_are_used_and_recorded(self):
        create_env, environments = self._factory()

        report = evaluate_checkpoint(
            self.checkpoint_path,
            environment_factory=create_env,
        )

        self.assertEqual(report["evaluation_seeds"], list(EVALUATION_SEEDS))
        self.assertEqual([episode["seed"] for episode in report["episodes"]], list(EVALUATION_SEEDS))
        self.assertEqual([env.reset_seeds for env in environments], [[seed] for seed in EVALUATION_SEEDS])
        self.assertEqual(report["encoder_version"], STATE_ENCODER_VERSION)
        self.assertEqual(report["environment_config"], ENVIRONMENT_CONFIG)
        self.assertEqual(report["training_config"]["seed"], 42)
        self.assertEqual(report["training_config"]["episodes_completed"], 5)
        self.assertEqual(report["training_config"]["alpha"], 0.1)
        self.assertEqual(report["training_config"]["gamma"], 0.99)
        self.assertTrue(
            set(report["evaluation_seeds"]).isdisjoint(
                range(42, 42 + report["training_config"]["episodes_completed"])
            )
        )

    def test_aggregation_uses_mean_population_std_min_and_max(self):
        aggregate = aggregate_values([1.0, 2.0, 3.0])

        self.assertEqual(aggregate["n"], 3)
        self.assertEqual(aggregate["mean"], 2.0)
        self.assertAlmostEqual(aggregate["std"], math.sqrt(2.0 / 3.0))
        self.assertEqual(aggregate["min"], 1.0)
        self.assertEqual(aggregate["max"], 3.0)
        self.assertEqual(
            aggregate_values([None, None]),
            {"n": 0, "mean": None, "std": None, "min": None, "max": None},
        )

    def test_truncation_calls_are_unresolved_not_failed(self):
        def create(config, seed):
            call = EmergencyCall(
                call_id=100 + seed,
                edge=(1, 2),
                distance_from_start=1.0,
                coord=(10.0, 10.0),
                severity=3,
                created_time=0.0,
                status=CallStatus.AT_SCENE,
                arrived_at_scene_time=1.0,
            )
            final_info = {
                "truncated": True,
                "backlog": {
                    "pending_calls": [],
                    "waiting_calls": [],
                    "at_scene_calls": [call],
                },
                "reward_breakdown": zero_breakdown(),
            }
            return FakeEvaluationEnv(
                config,
                seed,
                transitions=[
                    (make_observation(minute=1), 0.0, True, final_info)
                ],
                calls={call.call_id: call},
            )

        report = evaluate_checkpoint(
            self.checkpoint_path,
            environment_factory=create,
        )

        for episode in report["episodes"]:
            self.assertTrue(episode["truncated"])
            self.assertEqual(episode["unresolved_calls"], 1)
            self.assertEqual(episode["failed_calls"], 0)
            self.assertEqual(episode["total_generated_calls"], 1)

    def test_reward_components_are_summed_from_step_info(self):
        first_breakdown = {
            "wait_penalty": 1.0,
            "failure_penalty": 0.0,
            "distance_penalty": 0.5,
            "completion_bonus": 0.0,
            "total": -1.5,
        }
        second_breakdown = {
            "wait_penalty": 0.0,
            "failure_penalty": 5.0,
            "distance_penalty": 1.0,
            "completion_bonus": 10.0,
            "total": 4.0,
        }
        transitions = [
            (
                make_observation(minute=1),
                -1.5,
                False,
                {"truncated": False, "reward_breakdown": first_breakdown},
            ),
            (
                make_observation(minute=2),
                4.0,
                True,
                {"truncated": False, "reward_breakdown": second_breakdown},
            ),
        ]
        create_env, _ = self._factory(
            transitions=transitions,
            initial_observation=make_observation(last_step_reward=-2.0),
        )

        report = evaluate_checkpoint(
            self.checkpoint_path,
            environment_factory=create_env,
        )

        for episode in report["episodes"]:
            self.assertEqual(episode["reward_breakdown"]["wait_penalty"], 1.0)
            self.assertEqual(episode["reward_breakdown"]["failure_penalty"], 5.0)
            self.assertEqual(episode["reward_breakdown"]["distance_penalty"], 1.5)
            self.assertEqual(episode["reward_breakdown"]["completion_bonus"], 10.0)
            self.assertEqual(episode["reward_breakdown"]["total"], 2.5)
            self.assertEqual(episode["episode_return"], 0.5)
        self.assertEqual(report["summary"]["reward_total"]["mean"], 2.5)

    def test_dispatch_slack_uses_pre_dispatch_observation(self):
        edges = {
            1: {
                "edge_id": 1,
                "from_node": 1,
                "to_node": 2,
                "length": 5.0,
                "speed": 60.0,
            },
            2: {
                "edge_id": 2,
                "from_node": 2,
                "to_node": 3,
                "length": 4.0,
                "speed": 30.0,
            },
        }
        pending_call = {
            "call_id": 101,
            "id": 101,
            "severity": 1,
            "coord": (10.0, 10.0),
            "edge_id": 2,
            "distance_from_start": 1.0,
            "wait_time": 0.0,
            "failure_time": 60.0,
            "deadline": 25.0,
        }
        initial = make_observation(
            minute=10,
            call_slots=[pending_call, None, None],
            edges_by_id=edges,
        )
        call = EmergencyCall(
            call_id=101,
            edge=(2, 3),
            distance_from_start=1.0,
            coord=(10.0, 10.0),
            severity=1,
            created_time=0.0,
            status=CallStatus.COMPLETED,
            arrived_at_scene_time=14.0,
            completed_time=18.0,
        )
        transitions = [
            (
                make_observation(minute=11),
                0.0,
                True,
                {"truncated": False, "reward_breakdown": zero_breakdown()},
            )
        ]
        create_env, _ = self._factory(
            transitions=transitions,
            calls={101: call},
            initial_observation=initial,
            decoded_pairs=[(1, 101)],
        )
        state = encode_sarsa_state(initial)
        self.agent.q_table = {state: {0: 0.0, 1: 10.0, 2: 1000.0}}
        self.agent.save_checkpoint(self.checkpoint_path, ENVIRONMENT_CONFIG)

        eta = estimate_dispatch_eta_minutes(
            initial,
            initial["ambulances"][1],
            pending_call,
        )
        report = evaluate_checkpoint(
            self.checkpoint_path,
            environment_factory=create_env,
        )

        self.assertEqual(eta, 5.0)
        for episode in report["episodes"]:
            self.assertEqual(episode["avg_slack_at_dispatch"], 10.0)

    def test_evaluation_is_reproducible_with_same_checkpoint_and_seeds(self):
        create_first, _ = self._factory()
        create_second, _ = self._factory()

        first = evaluate_checkpoint(
            self.checkpoint_path,
            environment_factory=create_first,
        )
        second = evaluate_checkpoint(
            self.checkpoint_path,
            environment_factory=create_second,
        )

        self.assertEqual(first["evaluation_seeds"], second["evaluation_seeds"])
        self.assertEqual(first["episodes"], second["episodes"])
        self.assertEqual(first["summary"], second["summary"])

    def test_environment_is_used_only_through_read_api_and_episode_api(self):
        create_env, environments = self._factory()

        evaluate_checkpoint(
            self.checkpoint_path,
            environment_factory=create_env,
        )

        self.assertTrue(environments)
        allowed = {"reset", "decode_action", "encode_action", "step"}
        for env in environments:
            self.assertTrue(set(env.api_calls).issubset(allowed))
            self.assertEqual(env.api_calls[0], "reset")
            self.assertEqual(env.api_calls[-1], "step")


if __name__ == "__main__":
    unittest.main()
