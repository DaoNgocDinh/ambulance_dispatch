"""Kiểm thử nhỏ cho agent Q-learning."""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from dispatch_env import AmbulanceDispatchEnv
from map_env.node import MapNode
from q_learning import QLearningAgent
from ambulance_q_learning_adapter import AmbulanceDispatchAdapter


class TestQLearningAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.env = AmbulanceDispatchEnv(seed=123, max_time=5, max_overtime=0)
        self.agent = QLearningAgent(seed=7)
        self.adapter = AmbulanceDispatchAdapter()

    def test_selects_only_valid_action(self) -> None:
        state = self.adapter.encode_state(self.env)
        valid_actions = self.adapter.valid_actions(self.env)
        action = self.agent.choose_action(state, valid_actions, explore=False)
        self.assertIn(action, valid_actions)

    def test_same_environment_has_same_state_key(self) -> None:
        other_env = AmbulanceDispatchEnv(seed=123, max_time=5, max_overtime=0)

        self.assertEqual(
            self.adapter.encode_state(self.env),
            self.adapter.encode_state(other_env),
        )

    def test_changed_node_count_uses_a_different_state_namespace(self) -> None:
        original_state = self.adapter.encode_state(self.env)
        self.env.city_map.nodes[81] = MapNode(node_id=81, x=40.0, y=40.0)
        self.env.city_map.num_nodes = 81

        changed_state = self.adapter.encode_state(self.env)

        self.assertNotEqual(original_state, changed_state)

    def test_changed_action_space_uses_a_different_state_namespace(self) -> None:
        other_env = AmbulanceDispatchEnv(
            seed=123,
            num_ambulances=4,
            max_time=5,
            max_overtime=0,
        )

        self.assertNotEqual(
            self.adapter.encode_state(self.env),
            self.adapter.encode_state(other_env),
        )

    def test_q_learning_core_has_no_project_environment_imports(self) -> None:
        source = Path(__file__).parents[1] / "q_learning.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported_modules = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }

        self.assertFalse({"dispatch_env", "map_env"} & imported_modules)

    def test_q_learning_update(self) -> None:
        self.agent.alpha = 0.5
        self.agent.gamma = 0.9
        self.agent.q_table[("s", 1)] = 10.0
        self.agent.q_table[("s2", 2)] = 20.0
        self.agent.q_table[("s2", 3)] = 5.0

        self.agent.update("s", 1, 2.0, "s2", [2, 3], done=False)

        self.assertAlmostEqual(self.agent.q_table[("s", 1)], 15.0)

    def test_terminal_update_does_not_use_future_value(self) -> None:
        self.agent.alpha = 0.5
        self.agent.gamma = 0.9
        self.agent.q_table[("s", 1)] = 10.0
        self.agent.q_table[("s2", 2)] = 100.0

        self.agent.update("s", 1, 2.0, "s2", [2], done=True)

        self.assertAlmostEqual(self.agent.q_table[("s", 1)], 6.0)

    def test_q_table_can_be_saved_and_loaded(self) -> None:
        self.agent.q_table[("trạng_thái", 8)] = 3.5
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "q_table.json"
            self.agent.save(path)
            loaded_agent = QLearningAgent()
            loaded_agent.load(path)
        self.assertEqual(loaded_agent.q_table, {("trạng_thái", 8): 3.5})

    def test_candidate_actions_keeps_wait_and_partial_dispatch(self) -> None:
        env = AmbulanceDispatchEnv(seed=42)
        obs = env.reset(seed=42)
        for _ in range(120):
            slots = [c for c in obs["call_slots"] if c is not None]
            if len(slots) >= 2 and len(self.adapter.get_idle_ambulances_from_obs(obs)) >= 2:
                break
            obs, _, done, _ = env.step(action=0)
            if done:
                break

        all_valid = self.adapter.valid_actions(env, obs)
        candidates = self.adapter.candidate_actions(env, obs)
        self.assertIn(0, candidates)
        self.assertTrue(set(candidates).issubset(set(all_valid)))
        decoded_sizes = {len(env.decode_action(aid)) for aid in candidates}
        self.assertIn(1, decoded_sizes)

    def test_high_sev_kpi_none_on_empty_and_stats_excludes_missing(self) -> None:
        from train_q_learning import _stats, run_episode

        empty_env = AmbulanceDispatchEnv(seed=123, max_time=1, max_overtime=0, poisson_lambda=0.0)
        row = run_episode(empty_env, self.adapter, self.agent, training=False, seed=123)
        self.assertIsNone(row["high_sev_served_rate"])
        self.assertIsNone(row["high_sev_failed_rate"])
        self.assertIsNone(row["high_sev_response_time"])

        mean_v, _, min_v, max_v, n_valid = _stats([10.0, None, float("nan"), 20.0])
        self.assertEqual(n_valid, 2)
        self.assertAlmostEqual(mean_v, 15.0)
        self.assertAlmostEqual(min_v, 10.0)
        self.assertAlmostEqual(max_v, 20.0)

    def test_correctness_gate_includes_g3_and_g6(self) -> None:
        from train_q_learning import run_episode, verify_correctness_gate

        make_env = lambda sd: AmbulanceDispatchEnv(seed=sd)
        run_episode(make_env(123), self.adapter, self.agent, training=True, seed=123)
        q_snap = dict(self.agent.q_table)
        v_snap = dict(self.agent.visit_counts)
        ev_row = run_episode(make_env(124), self.adapter, self.agent, training=False, seed=124)
        ev_row["strategy"] = "Q-learning"
        checks = verify_correctness_gate(
            make_env=make_env,
            adapter=self.adapter,
            agent=self.agent,
            evaluation_rows=[ev_row],
            q_table_before_eval=q_snap,
            visits_before_eval=v_snap,
            repro_seed=124,
        )
        self.assertEqual(checks["G3_learning_update_correctness"][0], "PASS")
        self.assertEqual(checks["G6_env_static_and_config"][0], "PASS")
        self.assertEqual(checks["G6_env_git_working_tree"][0], "PASS")

        tampered_checks = verify_correctness_gate(
            make_env=lambda sd: AmbulanceDispatchEnv(seed=sd, max_time=10, max_overtime=0),
            adapter=self.adapter,
            agent=self.agent,
            evaluation_rows=[ev_row],
            q_table_before_eval=q_snap,
            visits_before_eval=v_snap,
            repro_seed=124,
        )
        self.assertEqual(tampered_checks["G6_env_static_and_config"][0], "FAIL")


if __name__ == "__main__":
    unittest.main()
