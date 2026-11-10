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


if __name__ == "__main__":
    unittest.main()
