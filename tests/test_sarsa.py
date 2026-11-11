import unittest

import numpy as np

from dispatch_env import AmbulanceDispatchEnv, SarsaAgent


class TestSarsaAgent(unittest.TestCase):
    def test_sarsa_update_uses_next_action_value(self):
        agent = SarsaAgent(alpha=0.2, gamma=0.9, epsilon=0.0, seed=7)

        state = "s0"
        next_state = "s1"
        action = 3
        next_action = 5
        reward = 1.5

        agent.q_table[state] = {action: 0.0}
        agent.q_table[next_state] = {next_action: 2.0}

        updated = agent.learn(state, action, reward, next_state, next_action, done=False)
        expected = 0.0 + 0.2 * (1.5 + 0.9 * 2.0 - 0.0)

        self.assertAlmostEqual(updated, expected)
        self.assertAlmostEqual(agent.q_table[state][action], expected)

    def test_choose_action_respects_valid_mask(self):
        env = AmbulanceDispatchEnv(seed=42)
        env.reset()

        agent = SarsaAgent(alpha=0.1, gamma=0.9, epsilon=0.2, seed=123)
        valid_actions = np.flatnonzero(env.get_action_mask())

        chosen = agent.choose_action(env.get_observation(), valid_actions)
        self.assertIn(chosen, valid_actions)

    def test_training_runs_and_decays_epsilon(self):
        env = AmbulanceDispatchEnv(seed=42)
        agent = SarsaAgent(env=env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.1, num_episodes=5, seed=99)

        history = agent.train(num_episodes=5, max_steps=20)

        self.assertEqual(len(history), 5)
        self.assertLessEqual(agent.epsilon, 1.0)
        self.assertGreaterEqual(agent.epsilon, 0.1)
