from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


class SarsaAgent:
    """On-policy SARSA controller for environments exposing valid action masks.

    The implementation is intentionally generic: it only assumes the environment
    provides:
      - `reset()` -> state
      - `get_action_mask()` -> 1D boolean mask over valid discrete actions
      - `step(action)` -> (next_state, reward, done, info)

    This keeps the SARSA logic independent from the ambulance simulation details
    while reusing the existing environment API.
    """

    def __init__(
        self,
        env: Any = None,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        num_episodes: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        self.env = env
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.num_episodes = int(num_episodes)
        self.rng = np.random.default_rng(seed)
        self.q_table: Dict[Tuple[Any, ...], Dict[int, float]] = defaultdict(dict)
        self.training_history: List[float] = []

    @staticmethod
    def _make_hashable(value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(sorted((k, SarsaAgent._make_hashable(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(SarsaAgent._make_hashable(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted(SarsaAgent._make_hashable(v) for v in value))
        if isinstance(value, np.ndarray):
            return tuple(value.tolist())
        return value

    def state_key(self, state: Any) -> Tuple[Any, ...]:
        return self._make_hashable(state)

    def _valid_actions_from_env(self, state: Any = None) -> List[int]:
        if self.env is None:
            return []
        mask = self.env.get_action_mask()
        if isinstance(mask, np.ndarray):
            return [int(idx) for idx in np.flatnonzero(mask)]
        return [int(idx) for idx, is_valid in enumerate(mask) if bool(is_valid)]

    def choose_action(
        self,
        state: Any,
        valid_actions: Optional[Iterable[int]] = None,
        epsilon: Optional[float] = None,
    ) -> int:
        """Choose a valid action using epsilon-greedy selection."""
        if valid_actions is None:
            valid_actions = self._valid_actions_from_env(state)
        valid_list = list(valid_actions)
        if not valid_list:
            return 0

        effective_epsilon = self.epsilon if epsilon is None else float(epsilon)
        state_id = self.state_key(state)
        q_values = self.q_table.setdefault(state_id, {})

        if self.rng.random() < effective_epsilon:
            return int(self.rng.choice(valid_list))

        max_value = max(q_values.get(action, 0.0) for action in valid_list)
        best_actions = [
            action for action in valid_list if abs(q_values.get(action, 0.0) - max_value) <= 1e-12
        ]
        if not best_actions:
            return int(self.rng.choice(valid_list))
        return int(self.rng.choice(best_actions))

    def learn(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        next_action: Optional[int],
        done: bool = False,
    ) -> float:
        """Apply the SARSA update using the next chosen action A'."""
        state_id = self.state_key(state)
        next_state_id = self.state_key(next_state)

        q_state = self.q_table.setdefault(state_id, {})
        q_next_state = self.q_table.setdefault(next_state_id, {})

        current_value = q_state.get(action, 0.0)
        next_value = 0.0 if next_action is None or done else q_next_state.get(next_action, 0.0)
        target = reward + self.gamma * next_value
        updated = current_value + self.alpha * (target - current_value)
        q_state[action] = updated
        return updated

    def train(
        self,
        env: Any = None,
        num_episodes: Optional[int] = None,
        max_steps: int = 200,
    ) -> List[float]:
        """Train the SARSA policy for a number of episodes."""
        if env is not None:
            self.env = env
        if num_episodes is None:
            num_episodes = self.num_episodes

        history: List[float] = []
        for _ in range(int(num_episodes)):
            state = self.env.reset()
            valid_actions = self._valid_actions_from_env(state)
            action = self.choose_action(state, valid_actions)

            episode_reward = 0.0
            done = False
            step_count = 0
            while not done and step_count < max_steps:
                next_state, reward, done, _ = self.env.step(action)
                next_valid_actions = self._valid_actions_from_env(next_state)
                next_action = self.choose_action(next_state, next_valid_actions) if not done else None
                self.learn(state, action, reward, next_state, next_action, done)
                episode_reward += float(reward)

                state = next_state
                action = next_action if next_action is not None else 0
                step_count += 1

            history.append(episode_reward)
            self.training_history.append(episode_reward)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return history

    def evaluate(
        self,
        env: Any = None,
        episodes: int = 10,
        greedy: bool = True,
        max_steps: int = 200,
    ) -> List[float]:
        """Run evaluation episodes with a greedy policy derived from the learned Q-table."""
        if env is not None:
            self.env = env

        rewards: List[float] = []
        for _ in range(int(episodes)):
            state = self.env.reset()
            total_reward = 0.0
            done = False
            step_count = 0

            while not done and step_count < max_steps:
                valid_actions = self._valid_actions_from_env(state)
                if not valid_actions:
                    break
                if greedy:
                    action = self.choose_action(state, valid_actions, epsilon=0.0)
                else:
                    action = self.choose_action(state, valid_actions)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += float(reward)
                state = next_state
                step_count += 1

            rewards.append(total_reward)

        return rewards


class SARSAAgent(SarsaAgent):
    """Backward-compatible alias for the SARSA implementation."""

    pass


__all__ = ["SarsaAgent", "SARSAAgent"]
