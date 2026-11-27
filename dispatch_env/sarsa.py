"""Sparse tabular on-policy SARSA agent."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from dispatch_env.sarsa_state import (
    STATE_ENCODER_VERSION,
    StateKey,
    encode_sarsa_state,
)
from dispatch_env.types import AmbulanceStatus, CallStatus, REWARD_CONFIG
DISPATCH_Q_INITIAL = 1.0
NO_OP_Q_INITIAL = -1.0
DISPATCH_SHAPING_SCALE = 10.0
NO_OP_SHAPING_PENALTY = 50.0
META_ACTION_COUNT = 6
EDGE_FIRST_META_ACTION = 3
IDLE_REPOSITION_META_ACTION = 4
NO_OP_META_ACTION = 5
META_ACTION_VERSION = "dispatch-meta-actions-v2"
WAIT_PENALTY_PER_MINUTE = 0.175
FAILURE_PENALTY = 40.0
PERFECT_COMPLETION_BONUS = 80.0
ON_TIME_COMPLETION_BONUS = 15.0
LOAD_BALANCE_PERFECT_MISS_PENALTY = 25.0


def _idle_ambulance_candidates(
    env: Any,
    observation: Mapping[str, Any],
    call: Mapping[str, Any],
) -> List[Tuple[float, float, int]]:
    """Return (center distance, ETA, ambulance id) for idle units."""
    city_map = getattr(env, "city_map", None)
    calls = getattr(env, "calls", None)
    if city_map is None or not isinstance(calls, Mapping):
        return []
    call_id = int(call.get("call_id", call.get("id")))
    call_object = calls.get(call_id)
    if call_object is None:
        return []
    now = float(observation.get("current_minute", 0.0))
    perfect_threshold = float(
        call.get(
            "perfect_threshold",
            float(call.get("created_time", now))
            + 0.5 * float(call.get("failure_time", 0.0)),
        )
    )
    perfect_remaining = perfect_threshold - now
    center_coord = observation.get("city_center_coord", (18.0, 16.0))
    candidates = []
    for ambulance_id, details in observation.get("ambulances", {}).items():
        if not isinstance(details, Mapping):
            continue
        status = details.get("status")
        status_value = status.value if isinstance(status, AmbulanceStatus) else status
        if (
            status_value != AmbulanceStatus.IDLE.value
            or details.get("current_call_id") is not None
        ):
            continue
        ambulance = getattr(env, "ambulances", {}).get(int(ambulance_id))
        if ambulance is None:
            continue
        route = city_map.route_from_road_to_road(
            start_edge=ambulance.current_edge,
            start_dist=ambulance.distance_from_start,
            dest_edge=call_object.edge,
            dest_dist=call_object.distance_from_start,
            by_travel_time=True,
        )
        eta = float(route["total_time_minutes"])
        if eta <= perfect_remaining:
            coord = details.get("coord", center_coord)
            center_distance = math.hypot(
                float(coord[0]) - float(center_coord[0]),
                float(coord[1]) - float(center_coord[1]),
            )
            candidates.append((center_distance, eta, int(ambulance_id)))
    return candidates


def sarsa_action_mask(
    observation: Mapping[str, Any],
    env: Any = None,
) -> np.ndarray:
    """Mask the dispatch, edge-first, reposition, and no-op meta-actions."""
    try:
        env_mask = np.asarray(observation["action_mask"])
        ambulances = observation["ambulances"]
        call_slots = observation["call_slots"]
    except KeyError as exc:
        raise ValueError(f"Observation is missing {exc.args[0]}") from exc
    if env_mask.ndim != 1 or env_mask.dtype.kind != "b" or len(env_mask) == 0:
        raise ValueError("action_mask must be a non-empty one-dimensional boolean array")
    if not isinstance(ambulances, Mapping) or not isinstance(call_slots, (list, tuple)):
        raise ValueError("Observation must contain ambulance and call-slot collections")

    idle_count = 0
    for details in ambulances.values():
        if not isinstance(details, Mapping):
            raise ValueError("Observation ambulance values must be mappings")
        status = details.get("status")
        status_value = status.value if isinstance(status, AmbulanceStatus) else status
        if (
            status_value == AmbulanceStatus.IDLE.value
            and details.get("current_call_id") is None
        ):
            idle_count += 1
    pending_available = any(call is not None for call in call_slots)
    if idle_count and pending_available:
        if not np.any(env_mask[1:]):
            raise ValueError("Environment exposes no valid dispatch action")
        mask = np.asarray([True, True, True, False, False, False], dtype=bool)
        call = _pending_call(observation)
        if (
            env is not None
            and call is not None
            and len(_idle_ambulance_candidates(env, observation, call)) >= 2
        ):
            mask[EDGE_FIRST_META_ACTION] = True
        return mask
    if not pending_available and env is not None:
        can_reposition = getattr(env, "can_reposition_idle_ambulance", None)
        if callable(can_reposition) and can_reposition():
            return np.asarray([False, False, False, False, True, True], dtype=bool)
    return np.asarray([False, False, False, False, False, True], dtype=bool)


def _pending_call(observation: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    calls = [
        call for call in observation.get("call_slots", ())
        if isinstance(call, Mapping)
    ]
    if not calls:
        return None
    return min(
        calls,
        key=lambda call: (
            -int(call.get("severity", 1)),
            float(call.get("deadline", float("inf"))),
            float(call.get("created_time", 0.0)),
        ),
    )


def resolve_meta_action(
    env: Any,
    observation: Mapping[str, Any],
    meta_action: int,
) -> Tuple[int, List[Tuple[int, int]]]:
    """Translate a strategic action into one legal Environment dispatch."""
    if meta_action in (IDLE_REPOSITION_META_ACTION, NO_OP_META_ACTION):
        if meta_action == IDLE_REPOSITION_META_ACTION:
            request_reposition = getattr(env, "request_idle_reposition", None)
            if not callable(request_reposition):
                raise TypeError("Environment must provide request_idle_reposition")
            if not request_reposition():
                raise RuntimeError("The idle reposition action is no longer available")
        return 0, []
    if meta_action not in (0, 1, 2, EDGE_FIRST_META_ACTION):
        raise ValueError("meta_action is not a supported dispatch action")

    call = _pending_call(observation)
    ambulances = observation.get("ambulances")
    if call is None or not isinstance(ambulances, Mapping):
        raise ValueError("dispatch meta-action requires a pending call and ambulances")

    call_coord = call.get("coord")
    if not isinstance(call_coord, Sequence) or len(call_coord) != 2:
        raise ValueError("pending call has an invalid coordinate")
    candidates = []
    for ambulance_id, details in ambulances.items():
        if not isinstance(details, Mapping):
            raise ValueError("Observation ambulance values must be mappings")
        status = details.get("status")
        status_value = status.value if isinstance(status, AmbulanceStatus) else status
        if (
            status_value != AmbulanceStatus.IDLE.value
            or details.get("current_call_id") is not None
        ):
            continue
        ambulance_coord = details.get("coord")
        if not isinstance(ambulance_coord, Sequence) or len(ambulance_coord) != 2:
            raise ValueError("idle ambulance has an invalid coordinate")
        distance = math.hypot(
            float(ambulance_coord[0]) - float(call_coord[0]),
            float(ambulance_coord[1]) - float(call_coord[1]),
        )
        driven = float(details.get("total_distance_traveled", 0.0))
        candidates.append((distance, driven, int(ambulance_id)))
    if not candidates:
        raise ValueError("dispatch meta-action requires an idle ambulance")

    if meta_action == 2:
        ambulance_id = min(candidates, key=lambda candidate: (candidate[1], candidate[2]))[2]
    elif meta_action == EDGE_FIRST_META_ACTION:
        perfect_candidates = _idle_ambulance_candidates(env, observation, call)
        if len(perfect_candidates) < 2:
            raise RuntimeError("Edge-first requires two units able to respond perfectly")
        ambulance_id = max(
            perfect_candidates,
            key=lambda candidate: (candidate[0], -candidate[1], -candidate[2]),
        )[2]
    else:
        candidates.sort(key=lambda candidate: (candidate[0], candidate[2]))
        ambulance_id = candidates[min(meta_action, len(candidates) - 1)][2]
    call_id = int(call.get("call_id", call.get("id")))
    pairs = [(ambulance_id, call_id)]
    encode_action = getattr(env, "encode_action", None)
    if not callable(encode_action):
        raise TypeError("Environment must provide encode_action for dispatch meta-actions")
    return int(encode_action(pairs)), pairs


def nearest_available_action_id(
    env: Any,
    observation: Mapping[str, Any],
    action_mask: Any,
) -> int:
    """Return the Environment action for the nearest-available meta-action."""
    mask = np.asarray(action_mask)
    if mask.ndim != 1 or len(mask) != META_ACTION_COUNT or mask.dtype.kind != "b":
        raise ValueError("SARSA meta-action mask must contain six booleans")
    if not mask[0]:
        if mask[NO_OP_META_ACTION] or mask[IDLE_REPOSITION_META_ACTION]:
            return 0
        raise ValueError("nearest-available meta-action is masked")
    environment_action, _ = resolve_meta_action(env, observation, 0)
    return environment_action


class SarsaAgent:
    """Train a tabular SARSA policy using observations from the environment API."""

    def __init__(
        self,
        env: Any = None,
        alpha: float = 0.15,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.02,
        num_episodes: int = 3000,
        seed: Optional[int] = None,
        dispatch_shaping_scale: float = DISPATCH_SHAPING_SCALE,
        no_op_shaping_penalty: float = NO_OP_SHAPING_PENALTY,
    ) -> None:
        self._validate_probability("alpha", alpha)
        self._validate_probability("gamma", gamma)
        self._validate_probability("epsilon", epsilon)
        self._validate_probability("epsilon_min", epsilon_min)
        self._validate_probability("epsilon_decay", epsilon_decay)
        if not math.isfinite(float(dispatch_shaping_scale)) or dispatch_shaping_scale < 0:
            raise ValueError("dispatch_shaping_scale must be finite and non-negative")
        if not math.isfinite(float(no_op_shaping_penalty)) or no_op_shaping_penalty < 0:
            raise ValueError("no_op_shaping_penalty must be finite and non-negative")
        if not isinstance(num_episodes, int) or num_episodes < 0:
            raise ValueError("num_episodes must be a non-negative integer")

        self.env = env
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.initial_epsilon = float(epsilon)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.num_episodes = num_episodes
        self.dispatch_shaping_scale = float(dispatch_shaping_scale)
        self.no_op_shaping_penalty = float(no_op_shaping_penalty)
        self.training_seed = seed
        self.rng = np.random.default_rng(seed)
        self.seed_rng = np.random.default_rng(None if seed is None else seed ^ 0x5A17)
        self.q_table: Dict[StateKey, Dict[int, float]] = {}
        self.training_history: List[float] = []
        self.episode_records: List[Dict[str, Any]] = []
        self.episode_seeds: List[int] = []
        self.episodes_completed = 0
        self.environment_config: Optional[Dict[str, Any]] = None

    @staticmethod
    def _validate_probability(name: str, value: float) -> None:
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("{} must be between 0 and 1".format(name))

    @staticmethod
    def _valid_actions(action_mask: Any) -> Tuple[int, ...]:
        mask = np.asarray(action_mask)
        if mask.ndim != 1 or mask.dtype.kind != "b":
            raise ValueError("action_mask must be a one-dimensional boolean array")
        actions = tuple(int(action) for action in np.flatnonzero(mask))
        if not actions:
            raise ValueError(
                "SARSA meta-action mask contains no valid action"
            )
        return actions

    def choose_action(
        self,
        state: StateKey,
        action_mask: Any,
        epsilon: Optional[float] = None,
        fallback_action: Optional[int] = None,
    ) -> int:
        """Choose a valid action using epsilon-greedy selection."""
        effective_epsilon = self.epsilon if epsilon is None else float(epsilon)
        self._validate_probability("epsilon", effective_epsilon)
        valid_actions = self._valid_actions(action_mask)

        if effective_epsilon > 0.0 and self.rng.random() < effective_epsilon:
            return int(self.rng.choice(valid_actions))

        values = self.q_table.get(state, {})
        if fallback_action is not None:
            if fallback_action not in valid_actions:
                raise ValueError("fallback_action must be valid under action_mask")
            if self.needs_fallback(state, action_mask):
                return int(fallback_action)

        initial_value = DISPATCH_Q_INITIAL if effective_epsilon > 0.0 else 0.0
        action_values = {
            action: values.get(
                action,
                NO_OP_Q_INITIAL if action == NO_OP_META_ACTION and effective_epsilon > 0.0
                else initial_value,
            )
            for action in valid_actions
        }
        best_value = max(action_values.values())
        best_actions = [
            action for action, value in action_values.items() if value == best_value
        ]
        return int(min(best_actions))

    def needs_fallback(self, state: StateKey, action_mask: Any) -> bool:
        """Check whether greedy evaluation has no learned preference to use."""
        valid_actions = self._valid_actions(action_mask)
        values = self.q_table.get(state, {})
        if not values:
            return True
        observed_values = [values.get(action, 0.0) for action in valid_actions]
        return all(value == observed_values[0] for value in observed_values)

    def _shaped_reward(
        self,
        observation: Mapping[str, Any],
        action: int,
        reward: float,
        info: Mapping[str, Any],
        dispatch_forecasts: Sequence[Tuple[float, float]] = (),
        strategic_penalty: float = 0.0,
    ) -> Tuple[float, float]:
        shaping = float(strategic_penalty)
        ambulances = observation.get("ambulances", {})
        call_slots = observation.get("call_slots", ())
        if not isinstance(ambulances, Mapping) or not isinstance(call_slots, Sequence):
            raise ValueError("Observation is missing SARSA reward-shaping inputs")
        idle_available = any(
            (details.get("status").value if isinstance(details.get("status"), AmbulanceStatus)
             else details.get("status")) == AmbulanceStatus.IDLE.value
            and details.get("current_call_id") is None
            for details in ambulances.values()
            if isinstance(details, Mapping)
        )
        pending_available = any(call is not None for call in call_slots)
        if action == NO_OP_META_ACTION and idle_available and pending_available:
            shaping -= self.no_op_shaping_penalty

        breakdown = info.get("reward_breakdown", {})
        if not isinstance(breakdown, Mapping):
            raise ValueError("Environment info reward_breakdown must be a mapping")
        wait_penalty = float(breakdown.get("wait_penalty", 0.0))
        failure_penalty = float(breakdown.get("failure_penalty", 0.0))
        completion_bonus = float(breakdown.get("completion_bonus", 0.0))
        wait_minutes = float(info.get("wait_minutes", 0.0))
        failed_call_count = int(info.get("failed_call_count", 0))
        completed_calls = info.get("completed_calls", ())
        if not isinstance(completed_calls, Sequence):
            raise ValueError("Environment info completed_calls must be a sequence")
        shaping += wait_penalty - WAIT_PENALTY_PER_MINUTE * wait_minutes
        shaping += failure_penalty - FAILURE_PENALTY * failed_call_count
        shaping -= completion_bonus
        for completed_call in completed_calls:
            if not isinstance(completed_call, Mapping):
                raise ValueError("Environment completed_calls entries must be mappings")
            shaping += (
                PERFECT_COMPLETION_BONUS
                if bool(completed_call.get("is_perfect"))
                else ON_TIME_COMPLETION_BONUS
            )

        for eta, deadline_remaining in dispatch_forecasts:
            if deadline_remaining > 0.0 and eta < deadline_remaining:
                shaping += self.dispatch_shaping_scale * (
                    1.0 - eta / deadline_remaining
                )
            else:
                shaping -= 15.0
        rerouted_dispatches = info.get("rerouted_dispatches", ())
        if not isinstance(rerouted_dispatches, Sequence):
            raise ValueError("Environment info rerouted_dispatches must be a sequence")
        for dispatch in rerouted_dispatches:
            if not isinstance(dispatch, Mapping):
                raise ValueError("Environment rerouted dispatch entries must be mappings")
            eta = float(dispatch["eta"])
            deadline_remaining = float(dispatch["deadline_remaining"])
            if deadline_remaining > 0.0 and eta < deadline_remaining:
                shaping += self.dispatch_shaping_scale * (
                    1.0 - eta / deadline_remaining
                )
            else:
                shaping -= 15.0
        return float(reward) + shaping, shaping

    def _load_balance_perfect_miss_penalty(
        self,
        env: Any,
        observation: Mapping[str, Any],
        action: int,
        dispatch_forecasts: Sequence[Tuple[float, float]],
    ) -> float:
        if action != 2 or not dispatch_forecasts:
            return 0.0
        call = _pending_call(observation)
        if call is None:
            return 0.0
        now = float(observation.get("current_minute", 0.0))
        failure_time = float(call.get("failure_time", 0.0))
        deadline_remaining = float(call.get("deadline", now + failure_time)) - now
        if int(call.get("severity", 1)) < 2 and deadline_remaining > 0.5 * failure_time:
            return 0.0
        perfect_threshold = float(
            call.get(
                "perfect_threshold",
                float(call.get("created_time", now)) + 0.5 * failure_time,
            )
        )
        perfect_time_remaining = perfect_threshold - now
        _, nearest_pairs = resolve_meta_action(env, observation, 0)
        nearest_forecast = self._dispatch_forecasts(env, nearest_pairs)
        if (
            nearest_forecast
            and nearest_forecast[0][0] <= perfect_time_remaining
            and dispatch_forecasts[0][0] > perfect_time_remaining
        ):
            return -LOAD_BALANCE_PERFECT_MISS_PENALTY
        return 0.0

    def learn(
        self,
        state: StateKey,
        action: int,
        reward: float,
        next_state: StateKey,
        next_action: Optional[int],
        terminal: bool,
    ) -> float:
        """Apply one on-policy SARSA update and return the updated Q value."""
        reward_value = float(reward)
        if not math.isfinite(reward_value):
            raise ValueError("reward must be finite")

        current_values = self.q_table.setdefault(state, {})
        current_value = current_values.get(
            action,
            NO_OP_Q_INITIAL if action == NO_OP_META_ACTION else DISPATCH_Q_INITIAL,
        )
        if terminal:
            target = reward_value
        else:
            if next_action is None:
                raise ValueError("next_action is required for a non-terminal update")
            next_value = self.q_table.get(next_state, {}).get(
                next_action,
                NO_OP_Q_INITIAL if next_action == NO_OP_META_ACTION else DISPATCH_Q_INITIAL,
            )
            target = reward_value + self.gamma * next_value

        updated_value = current_value + self.alpha * (target - current_value)
        current_values[action] = updated_value
        return updated_value

    @staticmethod
    def _observation_reward(observation: Mapping[str, Any]) -> float:
        try:
            reward = float(observation["last_step_reward"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "reset observation must contain numeric last_step_reward"
            ) from exc
        if not math.isfinite(reward):
            raise ValueError("last_step_reward must be finite")
        return reward

    @staticmethod
    def _truncation_backlog(info: Mapping[str, Any]) -> Dict[str, int]:
        if "backlog" not in info or not isinstance(info["backlog"], Mapping):
            raise ValueError("truncation info must contain a backlog mapping")
        backlog = info["backlog"]
        keys = ("pending_calls", "waiting_calls", "at_scene_calls")
        try:
            return {key: len(backlog[key]) for key in keys}
        except KeyError as exc:
            raise ValueError(
                "truncation backlog is missing {}".format(exc.args[0])
            ) from exc

    @staticmethod
    def _dispatch_forecasts(
        env: Any,
        pairs: Sequence[Tuple[int, int]],
    ) -> List[Tuple[float, float]]:
        city_map = getattr(env, "city_map", None)
        ambulances = getattr(env, "ambulances", None)
        calls = getattr(env, "calls", None)
        if not pairs or city_map is None or not isinstance(ambulances, Mapping) or not isinstance(calls, Mapping):
            return []
        current_minute = float(getattr(env, "current_minute"))
        forecasts = []
        for ambulance_id, call_id in pairs:
            ambulance = ambulances[int(ambulance_id)]
            call = calls[int(call_id)]
            route = city_map.route_from_road_to_road(
                start_edge=ambulance.current_edge,
                start_dist=ambulance.distance_from_start,
                dest_edge=call.edge,
                dest_dist=call.distance_from_start,
                by_travel_time=True,
            )
            forecasts.append(
                (
                    float(route["total_time_minutes"]),
                    float(call.deadline) - current_minute,
                )
            )
        return forecasts

    def train(
        self,
        env: Any = None,
        num_episodes: Optional[int] = None,
        seed: Optional[int] = None,
        safety_limit: Optional[int] = None,
        progress_interval: int = 500,
    ) -> List[float]:
        """Train until Environment termination, or an explicitly configured safety limit."""
        active_env = self.env if env is None else env
        if active_env is None:
            raise ValueError("an Environment instance is required for training")

        episode_count = self.num_episodes if num_episodes is None else num_episodes
        if not isinstance(episode_count, int) or episode_count < 0:
            raise ValueError("num_episodes must be a non-negative integer")
        if safety_limit is not None and (
            not isinstance(safety_limit, int) or safety_limit <= 0
        ):
            raise ValueError("safety_limit must be a positive integer when provided")
        if not isinstance(progress_interval, int) or progress_interval <= 0:
            raise ValueError("progress_interval must be a positive integer")

        if seed is not None:
            self.training_seed = seed
            self.rng = np.random.default_rng(seed)
            self.seed_rng = np.random.default_rng(seed ^ 0x5A17)
        self.num_episodes = episode_count
        base_seed = self.training_seed
        epsilon_decay_horizon = max(1, int(math.ceil(0.85 * episode_count)))
        scheduled_decay = (
            (self.epsilon_min / self.initial_epsilon)
            ** (1.0 / epsilon_decay_horizon)
            if self.initial_epsilon > 0.0
            else 0.0
        )
        self.epsilon_decay = scheduled_decay
        run_history: List[float] = []
        run_records: List[Dict[str, Any]] = []
        used_episode_seeds = set(self.episode_seeds)

        for _ in range(episode_count):
            episode_number = self.episodes_completed + 1
            episode_seed = None
            if base_seed is not None:
                while episode_seed is None or episode_seed in used_episode_seeds:
                    episode_seed = int(self.seed_rng.integers(0, 2**31 - 1))
                used_episode_seeds.add(episode_seed)
            if episode_seed is None:
                observation = active_env.reset()
            else:
                observation = active_env.reset(seed=episode_seed)
                self.episode_seeds.append(episode_seed)

            if not isinstance(observation, Mapping):
                raise TypeError("Environment.reset() must return an observation mapping")
            reset_reward = self._observation_reward(observation)
            environment_calls = getattr(active_env, "calls", {})
            reset_failed_calls = [
                call
                for call in (
                    environment_calls.values()
                    if isinstance(environment_calls, Mapping)
                    else ()
                )
                if getattr(getattr(call, "status", None), "value", None)
                == CallStatus.FAILED.value
            ]
            shaped_reset_reward = reset_reward + sum(
                REWARD_CONFIG[call.severity]["fail_penalty"]
                for call in reset_failed_calls
            ) - FAILURE_PENALTY * len(reset_failed_calls)
            state = encode_sarsa_state(observation)
            action_mask = sarsa_action_mask(observation, active_env)
            action = self.choose_action(state, action_mask)
            step_reward_total = 0.0
            raw_step_reward_total = 0.0
            no_op_actions = 0
            step_count = 0
            terminated = False
            truncated = False
            safety_limited = False
            backlog_counts: Optional[Dict[str, int]] = None

            while True:
                environment_action, dispatch_pairs = resolve_meta_action(
                    active_env,
                    observation,
                    action,
                )
                dispatch_forecasts = self._dispatch_forecasts(
                    active_env,
                    dispatch_pairs,
                )
                strategic_penalty = self._load_balance_perfect_miss_penalty(
                    active_env,
                    observation,
                    action,
                    dispatch_forecasts,
                )
                next_observation, reward, done, info = active_env.step(environment_action)
                if not isinstance(next_observation, Mapping):
                    raise TypeError("Environment.step() must return an observation mapping")
                if not isinstance(info, Mapping):
                    raise TypeError("Environment.step() info must be a mapping")
                if "truncated" not in info:
                    raise ValueError('Environment.step() info must contain "truncated"')

                raw_reward_value = float(reward)
                if not math.isfinite(raw_reward_value):
                    raise ValueError("Environment.step() reward must be finite")
                done_value = bool(done)
                truncated = bool(info["truncated"])
                if truncated and not done_value:
                    raise ValueError("Environment truncation must also set done=True")
                terminated = done_value and not truncated
                if truncated:
                    backlog_counts = self._truncation_backlog(info)

                reward_value, _ = self._shaped_reward(
                    observation,
                    action,
                    raw_reward_value,
                    info,
                    dispatch_forecasts,
                    strategic_penalty,
                )
                next_state = encode_sarsa_state(next_observation)
                next_action: Optional[int] = None
                if not terminated:
                    # A time-limit truncation is a cutoff, not a failed terminal
                    # state, so bootstrap using the policy's next valid action.
                    next_action = self.choose_action(
                        next_state, sarsa_action_mask(next_observation, active_env)
                    )

                self.learn(
                    state=state,
                    action=action,
                    reward=reward_value,
                    next_state=next_state,
                    next_action=next_action,
                    terminal=terminated,
                )
                step_reward_total += reward_value
                raw_step_reward_total += raw_reward_value
                no_op_actions += int(action == NO_OP_META_ACTION)
                step_count += 1
                observation = next_observation

                if done_value:
                    break
                if safety_limit is not None and step_count >= safety_limit:
                    safety_limited = True
                    break

                if next_action is None:
                    raise RuntimeError("non-terminal transition has no selected next action")
                state = next_state
                action = next_action

            total_return = shaped_reset_reward + step_reward_total
            raw_total_return = reset_reward + raw_step_reward_total
            self.epsilon = max(
                self.epsilon_min,
                self.initial_epsilon * (scheduled_decay ** episode_number),
            )
            record: Dict[str, Any] = {
                "episode": episode_number,
                "seed": episode_seed,
                "steps": step_count,
                "reset_reward": shaped_reset_reward,
                "environment_reset_reward": reset_reward,
                "step_reward_total": step_reward_total,
                "total_return": total_return,
                "environment_return": raw_total_return,
                "terminated": terminated,
                "truncated": truncated,
                "safety_limited": safety_limited,
                "epsilon": self.epsilon,
                "no_op_actions": no_op_actions,
                "no_op_rate": no_op_actions / max(1, step_count),
            }
            final_ambulances = observation.get("ambulances", {})
            served_calls = int(observation.get("served_calls", 0))
            failed_calls = int(observation.get("failed_calls", 0))
            active_calls = observation.get("active_calls", ())
            record["generated_calls"] = (
                served_calls + failed_calls + len(active_calls)
                if isinstance(active_calls, Sequence)
                else served_calls + failed_calls
            )
            record["served_calls"] = served_calls
            record["total_distance"] = sum(
                float(details.get("total_distance_traveled", 0.0))
                for details in final_ambulances.values()
                if isinstance(details, Mapping)
            ) if isinstance(final_ambulances, Mapping) else 0.0
            if truncated:
                record["backlog_counts"] = backlog_counts

            self.episode_records.append(record)
            run_records.append(record)
            self.training_history.append(total_return)
            run_history.append(total_return)
            self.episodes_completed += 1
            if self.episodes_completed % progress_interval == 0:
                batch = run_records[-progress_interval:]
                batch_steps = sum(item["steps"] for item in batch)
                print(
                    "SARSA progress | episodes={} | avg_reward_shaped={:.2f} | "
                    "avg_reward_environment={:.2f} | served={:.2f}/{:.2f} calls | "
                    "distance={:.2f} km | no_op={:.1f}%".format(
                        self.episodes_completed,
                        sum(item["total_return"] for item in batch) / len(batch),
                        sum(item["environment_return"] for item in batch) / len(batch),
                        sum(item["served_calls"] for item in batch) / len(batch),
                        sum(item["generated_calls"] for item in batch) / len(batch),
                        sum(item["total_distance"] for item in batch) / len(batch),
                        100.0 * sum(item["no_op_actions"] for item in batch)
                        / max(1, batch_steps),
                    )
                )

        return run_history

    @staticmethod
    def _state_to_json(value: Any) -> Any:
        if isinstance(value, tuple):
            return [SarsaAgent._state_to_json(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError("Q-table state contains a non-JSON-compatible value")

    @staticmethod
    def _state_from_json(value: Any) -> Any:
        if isinstance(value, list):
            return tuple(SarsaAgent._state_from_json(item) for item in value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise ValueError("checkpoint contains an invalid state key")

    def save_checkpoint(
        self,
        path: Union[str, Path],
        environment_config: Mapping[str, Any],
    ) -> None:
        """Write the policy and its training metadata as a JSON checkpoint."""
        if not isinstance(environment_config, Mapping):
            raise TypeError("environment_config must be a mapping")
        rows = []
        for state, action_values in sorted(
            self.q_table.items(),
            key=lambda item: json.dumps(
                self._state_to_json(item[0]), sort_keys=True, separators=(",", ":")
            ),
        ):
            rows.append(
                {
                    "state": self._state_to_json(state),
                    "actions": [
                        [int(action), float(value)]
                        for action, value in sorted(action_values.items())
                    ],
                }
            )

        self.environment_config = dict(environment_config)
        checkpoint = {
            "algorithm": "SARSA",
            "state_encoder_version": STATE_ENCODER_VERSION,
            "action_space_version": META_ACTION_VERSION,
            "hyperparameters": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon_start": self.initial_epsilon,
                "epsilon_current": self.epsilon,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min": self.epsilon_min,
                "num_episodes": self.num_episodes,
                "dispatch_shaping_scale": self.dispatch_shaping_scale,
                "no_op_shaping_penalty": self.no_op_shaping_penalty,
            },
            "training": {
                "seed": self.training_seed,
                "episodes_completed": self.episodes_completed,
                "episode_seed_scheme": "random_seed_stream",
                "episode_seeds": list(self.episode_seeds),
                "environment_config": self.environment_config,
            },
            "q_table": rows,
        }
        with Path(path).open("w", encoding="utf-8") as checkpoint_file:
            json.dump(checkpoint, checkpoint_file, indent=2, allow_nan=False)

    @classmethod
    def load_checkpoint(
        cls, path: Union[str, Path], env: Any = None
    ) -> "SarsaAgent":
        """Load a compatible JSON SARSA checkpoint."""
        with Path(path).open("r", encoding="utf-8") as checkpoint_file:
            checkpoint = json.load(checkpoint_file)
        if checkpoint.get("algorithm") != "SARSA":
            raise ValueError("checkpoint algorithm is not SARSA")
        checkpoint_encoder_ver = checkpoint.get("state_encoder_version")
        if checkpoint_encoder_ver not in (STATE_ENCODER_VERSION, "sarsa-state-meta-v1"):
            raise ValueError("checkpoint state encoder version is incompatible")
        if checkpoint.get("action_space_version") not in (META_ACTION_VERSION, "dispatch-meta-actions-v1"):
            raise ValueError("checkpoint action space version is incompatible")

        hyperparameters = checkpoint["hyperparameters"]
        training = checkpoint["training"]
        agent = cls(
            env=env,
            alpha=hyperparameters["alpha"],
            gamma=hyperparameters["gamma"],
            epsilon=hyperparameters["epsilon_start"],
            epsilon_decay=hyperparameters["epsilon_decay"],
            epsilon_min=hyperparameters["epsilon_min"],
            num_episodes=hyperparameters["num_episodes"],
            seed=training["seed"],
            dispatch_shaping_scale=hyperparameters.get(
                "dispatch_shaping_scale", DISPATCH_SHAPING_SCALE
            ),
            no_op_shaping_penalty=hyperparameters.get(
                "no_op_shaping_penalty", NO_OP_SHAPING_PENALTY
            ),
        )
        agent.state_encoder_version = checkpoint_encoder_ver
        agent.epsilon = float(hyperparameters["epsilon_current"])
        agent.episodes_completed = int(training["episodes_completed"])
        agent.episode_seeds = [
            int(episode_seed) for episode_seed in training.get("episode_seeds", ())
        ]
        agent.environment_config = training["environment_config"]

        for row in checkpoint["q_table"]:
            state = cls._state_from_json(row["state"])
            if not isinstance(state, tuple) or state in agent.q_table:
                raise ValueError("checkpoint contains an invalid or duplicate state key")
            action_values: Dict[int, float] = {}
            for action, value in row["actions"]:
                action_id = int(action)
                q_value = float(value)
                if action_id < 0 or action_id >= META_ACTION_COUNT or not math.isfinite(q_value):
                    raise ValueError("checkpoint contains an invalid Q-table entry")
                if action_id in action_values:
                    raise ValueError("checkpoint contains a duplicate action")
                action_values[action_id] = q_value
            agent.q_table[state] = action_values
        return agent


SARSAAgent = SarsaAgent

__all__ = [
    "SarsaAgent",
    "SARSAAgent",
    "sarsa_action_mask",
    "nearest_available_action_id",
    "resolve_meta_action",
    "META_ACTION_VERSION",
]
