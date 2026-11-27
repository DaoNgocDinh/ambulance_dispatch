"""Reproducible, non-learning evaluation for saved tabular SARSA policies."""

from __future__ import annotations

import heapq
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dispatch_env import AmbulanceDispatchEnv
from dispatch_env.sarsa import (
    META_ACTION_VERSION,
    SarsaAgent,
    resolve_meta_action,
    sarsa_action_mask,
)
from dispatch_env.sarsa_state import STATE_ENCODER_VERSION, encode_sarsa_state
from dispatch_env.types import (
    AmbulanceStatus,
    CallStatus,
    DEFAULT_MAX_OVERTIME,
    DEFAULT_MAX_PENDING_CALLS,
    DEFAULT_MAX_TIME,
    DEFAULT_NUM_AMBULANCES,
    DEFAULT_POISSON_LAMBDA,
)


# Fixed common seeds for future RL and heuristic comparisons.
EVALUATION_SEEDS = tuple(range(50001, 50011))
ENVIRONMENT_CONFIG = {
    "num_ambulances": DEFAULT_NUM_AMBULANCES,
    "max_time": DEFAULT_MAX_TIME,
    "max_overtime": DEFAULT_MAX_OVERTIME,
    "poisson_lambda": DEFAULT_POISSON_LAMBDA,
    "max_pending_calls": DEFAULT_MAX_PENDING_CALLS,
}

_REWARD_COMPONENTS = (
    "wait_penalty",
    "failure_penalty",
    "distance_penalty",
    "completion_bonus",
    "total",
)
_UNRESOLVED_BACKLOG_KEYS = (
    "pending_calls",
    "waiting_calls",
    "at_scene_calls",
)
_AGGREGATE_FIELDS = (
    "episode_return",
    "steps",
    "fallback_steps",
    "fallback_rate",
    "total_generated_calls",
    "served_calls",
    "served_rate",
    "failed_calls",
    "failed_rate",
    "unresolved_calls",
    "high_severity_served",
    "high_severity_served_rate",
    "high_severity_failed",
    "high_severity_failed_rate",
    "avg_waiting",
    "response_time_all",
    "high_severity_response_time",
    "deadline_success_rate",
    "high_severity_deadline_success_rate",
    "avg_slack_at_dispatch",
    "priority_violations",
    "total_distance",
    "ambulance_utilization",
    "reset_reward",
)
_AGGREGATE_LABELS = {
    "episode_return": "Episode Reward",
    "steps": "Steps",
    "fallback_steps": "Fallback Steps",
    "fallback_rate": "Fallback Rate",
    "total_generated_calls": "Generated Calls",
    "served_calls": "Served Calls",
    "served_rate": "Served Rate",
    "failed_calls": "Failed Calls",
    "failed_rate": "Failed Rate",
    "unresolved_calls": "Unresolved Calls",
    "high_severity_served": "High-Severity Served",
    "high_severity_served_rate": "High-Severity Served Rate",
    "high_severity_failed": "High-Severity Failed",
    "high_severity_failed_rate": "High-Severity Failed Rate",
    "avg_waiting": "Average Waiting Time (min)",
    "response_time_all": "Response Time, All Severities (min)",
    "high_severity_response_time": "High-Severity Response Time (min)",
    "deadline_success_rate": "Deadline Success Rate, All Severities",
    "high_severity_deadline_success_rate": "High-Severity Deadline Success Rate",
    "avg_slack_at_dispatch": "Average Slack at Dispatch (min)",
    "priority_violations": "Priority Violations",
    "total_distance": "Total Distance (km)",
    "ambulance_utilization": "Ambulance Utilization",
    "reset_reward": "Reset Reward (t=0)",
    "reward_wait_penalty": "Wait Penalty",
    "reward_failure_penalty": "Failure Penalty",
    "reward_distance_penalty": "Distance Penalty",
    "reward_completion_bonus": "Completion Bonus",
    "reward_total": "Step Reward Total",
}


EnvironmentFactory = Callable[[Mapping[str, Any], int], Any]


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be numeric".format(label)) from exc
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(label))
    return result


def aggregate_values(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """Return population mean/std/min/max, excluding explicitly unavailable metrics."""
    available = []
    for value in values:
        if value is None:
            continue
        number = _finite_float(value, "metric")
        available.append(number)
    if not available:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(available),
        "mean": statistics.fmean(available),
        "std": statistics.pstdev(available),
        "min": min(available),
        "max": max(available),
    }


def aggregate_episode_metrics(
    episodes: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Aggregate episode-level KPI values across evaluation seeds."""
    fields = list(_AGGREGATE_FIELDS)
    fields.extend("reward_{}".format(name) for name in _REWARD_COMPONENTS)
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for field in fields:
        values = []
        for episode in episodes:
            if field.startswith("reward_"):
                component = field[len("reward_") :]
                breakdown = episode.get("reward_breakdown")
                value = breakdown.get(component) if isinstance(breakdown, Mapping) else None
            else:
                value = episode.get(field)
            values.append(value)
        summary[field] = aggregate_values(values)
    return summary


def _read_checkpoint(
    checkpoint_path: Union[str, Path],
    seeds: Sequence[int],
) -> Tuple[SarsaAgent, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    path = Path(checkpoint_path)
    with path.open("r", encoding="utf-8") as checkpoint_file:
        checkpoint = json.load(checkpoint_file)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a JSON object")
    if checkpoint.get("algorithm") != "SARSA":
        raise ValueError("checkpoint algorithm must be SARSA")
    if checkpoint.get("state_encoder_version") != STATE_ENCODER_VERSION:
        raise ValueError(
            "checkpoint state encoder version is incompatible: expected {}".format(
                STATE_ENCODER_VERSION
            )
        )
    if checkpoint.get("action_space_version") != META_ACTION_VERSION:
        raise ValueError("checkpoint action space version is incompatible")

    hyperparameters = checkpoint.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise ValueError("checkpoint is missing hyperparameters")
    required_hyperparameters = {
        "alpha",
        "gamma",
        "epsilon_start",
        "epsilon_current",
        "epsilon_decay",
        "epsilon_min",
        "num_episodes",
    }
    missing_hyperparameters = required_hyperparameters.difference(hyperparameters)
    if missing_hyperparameters:
        raise ValueError(
            "checkpoint is missing hyperparameters: {}".format(
                ", ".join(sorted(missing_hyperparameters))
            )
        )
    for name in (
        "alpha",
        "gamma",
        "epsilon_start",
        "epsilon_current",
        "epsilon_decay",
        "epsilon_min",
    ):
        value = _finite_float(hyperparameters[name], "checkpoint {}".format(name))
        if not 0.0 <= value <= 1.0:
            raise ValueError("checkpoint {} must be between 0 and 1".format(name))
    configured_episodes = hyperparameters["num_episodes"]
    if (
        isinstance(configured_episodes, bool)
        or not isinstance(configured_episodes, int)
        or configured_episodes < 0
    ):
        raise ValueError("checkpoint num_episodes must be a non-negative integer")

    training = checkpoint.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("checkpoint is missing training metadata")
    if training.get("episode_seed_scheme") != "random_seed_stream":
        raise ValueError("checkpoint training seed scheme is missing or unsupported")
    training_seed = training.get("seed")
    completed_episodes = training.get("episodes_completed")
    if (
        isinstance(training_seed, bool)
        or not isinstance(training_seed, int)
        or training_seed < 0
    ):
        raise ValueError("checkpoint must contain a non-negative integer training seed")
    if (
        isinstance(completed_episodes, bool)
        or not isinstance(completed_episodes, int)
        or completed_episodes <= 0
    ):
        raise ValueError("checkpoint must contain completed training episodes")
    episode_seeds = training.get("episode_seeds")
    if (
        not isinstance(episode_seeds, list)
        or len(episode_seeds) != completed_episodes
        or any(
            isinstance(episode_seed, bool)
            or not isinstance(episode_seed, int)
            or episode_seed < 0
            for episode_seed in episode_seeds
        )
        or len(set(episode_seeds)) != len(episode_seeds)
    ):
        raise ValueError("checkpoint must record one non-negative seed per episode")

    checkpoint_environment = training.get("environment_config")
    if not isinstance(checkpoint_environment, Mapping):
        raise ValueError("checkpoint is missing environment_config")
    if set(checkpoint_environment) != set(ENVIRONMENT_CONFIG):
        raise ValueError(
            "checkpoint environment_config fields do not match the supported "
            "Environment schema"
        )
    if dict(checkpoint_environment) != ENVIRONMENT_CONFIG:
        raise ValueError(
            "checkpoint Environment config is incompatible with evaluation defaults: "
            "expected {}, got {}".format(
                ENVIRONMENT_CONFIG, dict(checkpoint_environment)
            )
        )

    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("evaluation requires exactly 10 unique common seeds")
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("evaluation seeds must be non-negative integers")
    trained_seeds = set(episode_seeds)
    overlap = trained_seeds.intersection(seeds)
    if overlap:
        raise ValueError(
            "evaluation seeds overlap training seeds: {}".format(sorted(overlap))
        )

    agent = SarsaAgent.load_checkpoint(path)
    if agent.environment_config != dict(checkpoint_environment):
        raise ValueError("loaded checkpoint Environment config did not round-trip")
    if not agent.q_table or not any(agent.q_table.values()):
        raise ValueError("checkpoint has an empty Q-table; refusing untrained evaluation")

    training_config = {
        "seed": training_seed,
        "episodes_completed": completed_episodes,
        "unique_q_table_states": len(agent.q_table),
        "alpha": agent.alpha,
        "gamma": agent.gamma,
        "epsilon_start": agent.initial_epsilon,
        "epsilon_current": agent.epsilon,
        "epsilon_decay": agent.epsilon_decay,
        "epsilon_min": agent.epsilon_min,
        "configured_episodes": agent.num_episodes,
        "dispatch_shaping_scale": agent.dispatch_shaping_scale,
        "no_op_shaping_penalty": agent.no_op_shaping_penalty,
    }
    return agent, dict(checkpoint_environment), training_config, checkpoint


def _observation_edge_map(observation: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    edges = observation.get("edges_by_id")
    if not isinstance(edges, Mapping):
        raise ValueError("Observation must contain edges_by_id for dispatch ETA")
    result: Dict[int, Mapping[str, Any]] = {}
    for edge_id, details in edges.items():
        if not isinstance(details, Mapping):
            raise ValueError("Observation edges_by_id values must be mappings")
        normalized_id = int(details.get("edge_id", edge_id))
        result[normalized_id] = details
    return result


def _shortest_travel_time(
    adjacency: Mapping[int, Sequence[Tuple[int, float]]],
    start_node: int,
    target_node: int,
) -> float:
    if start_node == target_node:
        return 0.0
    distances = {start_node: 0.0}
    frontier = [(0.0, start_node)]
    while frontier:
        elapsed, node = heapq.heappop(frontier)
        if elapsed != distances.get(node):
            continue
        if node == target_node:
            return elapsed
        for next_node, edge_time in adjacency.get(node, ()):
            candidate = elapsed + edge_time
            if candidate < distances.get(next_node, float("inf")):
                distances[next_node] = candidate
                heapq.heappush(frontier, (candidate, next_node))
    raise ValueError(
        "Observation road graph has no route from node {} to {}".format(
            start_node, target_node
        )
    )


def estimate_dispatch_eta_minutes(
    observation: Mapping[str, Any],
    ambulance: Mapping[str, Any],
    call: Mapping[str, Any],
) -> float:
    """Estimate route ETA from the pre-dispatch Observation, without Environment internals."""
    edges = _observation_edge_map(observation)
    adjacency: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for edge in edges.values():
        start_node = int(edge["from_node"])
        end_node = int(edge["to_node"])
        length = _finite_float(edge["length"], "edge length")
        speed = _finite_float(edge["speed"], "edge speed")
        if length <= 0.0 or speed <= 0.0:
            raise ValueError("Observation road edge length and speed must be positive")
        adjacency[start_node].append((end_node, length / speed * 60.0))

    start_edge_id = int(ambulance["current_edge_id"])
    target_edge_id = int(call["edge_id"])
    start_edge = edges.get(start_edge_id)
    target_edge = edges.get(target_edge_id)
    if start_edge is None or target_edge is None:
        raise ValueError("dispatch ETA references an unknown Observation edge")

    start_distance = _finite_float(
        ambulance["distance_from_start"], "ambulance distance_from_start"
    )
    target_distance = _finite_float(
        call["distance_from_start"], "call distance_from_start"
    )
    start_speed = _finite_float(start_edge["speed"], "ambulance edge speed")
    target_speed = _finite_float(target_edge["speed"], "call edge speed")

    if start_edge_id == target_edge_id and start_distance <= target_distance:
        return round((target_distance - start_distance) / start_speed * 60.0, 3)

    remaining_start = max(
        0.0,
        _finite_float(start_edge["length"], "start edge length") - start_distance,
    )
    start_node = int(start_edge["to_node"])
    destination_node = int(target_edge["from_node"])
    middle_eta = _shortest_travel_time(adjacency, start_node, destination_node)
    eta = (
        remaining_start / start_speed * 60.0
        + middle_eta
        + target_distance / target_speed * 60.0
    )
    return round(eta, 3)


def _dispatch_slacks(
    env: Any,
    observation: Mapping[str, Any],
    action: int,
) -> List[float]:
    dispatch_pairs = env.decode_action(action)
    if not isinstance(dispatch_pairs, Sequence):
        raise TypeError("Environment.decode_action() must return a sequence")
    ambulances = observation.get("ambulances")
    call_slots = observation.get("call_slots")
    if not isinstance(ambulances, Mapping) or not isinstance(call_slots, Sequence):
        raise ValueError("Observation is missing dispatch inputs")

    calls_by_id = {}
    for pending_call in call_slots:
        if pending_call is None:
            continue
        if not isinstance(pending_call, Mapping):
            raise ValueError("Observation call_slots must contain mappings or None")
        call_id = pending_call.get("call_id", pending_call.get("id"))
        if call_id is not None:
            calls_by_id[int(call_id)] = pending_call

    current_minute = _finite_float(
        observation["current_minute"], "current_minute"
    )
    slacks = []
    for ambulance_id, call_id in dispatch_pairs:
        ambulance = ambulances.get(ambulance_id)
        call = calls_by_id.get(int(call_id))
        if not isinstance(ambulance, Mapping) or call is None:
            raise ValueError("decoded dispatch does not match the current Observation")
        eta = estimate_dispatch_eta_minutes(observation, ambulance, call)
        deadline = _finite_float(call["deadline"], "call deadline")
        slacks.append(deadline - (current_minute + eta))
    return slacks


def _attribute(call: Any, name: str) -> Any:
    if isinstance(call, Mapping):
        return call[name]
    return getattr(call, name)


def _status_value(call: Any) -> Any:
    status = _attribute(call, "status")
    return getattr(status, "value", status)


def _unresolved_calls(info: Mapping[str, Any]) -> List[Any]:
    backlog = info.get("backlog")
    if not isinstance(backlog, Mapping):
        raise ValueError("truncation info must contain a backlog mapping")
    unresolved = []
    for key in _UNRESOLVED_BACKLOG_KEYS:
        if key not in backlog or not isinstance(backlog[key], Sequence):
            raise ValueError("truncation backlog must contain {}".format(key))
        unresolved.extend(backlog[key])
    return unresolved


def _episode_call_metrics(
    env: Any,
    final_info: Mapping[str, Any],
    truncated: bool,
) -> Dict[str, Any]:
    calls_mapping = getattr(env, "calls", None)
    if not isinstance(calls_mapping, Mapping):
        raise ValueError("Environment evaluator requires read-only env.calls mapping")
    calls = list(calls_mapping.values())
    unresolved = _unresolved_calls(final_info) if truncated else []
    unresolved_ids = [int(_attribute(call, "call_id")) for call in unresolved]
    if len(unresolved_ids) != len(set(unresolved_ids)):
        raise ValueError("truncation backlog contains duplicate unresolved calls")
    known_call_ids = {int(_attribute(call, "call_id")) for call in calls}
    if not set(unresolved_ids).issubset(known_call_ids):
        raise ValueError("truncation backlog references a call absent from env.calls")

    served = [call for call in calls if _status_value(call) == CallStatus.COMPLETED.value]
    failed = [call for call in calls if _status_value(call) == CallStatus.FAILED.value]
    unresolved_id_set = set(unresolved_ids)
    classified_ids = {
        int(_attribute(call, "call_id")) for call in served + failed
    }.union(unresolved_id_set)
    if classified_ids != known_call_ids:
        raise ValueError(
            "generated-call reconciliation failed: every call must be COMPLETED, "
            "FAILED, or unresolved at truncation"
        )

    generated_count = len(calls)
    served_count = len(served)
    failed_count = len(failed)
    unresolved_count = len(unresolved_id_set)
    if generated_count != served_count + failed_count + unresolved_count:
        raise ValueError(
            "generated-call reconciliation failed: generated != served + failed + unresolved"
        )

    waiting_times = [
        _finite_float(_attribute(call, "wait_time"), "call wait_time")
        for call in calls
    ]
    response_times = []
    deadline_successes = 0
    severity_counts: Dict[str, Dict[str, int]] = {
        str(severity): {
            "generated": 0,
            "served": 0,
            "failed": 0,
            "unresolved": 0,
        }
        for severity in (1, 2, 3)
    }
    for call in calls:
        severity = int(_attribute(call, "severity"))
        if str(severity) not in severity_counts:
            raise ValueError("Environment call has unsupported severity")
        severity_counts[str(severity)]["generated"] += 1
        call_id = int(_attribute(call, "call_id"))
        status = _status_value(call)
        if status == CallStatus.COMPLETED.value:
            severity_counts[str(severity)]["served"] += 1
        elif status == CallStatus.FAILED.value:
            severity_counts[str(severity)]["failed"] += 1
        elif call_id in unresolved_id_set:
            severity_counts[str(severity)]["unresolved"] += 1
        else:
            raise ValueError("call status is not represented in KPI reconciliation")

        arrival = _attribute(call, "arrived_at_scene_time")
        if arrival is not None:
            response_time = _finite_float(
                arrival, "call arrived_at_scene_time"
            ) - _finite_float(_attribute(call, "created_time"), "call created_time")
            response_times.append(response_time)
            deadline = _finite_float(_attribute(call, "deadline"), "call deadline")
            if _finite_float(arrival, "call arrived_at_scene_time") <= deadline + 1e-6:
                deadline_successes += 1

    return {
        "total_generated_calls": generated_count,
        "served_calls": served_count,
        "served_rate": served_count / generated_count if generated_count else None,
        "failed_calls": failed_count,
        "failed_rate": failed_count / generated_count if generated_count else None,
        "unresolved_calls": unresolved_count,
        "high_severity_served": None,
        "high_severity_served_rate": None,
        "high_severity_failed": None,
        "high_severity_failed_rate": None,
        "avg_waiting": statistics.fmean(waiting_times) if waiting_times else None,
        "response_time_all": (
            statistics.fmean(response_times) if response_times else None
        ),
        "high_severity_response_time": None,
        "deadline_success_rate": (
            deadline_successes / generated_count if generated_count else None
        ),
        "high_severity_deadline_success_rate": None,
        "priority_violations": None,
        "severity_breakdown": severity_counts,
    }


def _accumulate_reward(
    totals: Dict[str, float],
    info: Mapping[str, Any],
    step_reward: float,
) -> None:
    breakdown = info.get("reward_breakdown")
    if not isinstance(breakdown, Mapping):
        raise ValueError("Environment step info must contain reward_breakdown")
    values = {
        name: _finite_float(breakdown[name], "reward_breakdown.{}".format(name))
        for name in _REWARD_COMPONENTS
    }
    recomputed_total = (
        -values["wait_penalty"]
        - values["failure_penalty"]
        - values["distance_penalty"]
        + values["completion_bonus"]
    )
    if not math.isclose(values["total"], recomputed_total, rel_tol=1e-9, abs_tol=1e-7):
        raise ValueError("Environment reward_breakdown components do not reconcile")
    if not math.isclose(values["total"], step_reward, rel_tol=1e-9, abs_tol=1e-7):
        raise ValueError("step reward does not match info reward_breakdown.total")
    for name in _REWARD_COMPONENTS:
        totals[name] += values[name]


def _make_environment(config: Mapping[str, Any], seed: int) -> AmbulanceDispatchEnv:
    return AmbulanceDispatchEnv(**dict(config), seed=seed)


def evaluate_checkpoint(
    checkpoint_path: Union[str, Path],
    seeds: Sequence[int] = EVALUATION_SEEDS,
    environment_factory: Optional[EnvironmentFactory] = None,
) -> Dict[str, Any]:
    """Load a trained checkpoint and evaluate greedy policy on exactly 10 seeds."""
    agent, environment_config, training_config, checkpoint = _read_checkpoint(
        checkpoint_path, seeds
    )
    factory = _make_environment if environment_factory is None else environment_factory
    q_table_before = {
        state: dict(action_values) for state, action_values in agent.q_table.items()
    }
    q_states_before = len(q_table_before)
    q_entries_before = sum(len(values) for values in q_table_before.values())
    epsilon_before = agent.epsilon
    episodes: List[Dict[str, Any]] = []
    maximum_steps = (
        int(environment_config["max_time"])
        + int(environment_config["max_overtime"])
    )

    for seed in seeds:
        env = factory(environment_config, seed)
        observation = env.reset(seed=seed)
        if not isinstance(observation, Mapping):
            raise TypeError("Environment.reset() must return an observation mapping")
        reset_reward = _finite_float(
            observation["last_step_reward"], "reset observation last_step_reward"
        )
        ambulance_count = len(observation["ambulances"])
        if ambulance_count <= 0:
            raise ValueError("Environment Observation must contain ambulances")

        episode_return = reset_reward
        step_reward_total = 0.0
        reward_totals = {name: 0.0 for name in _REWARD_COMPONENTS}
        dispatch_slacks: List[float] = []
        busy_ambulance_samples = 0
        fallback_steps = 0
        steps = 0
        terminated = False
        truncated = False
        final_info: Optional[Mapping[str, Any]] = None
        final_observation = observation

        while True:
            state = encode_sarsa_state(observation)
            action_mask = sarsa_action_mask(observation, env)
            used_fallback = bool(
                action_mask[0] and agent.needs_fallback(state, action_mask)
            )
            fallback_steps += int(used_fallback)
            meta_action = agent.choose_action(
                state,
                action_mask,
                epsilon=0.0,
                fallback_action=0 if used_fallback else None,
            )
            action, _ = resolve_meta_action(env, observation, meta_action)
            dispatch_slacks.extend(_dispatch_slacks(env, observation, action))

            next_observation, reward, done, info = env.step(action)
            if not isinstance(next_observation, Mapping):
                raise TypeError("Environment.step() must return an observation mapping")
            if not isinstance(info, Mapping):
                raise TypeError("Environment.step() info must be a mapping")
            if "truncated" not in info:
                raise ValueError('Environment.step() info must contain "truncated"')
            step_reward = _finite_float(reward, "Environment.step() reward")
            done_value = bool(done)
            truncated = bool(info["truncated"])
            if truncated and not done_value:
                raise ValueError("Environment truncation must also set done=True")
            terminated = done_value and not truncated

            _accumulate_reward(reward_totals, info, step_reward)
            step_reward_total += step_reward
            episode_return += step_reward
            steps += 1
            ambulances = next_observation.get("ambulances")
            if not isinstance(ambulances, Mapping) or len(ambulances) != ambulance_count:
                raise ValueError("Environment Observation ambulance count changed")
            busy_ambulance_samples += sum(
                details.get("status") != AmbulanceStatus.IDLE.value
                for details in ambulances.values()
                if isinstance(details, Mapping)
            )
            if any(not isinstance(details, Mapping) for details in ambulances.values()):
                raise ValueError("Observation ambulance values must be mappings")
            final_info = info
            final_observation = next_observation

            if done_value:
                break
            if steps >= maximum_steps:
                raise RuntimeError(
                    "Environment did not terminate or truncate within its configured horizon"
                )
            observation = next_observation

        if final_info is None:
            raise RuntimeError("evaluation episode completed without an Environment step")
        call_metrics = _episode_call_metrics(env, final_info, truncated)
        final_ambulances = final_observation["ambulances"]
        total_distance = sum(
            _finite_float(details["total_distance_traveled"], "total_distance_traveled")
            for details in final_ambulances.values()
        )
        episode = {
            "seed": seed,
            "episode_return": episode_return,
            "reset_reward": reset_reward,
            "steps": steps,
            "fallback_steps": fallback_steps,
            "fallback_rate": fallback_steps / max(1, steps),
            "terminated": terminated,
            "truncated": truncated,
            "reward_breakdown": reward_totals,
            "step_reward_total": step_reward_total,
            "total_distance": total_distance,
            "ambulance_utilization": (
                busy_ambulance_samples / (ambulance_count * steps)
                if steps
                else None
            ),
            "avg_slack_at_dispatch": (
                statistics.fmean(dispatch_slacks) if dispatch_slacks else None
            ),
            **call_metrics,
        }
        episodes.append(episode)

    if agent.q_table != q_table_before:
        raise RuntimeError("Q-table changed during evaluation")
    if agent.epsilon != epsilon_before:
        raise RuntimeError("agent epsilon changed during evaluation")

    q_entries_after = sum(len(values) for values in agent.q_table.values())
    limitations = [
        "High-Severity is undefined in RL_ALGORITHM_STANDARD.md (V-03); "
        "high-severity KPI fields are null and severity 1/2/3 counts are retained.",
        "Priority Violation has no approved definition in the standard (V-04); "
        "priority_violations is null rather than applying an unapproved heuristic.",
        "Response Time uses arrived_at_scene_time - created_time for all severities; "
        "the required high-severity subset cannot be selected until its definition is agreed.",
        "Ambulance Utilization is the post-step observation estimate: non-IDLE "
        "ambulance samples divided by ambulance_count * steps.",
        "Dispatch ETA/Slack is calculated from the pre-dispatch Observation road "
        "graph and its current speed/length values; ETA is rounded to 0.001 minute.",
        "t=0 reset reward is included in episode_return; reward_breakdown sums "
        "step() info only because reset() supplies no reward breakdown.",
    ]
    return {
        "algorithm": "SARSA",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "encoder_version": checkpoint["state_encoder_version"],
        "training_config": training_config,
        "environment_config": environment_config,
        "evaluation_seeds": list(seeds),
        "aggregation": {"std": "population standard deviation (ddof=0)"},
        "metric_definitions": {
            "generated_calls": "len(env.calls); must reconcile with served + failed + unresolved",
            "served_calls": "CallStatus.COMPLETED (treatment completed at scene)",
            "failed_calls": "CallStatus.FAILED, including queue_overflow",
            "unresolved_calls": (
                "Only pending_calls + waiting_calls + at_scene_calls from "
                "info['backlog'] when truncated"
            ),
            "served_rate": "served_calls / generated_calls",
            "failed_rate": "failed_calls / generated_calls",
            "avg_waiting": "Mean call.wait_time over all generated calls",
            "response_time_all": (
                "Mean arrived_at_scene_time - created_time for calls that arrived"
            ),
            "deadline_success_rate": (
                "Calls with arrived_at_scene_time <= deadline / generated_calls"
            ),
            "avg_slack_at_dispatch": (
                "Mean deadline - (dispatch current_minute + observation-derived ETA) "
                "over dispatched calls"
            ),
            "total_distance": (
                "Sum of final Observation ambulance total_distance_traveled"
            ),
            "ambulance_utilization": (
                "Post-step non-IDLE ambulance samples / (ambulance_count * steps)"
            ),
            "reward_breakdown": (
                "Sum of info['reward_breakdown'] over step() calls; reset reward is "
                "separate and included only in episode_return"
            ),
        },
        "episodes": episodes,
        "summary": aggregate_episode_metrics(episodes),
        "verification": {
            "q_table_unchanged": True,
            "q_table_states_before": q_states_before,
            "q_table_states_after": len(agent.q_table),
            "q_table_entries_before": q_entries_before,
            "q_table_entries_after": q_entries_after,
            "learning_updates": 0,
            "evaluation_epsilon": 0.0,
            "agent_epsilon_before": epsilon_before,
            "agent_epsilon_after": agent.epsilon,
        },
        "limitations": limitations,
    }


def save_evaluation_report(
    report: Mapping[str, Any],
    output_path: Union[str, Path],
) -> None:
    with Path(output_path).open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, allow_nan=False)


def format_evaluation_report(report: Mapping[str, Any]) -> str:
    lines = [
        "SARSA Evaluation",
        "----------------",
        "Checkpoint: {}".format(report["checkpoint"]),
        "Encoder version: {}".format(report["encoder_version"]),
        "Environment config: {}".format(
            json.dumps(report["environment_config"], sort_keys=True)
        ),
        "Training config: {}".format(json.dumps(report["training_config"], sort_keys=True)),
        "Seeds: {}".format(report["evaluation_seeds"]),
        "Std definition: {}".format(report["aggregation"]["std"]),
        "",
        "Metric                                      Mean       Std       Min       Max",
        "--------------------------------------------------------------------------------",
    ]
    summary = report["summary"]
    for field in _AGGREGATE_FIELDS:
        stats = summary[field]
        label = _AGGREGATE_LABELS[field]
        lines.append(
            "{:<44} {:>8} {:>9} {:>9} {:>9}".format(
                label,
                _format_metric(stats["mean"]),
                _format_metric(stats["std"]),
                _format_metric(stats["min"]),
                _format_metric(stats["max"]),
            )
        )
    for component in _REWARD_COMPONENTS:
        field = "reward_{}".format(component)
        stats = summary[field]
        lines.append(
            "{:<44} {:>8} {:>9} {:>9} {:>9}".format(
                _AGGREGATE_LABELS[field],
                _format_metric(stats["mean"]),
                _format_metric(stats["std"]),
                _format_metric(stats["min"]),
                _format_metric(stats["max"]),
            )
        )
    lines.extend(
        [
            "",
            "Per seed:",
            "Seed   Return     Steps Fallback% Served  Failed  Unresolved  Distance  Utilization",
        ]
    )
    for episode in report["episodes"]:
        lines.append(
            "{seed:<6} {episode_return:>9.2f} {steps:>8} {fallback_rate:>9.1%} "
            "{served_calls:>7} {failed_calls:>7} {unresolved_calls:>11} "
            "{total_distance:>9.2f} {ambulance_utilization:>12.4f}".format(
                **episode
            )
        )
    verification = report["verification"]
    lines.extend(
        [
            "",
            "Q-table states: {} -> {}; state-action entries: {} -> {}".format(
                verification["q_table_states_before"],
                verification["q_table_states_after"],
                verification["q_table_entries_before"],
                verification["q_table_entries_after"],
            ),
            "Q-table unchanged: {}; learning updates: {}".format(
                verification["q_table_unchanged"],
                verification["learning_updates"],
            ),
            "Limitations:",
        ]
    )
    lines.extend("- {}".format(limitation) for limitation in report["limitations"])
    return "\n".join(lines)


def _format_metric(value: Optional[float]) -> str:
    return "N/A" if value is None else "{:.4f}".format(value)


__all__ = [
    "EVALUATION_SEEDS",
    "ENVIRONMENT_CONFIG",
    "aggregate_episode_metrics",
    "aggregate_values",
    "estimate_dispatch_eta_minutes",
    "evaluate_checkpoint",
    "format_evaluation_report",
    "save_evaluation_report",
]
