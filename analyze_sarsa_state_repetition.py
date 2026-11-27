from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, Union

from dispatch_env import AmbulanceDispatchEnv
from dispatch_env.sarsa_state import StateKey, encode_sarsa_state


def analyze_repetition(
    seeds: Iterable[int] = (42, 43, 44, 45, 46),
) -> Dict[str, Union[int, float]]:
    """Measure encoder collisions under a no-dispatch policy, without training."""
    seed_list = tuple(seeds)
    if not seed_list:
        raise ValueError("At least one seed is required")

    visits: Counter[StateKey] = Counter()
    episode_presence = {}
    states_per_episode = []

    for episode_index, seed in enumerate(seed_list):
        env = AmbulanceDispatchEnv(seed=seed)
        observation = env.reset(seed=seed)
        episode_states = 0

        while True:
            state = encode_sarsa_state(observation)
            visits[state] += 1
            episode_presence.setdefault(state, set()).add(episode_index)
            episode_states += 1

            action_mask = observation["action_mask"]
            if len(action_mask) == 0 or not bool(action_mask[0]):
                raise RuntimeError("The Environment no-op action is unexpectedly masked")

            observation, _, done, _ = env.step(0)
            if done:
                state = encode_sarsa_state(observation)
                visits[state] += 1
                episode_presence.setdefault(state, set()).add(episode_index)
                episode_states += 1
                break

        states_per_episode.append(episode_states)

    total_states = sum(states_per_episode)
    unique_states = len(visits)
    repeated_state_count = sum(count >= 2 for count in visits.values())
    cross_episode_repeated_count = sum(
        len(episodes) >= 2 for episodes in episode_presence.values()
    )
    duplicate_visit_rate = (
        (total_states - unique_states) / total_states if total_states else 0.0
    )

    return {
        "episodes": len(seed_list),
        "total_states": total_states,
        "unique_states": unique_states,
        "repeated_states": repeated_state_count,
        "cross_episode_repeated_states": cross_episode_repeated_count,
        "duplicate_visit_rate": duplicate_visit_rate,
        "average_states_per_episode": total_states / len(seed_list),
    }


if __name__ == "__main__":
    results = analyze_repetition()
    print(f"Episodes: {int(results['episodes'])}")
    print(f"Total states: {int(results['total_states'])}")
    print(f"Unique states: {int(results['unique_states'])}")
    print(f"States seen at least twice: {int(results['repeated_states'])}")
    print(
        "States seen in at least two episodes: "
        f"{int(results['cross_episode_repeated_states'])}"
    )
    print(f"Duplicate-visit rate: {results['duplicate_visit_rate']:.2%}")
    print(f"Average states per episode: {results['average_states_per_episode']:.1f}")
