from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

from dispatch_env import AmbulanceDispatchEnv, SarsaAgent
from dispatch_env.types import (
    DEFAULT_MAX_OVERTIME,
    DEFAULT_MAX_PENDING_CALLS,
    DEFAULT_MAX_TIME,
    DEFAULT_NUM_AMBULANCES,
    DEFAULT_POISSON_LAMBDA,
)


ENVIRONMENT_CONFIG = {
    "num_ambulances": DEFAULT_NUM_AMBULANCES,
    "max_time": DEFAULT_MAX_TIME,
    "max_overtime": DEFAULT_MAX_OVERTIME,
    "poisson_lambda": DEFAULT_POISSON_LAMBDA,
    "max_pending_calls": DEFAULT_MAX_PENDING_CALLS,
}


def run_training(
    train_episodes: int = 3000,
    alpha: float = 0.15,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.9995,
    epsilon_min: float = 0.02,
    seed: int = 42,
    safety_limit: Optional[int] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> SarsaAgent:
    env = AmbulanceDispatchEnv(seed=seed)
    agent = SarsaAgent(
        env=env,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        num_episodes=train_episodes,
        seed=seed,
    )
    train_history = agent.train(
        num_episodes=train_episodes,
        seed=seed,
        safety_limit=safety_limit,
        progress_interval=500,
    )
    if checkpoint_path is not None:
        agent.save_checkpoint(checkpoint_path, ENVIRONMENT_CONFIG)

    mean_train = sum(train_history) / max(1, len(train_history))
    recent_records = agent.episode_records[-len(train_history):] if train_history else []
    natural_terminations = sum(record["terminated"] for record in recent_records)
    truncations = sum(record["truncated"] for record in recent_records)
    safety_limited = sum(
        record["safety_limited"] for record in recent_records
    )

    print("SARSA training summary")
    print(f"- training episodes: {train_episodes}")
    print(
        "- alpha={}, gamma={}, epsilon_start={}, scheduled_decay={}, epsilon_min={}".format(
            alpha, gamma, epsilon, agent.epsilon_decay, epsilon_min
        )
    )
    print(f"- seed: {seed}")
    print("- epsilon schedule: smooth exponential decay to epsilon_min at 85% of training")
    print(f"- mean shaped training return (includes reset reward): {mean_train:.2f}")
    mean_environment_return = (
        sum(record["environment_return"] for record in recent_records)
        / max(1, len(recent_records))
    )
    print(f"- mean Environment return (unshaped): {mean_environment_return:.2f}")
    print(f"- natural terminations: {natural_terminations}")
    print(f"- Environment truncations: {truncations}")
    print(f"- safety-limited episodes: {safety_limited}")
    print(f"- final epsilon: {agent.epsilon:.4f}")
    print(f"- unique Q-table states: {len(agent.q_table)}")
    if checkpoint_path is not None:
        print(f"- checkpoint: {checkpoint_path}")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tabular SARSA ambulance dispatch agent")
    parser.add_argument("--train-episodes", type=int, default=3000, help="Number of training episodes")
    parser.add_argument("--safety-limit", type=int, help="Optional maximum Environment steps per episode")
    parser.add_argument("--alpha", type=float, default=0.15, help="SARSA learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon for exploration")
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.9995,
        help="Legacy setting; training reaches epsilon_min at 85% of episodes",
    )
    parser.add_argument("--epsilon-min", type=float, default=0.02, help="Lower bound for epsilon")
    parser.add_argument("--seed", type=int, default=42, help="Training seed")
    parser.add_argument("--checkpoint", type=Path, help="Optional path for a JSON Q-table checkpoint")
    args = parser.parse_args()

    run_training(
        train_episodes=args.train_episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        seed=args.seed,
        safety_limit=args.safety_limit,
        checkpoint_path=args.checkpoint,
    )
