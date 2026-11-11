from __future__ import annotations

import argparse

from dispatch_env import AmbulanceDispatchEnv, SarsaAgent


def run_training(
    train_episodes: int = 100,
    eval_episodes: int = 20,
    max_steps: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.05,
    seed: int = 42,
) -> None:
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

    train_history = agent.train(num_episodes=train_episodes, max_steps=max_steps)
    eval_rewards = agent.evaluate(env=env, episodes=eval_episodes, greedy=True, max_steps=max_steps)

    mean_train = sum(train_history) / max(1, len(train_history))
    mean_eval = sum(eval_rewards) / max(1, len(eval_rewards))
    min_eval = min(eval_rewards) if eval_rewards else 0.0
    max_eval = max(eval_rewards) if eval_rewards else 0.0

    print("SARSA training summary")
    print(f"- train episodes: {train_episodes}")
    print(f"- eval episodes: {eval_episodes}")
    print(f"- alpha={alpha}, gamma={gamma}, epsilon={epsilon}, epsilon_decay={epsilon_decay}, epsilon_min={epsilon_min}")
    print(f"- mean train reward: {mean_train:.2f}")
    print(f"- mean eval reward: {mean_eval:.2f}")
    print(f"- eval range: [{min_eval:.2f}, {max_eval:.2f}]")
    print(f"- final epsilon: {agent.epsilon:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a SARSA agent for ambulance dispatch")
    parser.add_argument("--train-episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument("--alpha", type=float, default=0.1, help="SARSA learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon for exploration")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay per episode")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Lower bound for epsilon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_training(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        seed=args.seed,
    )
