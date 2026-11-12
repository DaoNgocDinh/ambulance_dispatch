"""Script huan luyen DUNG CHUNG cho ca 3 thuat toan cua nhom:
on-policy MC, Q-learning, SARSA - tren cung env, cung adapter.

Cach chay (tai thu muc goc cua repo ambulance_dispatch):
    python train.py --algo mc           # huan luyen on-policy MC
    python train.py --algo qlearning    # huan luyen Q-learning
    python train.py --algo sarsa        # huan luyen SARSA

Moi nguoi chi can viet 1 file agent tuan thu "hop dong interface"
(xem phan HOP DONG ben duoi). Khong ai duoc sua group_adapter.py
hay code trong dispatch_env/ - env va adapter la phan DUNG CHUNG.

HOP DONG INTERFACE (bat buoc cho ca 3 agent):
  1. Constructor: Agent(number_of_actions, discount_factor=0.95,
       epsilon=1.0, minimum_epsilon=0.05, epsilon_decay=0.995, seed=42)
  2. select_action(state, valid_actions, training=True) -> int
     (state la tuple do adapter tao; chi chon trong valid_actions)
  3. decay_exploration() -> None   (giam epsilon sau moi episode)
  4. save(path) / load(path)       (luu/nạp model, khuyen nghi pickle)
  5. Ham cap nhat:
     - MC:       update_episode(episode)  (episode = list (s, a, r))
     - Q-learning/SARSA: update(state, action, reward,
                                next_state, next_valid_actions, done)
       (SARSA tu chon a' ben trong update bang select_action
        cua chinh no -> vong train giong het Q-learning)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from group_adapter import GroupEnvAdapter


def build_agent(algo: str, number_of_actions: int, args) -> object:
    """Khoi tao agent theo ten; import lazy de thieu file van chay."""
    common = dict(
        number_of_actions=number_of_actions,
        discount_factor=args.gamma,
        epsilon=1.0,
        minimum_epsilon=0.05,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )
    if algo == "mc":
        from mc_agent import MonteCarloAgent

        return MonteCarloAgent(**common)
    if algo == "qlearning":
        from qlearning_agent import QLearningAgent  # file cua ban Q-learning

        return QLearningAgent(learning_rate=args.lr, **common)
    if algo == "sarsa":
        from sarsa_agent import SarsaAgent  # file cua ban SARSA

        return SarsaAgent(learning_rate=args.lr, **common)
    raise ValueError(f"algo khong hop le: {algo} (chon mc/qlearning/sarsa)")


def train(adapter: GroupEnvAdapter, agent, args) -> object:
    """Vong huan luyen chung: tu phan biet MC (hoc cuoi episode)
    va TD (hoc ngay tung buoc) qua hasattr."""
    is_mc = hasattr(agent, "update_episode")
    mode = "MC (hoc cuoi episode)" if is_mc else "TD (hoc tung buoc)"
    print(f"Thuat toan: {args.algo} | Che do: {mode} | "
          f"Actions: {adapter.number_of_actions}")

    training_logs: list[dict] = []
    for episode in range(args.episodes):
        state = adapter.reset(seed=args.seed + episode)
        done = False
        episode_data: list[tuple] = []

        while not done:
            valid_actions = adapter.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=True)
            next_state, reward, done = adapter.step(action)
            if is_mc:
                # MC: chi ghi lai, chua hoc.
                episode_data.append((state, action, reward))
            else:
                # TD (Q-learning/SARSA): hoc ngay sau moi buoc.
                # next_valid_actions duoc lay SAU step -> ung voi next_state.
                next_valid = adapter.get_valid_actions()
                agent.update(state, action, reward,
                             next_state, next_valid, done)
            state = next_state

        if is_mc:
            # MC: het episode moi cap nhat Q tu loi nhuan thuc te.
            agent.update_episode(episode_data)
        agent.decay_exploration()

        ep_return = sum(r for _s, _a, r in episode_data) if is_mc else None
        summary = adapter.get_summary()
        row = {
            "episode": episode + 1,
            "algo": args.algo,
            "epsilon": round(agent.epsilon, 4),
            "states_learned": len(agent.q_table),
            **summary,
        }
        if ep_return is not None:
            row["return"] = round(ep_return, 2)
        training_logs.append(row)

        if (episode + 1) % args.log_every == 0:
            print(f"Ep {episode + 1}/{args.episodes} | "
                  f"served {summary['served']} failed {summary['failed']} | "
                  f"states {len(agent.q_table)} | eps {agent.epsilon:.3f}")

    if args.model:
        agent.save(args.model)
        print(f"Model saved to: {Path(args.model).resolve()}")
    if args.log:
        save_logs(training_logs, args.log)
        print(f"Logs saved to: {Path(args.log).resolve()}")
    return agent


def evaluate(adapter: GroupEnvAdapter, agent, episodes: int = 50,
             seed: int = 999) -> dict:
    """Danh gia chinh sach tham lam (khong tham do)."""
    returns, served, failed = [], 0, 0
    is_mc = hasattr(agent, "update_episode")
    for ep in range(episodes):
        state = adapter.reset(seed=seed + ep)
        done, total = False, 0.0
        ep_data = []
        while not done:
            valid = adapter.get_valid_actions()
            action = agent.select_action(state, valid, training=False)
            state, reward, done = adapter.step(action)
            total += reward
            ep_data.append((state, action, reward))
        returns.append(total)
        s = adapter.get_summary()
        served += s["served"]
        failed += s["failed"]
    return {
        "episodes": episodes,
        "mean_return": round(float(np.mean(returns)), 2),
        "served/ep": round(served / episodes, 2),
        "failed/ep": round(failed / episodes, 2),
    }


def save_logs(rows: list[dict], output_file: str) -> None:
    if not rows:
        raise ValueError("The training log is empty.")
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Huan luyen chung MC/Q-learning/SARSA")
    p.add_argument("--algo", choices=["mc", "qlearning", "sarsa"],
                   default="mc", help="thuat toan can huan luyen")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=0.1,
                   help="learning rate cho Q-learning/SARSA (MC khong dung)")
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--max-time", type=int, default=360,
                   help="so phut mo phong moi episode")
    p.add_argument("--poisson", type=float, default=0.5,
                   help="lambda sinh ca goi")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--log", type=str, default=None)
    p.add_argument("--eval-episodes", type=int, default=0,
                   help="so episode danh gia tham lam sau train (0 = bo qua)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    adapter = GroupEnvAdapter(
        max_time=args.max_time,
        poisson_lambda=args.poisson,
        max_pending_calls=3,
        num_ambulances=4,
        seed=args.seed,
    )
    agent = build_agent(args.algo, adapter.number_of_actions, args)
    agent = train(adapter, agent, args)
    if args.eval_episodes > 0:
        print("Danh gia:", evaluate(adapter, agent,
                                    episodes=args.eval_episodes))
