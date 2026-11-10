"""Huấn luyện Q-learning và so sánh với heuristic auto-dispatch của môi trường."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dispatch_env import (
    AmbulanceDispatchEnv,
    DEFAULT_MAX_OVERTIME,
    DEFAULT_MAX_TIME,
    DEFAULT_NUM_AMBULANCES,
    DEFAULT_POISSON_LAMBDA,
    DEFAULT_SEED,
)
from q_learning import QLearningAgent
from ambulance_q_learning_adapter import AmbulanceDispatchAdapter


def run_episode(
    env: AmbulanceDispatchEnv,
    adapter: AmbulanceDispatchAdapter,
    agent: Optional[QLearningAgent],
    training: bool,
    seed: int,
    capture_log: bool = False,
) -> Dict[str, object]:
    """Chạy một lượt mô phỏng từ đầu đến khi môi trường báo kết thúc.

    agent=None nghĩa là chạy baseline auto_dispatch; nếu có agent thì chọn action
    bằng Q-learning. Khi training=False, agent chỉ khai thác Q-table, không cập nhật.
    """
    env.reset(seed=seed)
    total_reward = 0.0  # Tổng reward cộng dồn qua toàn bộ các phút của episode.
    log_lines: List[str] = []
    done = False

    while not done:
        if agent is None:
            # Baseline dùng luật auto_dispatch tích hợp sẵn trong môi trường.
            _, reward, done, info = env.step(auto_dispatch=True)
            action_text = "Heuristic auto_dispatch"
        else:
            # Bước 1: adapter biến môi trường hiện tại thành khóa state và action hợp lệ.
            state = adapter.encode_state(env)
            valid_actions = adapter.valid_actions(env)
            # Bước 2: chọn action; lúc học có khám phá, lúc đánh giá chỉ khai thác.
            action_id = agent.choose_action(state, valid_actions, explore=training)
            decoded_action = env.decode_action(action_id)
            action_text = f"action_id={action_id}, phân công={decoded_action}"

            # Bước 3: môi trường thực thi action và trả reward/state kế tiếp.
            _, reward, done, info = adapter.step(env, action_id)
            next_state = adapter.encode_state(env)
            next_actions = adapter.valid_actions(env)
            if training:
                # Bước 4: dùng trải nghiệm (state, action, reward, next_state) cập nhật Q.
                agent.update(
                    state=state,
                    action=action_id,
                    reward=reward,
                    next_state=next_state,
                    next_valid_actions=next_actions,
                    done=done,
                )

        total_reward += reward
        if capture_log:
            # Chỉ lưu log chi tiết cho episode mẫu để tránh tạo log khổng lồ.
            log_lines.append(f"Quyết định: {action_text}\n{env.format_log_step()}")

    # Tóm tắt cuối episode; max(1, call_count) tránh chia cho 0 khi không có cuộc gọi.
    call_count = len(env.calls)
    return {
        "seed": seed,
        "total_reward": total_reward,
        "calls": call_count,
        "served": env.served_calls_count,
        "failed": env.failed_calls_count,
        "perfect": env.perfect_calls_count,
        "success_rate": env.served_calls_count / max(1, call_count),
        "failure_rate": env.failed_calls_count / max(1, call_count),
        "perfect_rate": env.perfect_calls_count / max(1, call_count),
        "steps": env.current_minute,
        "truncated": env.is_truncated,
        "log": "\n\n".join(log_lines),
    }


def print_episode_progress(
    phase: str,
    episode: int,
    total_episodes: int,
    started_at: float,
    result: Dict[str, object],
    details: str = "",
) -> None:
    """In một dòng tiến độ có cùng bố cục cho huấn luyện và đánh giá."""
    elapsed = time.perf_counter() - started_at
    average_seconds = elapsed / episode
    remaining_seconds = average_seconds * (total_episodes - episode)
    calls = int(result["calls"])
    served = int(result["served"])
    failed = int(result["failed"])
    success_rate = 100 * float(result["success_rate"])
    progress = 100 * episode / total_episodes

    message = (
        f"[{phase:<9}] {episode:>3}/{total_episodes:<3} ({progress:5.1f}%) | "
        f"Reward {float(result['total_reward']):>9.2f} | "
        f"Cuộc gọi {served:>3}/{calls:<3} hoàn thành ({success_rate:5.1f}%), "
        f"thất bại {failed:>3} | "
        f"Đã chạy {elapsed / 60:6.1f} phút, còn ~{remaining_seconds / 60:6.1f} phút"
    )
    if details:
        message += f" | {details}"

    # flush=True giúp dòng hiện ngay trong terminal sau khi episode kết thúc.
    print(message, flush=True)


def print_final_summary(
    strategy: str,
    rows: List[Dict[str, object]],
) -> None:
    """In trung bình reward và tỷ lệ hoàn thành của một chiến lược."""
    average_success = 100 * sum(float(row["success_rate"]) for row in rows) / len(rows)
    average_reward = sum(float(row["total_reward"]) for row in rows) / len(rows)
    print(
        f"  {strategy:<14} | Hoàn thành trung bình: {average_success:5.1f}% "
        f"| Reward trung bình: {average_reward:9.2f}"
    )


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    """Ghi metrics thành CSV; bỏ log chi tiết dài khỏi bảng chỉ số."""
    if not rows:
        raise ValueError("Không có kết quả để ghi CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [key for key in rows[0] if key != "log"]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)


def plot_results(
    training_rows: List[Dict[str, object]],
    evaluation_rows: List[Dict[str, object]],
    path: Path,
) -> None:
    """Vẽ reward từng episode và tỷ lệ hoàn thành khi đánh giá."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    rewards = [float(row["total_reward"]) for row in training_rows]
    axes[0].plot(range(1, len(rewards) + 1), rewards, alpha=0.45, label="Reward mỗi episode")
    window = min(10, len(rewards))
    averages = [
        sum(rewards[max(0, i - window + 1):i + 1])
        / len(rewards[max(0, i - window + 1):i + 1])
        for i in range(len(rewards))
    ]
    axes[0].plot(range(1, len(averages) + 1), averages, label="Trung bình trượt")
    axes[0].set(title="Reward trong huấn luyện", xlabel="Episode", ylabel="Tổng reward")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    strategies = ("Q-learning", "Auto-dispatch")
    means = []
    for strategy in strategies:
        rates = [
            100 * float(row["success_rate"])
            for row in evaluation_rows
            if row["strategy"] == strategy
        ]
        means.append(sum(rates) / max(1, len(rates)))
    axes[1].bar([0, 1], means, color=["#2878b5", "#888888"])
    axes[1].set_xticks([0, 1], strategies)
    axes[1].set(ylim=(0, 100), title="Tỷ lệ cuộc gọi hoàn thành", ylabel="Phần trăm")
    axes[1].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    """Đọc tùy chọn, huấn luyện, đánh giá và ghi kết quả ra thư mục."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50, help="Số episode huấn luyện.")
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--max-time", type=int, default=DEFAULT_MAX_TIME)
    parser.add_argument("--max-overtime", type=int, default=DEFAULT_MAX_OVERTIME)
    parser.add_argument("--poisson", type=float, default=DEFAULT_POISSON_LAMBDA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=Path("results/q_learning"))
    args = parser.parse_args()

    if args.episodes < 1 or args.evaluation_episodes < 1:
        parser.error("episodes và evaluation-episodes phải lớn hơn 0.")
    if args.max_time < 1 or args.max_overtime < 0:
        parser.error("max-time phải lớn hơn 0 và max-overtime không được âm.")
    if not math.isfinite(args.poisson) or args.poisson < 0:
        parser.error("poisson phải là số hữu hạn và không âm.")

    # Dùng cùng cấu hình cho mọi episode; chỉ thay seed để tạo tình huống khác nhau.
    def make_env(seed: int) -> AmbulanceDispatchEnv:
        return AmbulanceDispatchEnv(
            num_ambulances=DEFAULT_NUM_AMBULANCES,
            max_time=args.max_time,
            max_overtime=args.max_overtime,
            poisson_lambda=args.poisson,
            seed=seed,
        )

    agent = QLearningAgent(seed=args.seed)
    training_rows: List[Dict[str, object]] = []
    training_started_at = time.perf_counter()
    for episode in range(args.episodes):
        seed = args.seed + episode
        result = run_episode(
            make_env(seed), AmbulanceDispatchAdapter(), agent, training=True, seed=seed
        )
        result["episode"] = episode + 1
        result["epsilon"] = agent.epsilon
        result["q_table_size"] = len(agent.q_table)
        training_rows.append(result)
        # Báo cáo sau mỗi episode để người dùng biết chương trình vẫn đang chạy.
        print_episode_progress(
            phase="Huấn luyện",
            episode=episode + 1,
            total_episodes=args.episodes,
            started_at=training_started_at,
            result=result,
            details=(
                f"epsilon={agent.epsilon:.4f} | "
                f"Q-table={len(agent.q_table)} cặp"
            ),
        )
        agent.decay_exploration()

    # Lưu dữ liệu huấn luyện và Q-table trước khi bước sang đánh giá.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "training_metrics.csv", training_rows)
    agent.save(args.output_dir / "q_table.json")

    # Đánh giá trên seed mới, không khám phá và không cập nhật Q-table.
    # Mỗi seed được dùng chung cho Q-learning và baseline để so sánh công bằng.
    evaluation_rows: List[Dict[str, object]] = []
    sample_log = ""
    evaluation_started_at = time.perf_counter()
    for index in range(args.evaluation_episodes):
        seed = args.seed + 100_000 + index
        q_result = run_episode(
            make_env(seed),
            AmbulanceDispatchAdapter(),
            agent,
            training=False,
            seed=seed,
            capture_log=index == 0,
        )
        if index == 0:
            sample_log = str(q_result["log"])
        q_result.update({"episode": index + 1, "strategy": "Q-learning"})
        evaluation_rows.append(q_result)

        baseline = run_episode(
            make_env(seed),
            AmbulanceDispatchAdapter(),
            agent=None,
            training=False,
            seed=seed,
        )
        baseline.update({"episode": index + 1, "strategy": "Auto-dispatch"})
        evaluation_rows.append(baseline)
        print_episode_progress(
            phase="Đánh giá",
            episode=index + 1,
            total_episodes=args.evaluation_episodes,
            started_at=evaluation_started_at,
            result=q_result,
            details=(
                f"Auto-dispatch reward={float(baseline['total_reward']):.2f}, "
                f"hoàn thành={100 * float(baseline['success_rate']):.1f}%"
            ),
        )

    write_csv(args.output_dir / "evaluation_metrics.csv", evaluation_rows)
    (args.output_dir / "q_learning_episode.log").write_text(sample_log + "\n", encoding="utf-8")
    plot_results(training_rows, evaluation_rows, args.output_dir / "q_learning_results.png")

    print("\n" + "=" * 78)
    print("TỔNG KẾT ĐÁNH GIÁ")
    print("-" * 78)
    for strategy in ("Q-learning", "Auto-dispatch"):
        rows = [row for row in evaluation_rows if row["strategy"] == strategy]
        print_final_summary(strategy, rows)
    print("-" * 78)
    print(f"Q-table: {len(agent.q_table):,} cặp trạng thái-hành động")
    print(f"Thư mục kết quả: {args.output_dir.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    main()
