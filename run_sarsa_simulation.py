from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

from dispatch_env import AmbulanceDispatchEnv, SarsaAgent
from dispatch_env.sarsa import resolve_meta_action, sarsa_action_mask
from dispatch_env.sarsa_state import encode_sarsa_state


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_evaluation(
    checkpoint_path: Path,
    seed: int = 50001,
    log_filename: Path = Path("sarsa_simulation.log"),
) -> None:
    agent = SarsaAgent.load_checkpoint(checkpoint_path)
    if agent.environment_config is None:
        raise ValueError("SARSA checkpoint is missing environment configuration")
    if seed in set(agent.episode_seeds):
        raise ValueError("evaluation seed overlaps the checkpoint's training seeds")

    env = AmbulanceDispatchEnv(**agent.environment_config, seed=seed)
    observation = env.reset(seed=seed)

    print("=" * 85)
    print("      ĐÁNH GIÁ POLICY SARSA (GREEDY META-ACTION)")
    print("=" * 85)
    print(f"- Checkpoint SARSA                       : {checkpoint_path}")
    print(f"- Hạt giống đánh giá                     : {seed}")
    print(f"- Thời gian mô phỏng tối đa sinh cuộc gọi: {env.max_time} phút")
    print(f"- Số lượng xe cứu thương                 : {env.num_ambulances} xe")
    print(f"- File log                               : {log_filename}")
    print("=" * 85)

    log_entries: List[str] = [env.format_log_step()]
    start_time = time.time()
    step_count = 0
    fallback_steps = 0
    done = False

    while not done:
        state = encode_sarsa_state(observation)
        action_mask = sarsa_action_mask(observation, env)
        used_fallback = bool(action_mask[0] and agent.needs_fallback(state, action_mask))
        fallback_steps += int(used_fallback)
        meta_action = agent.choose_action(
            state,
            action_mask,
            epsilon=0.0,
            fallback_action=0 if used_fallback else None,
        )
        action, _ = resolve_meta_action(env, observation, meta_action)
        observation, _, done, _ = env.step(action)
        step_count += 1
        log_entries.append(env.format_log_step())

        if step_count % 30 == 0 or done:
            status_tag = " [TRUNCATED]" if env.is_truncated else ""
            print(
                f"[+] Phút {env.current_minute:>3}/{env.max_time} | "
                f"Đang chờ: {len(env.pending_calls)} | "
                f"Đang xử lý: {len(env.active_calls)} | "
                f"Thành công: {env.served_calls_count:>3} "
                f"(Hoàn hảo: {env.perfect_calls_count:>3}) | "
                f"Thất bại: {env.failed_calls_count:>3} | "
                f"Reward: {env.total_reward:>8.2f}{status_tag}"
            )

    elapsed_time = time.time() - start_time
    log_filename.write_text("\n\n".join(log_entries) + "\n", encoding="utf-8")

    total_calls = len(env.calls)
    print("\n" + "=" * 85)
    print("                        KẾT QUẢ ĐÁNH GIÁ SARSA")
    print("=" * 85)
    if env.is_truncated:
        print("\n" + env.format_backlog_report() + "\n")
    print(f"- Tổng thời gian mô phỏng hoàn tất : {env.current_minute} phút")
    termination = (
        f"Bị dừng (TRUNCATED - quá giờ tối đa {env.max_overtime} phút)"
        if env.is_truncated
        else "Hoàn thành tất cả cuộc gọi (COMPLETED)"
    )
    print(f"- Trạng thái kết thúc              : {termination}")
    print(f"- Thời gian chạy thực tế           : {elapsed_time:.2f} giây")
    print(
        f"- Bước dùng fallback nearest       : {fallback_steps}/{step_count} "
        f"({fallback_steps / max(1, step_count) * 100:.1f}%)"
    )
    print(f"- Unique states trong Q-table      : {len(agent.q_table)}")
    print(f"- Tổng số cuộc gọi đã phát sinh    : {total_calls}")
    print(
        f"- Tổng số cuộc gọi THÀNH CÔNG      : {env.served_calls_count} "
        f"({env.served_calls_count / max(1, total_calls) * 100:.1f}%)"
    )
    print(f"  + Trong đó HOÀN HẢO (trước 50%)  : {env.perfect_calls_count}")
    print(
        f"- Tổng số cuộc gọi THẤT BẠI        : {env.failed_calls_count} "
        f"({env.failed_calls_count / max(1, total_calls) * 100:.1f}%)"
    )
    print(f"- Tổng điểm thưởng tích lũy (R)    : {env.total_reward:.2f}")

    print(f"\n- Thống kê hoạt động của {env.num_ambulances} xe cứu thương:")
    for ambulance_id, ambulance in env.ambulances.items():
        coord = ambulance.get_coord(env.city_map)
        print(
            f"  * Xe {ambulance_id}: Quãng đường di chuyển = "
            f"{ambulance.total_distance_traveled:.2f} km | "
            f"Số lần reroute tính lại đường = {ambulance.reroute_count} | "
            f"Vị trí cuối = ({coord[0]:.2f}, {coord[1]:.2f}) km | "
            f"Trạng thái = {ambulance.status.value}"
        )

    print(f"\n[+] Đã lưu lịch sử chi tiết vào file: {log_filename}")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a saved SARSA policy in the ambulance dispatch environment"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trained SARSA JSON checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=50001,
        help="Evaluation seed (default: 50001)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("sarsa_simulation.log"),
        help="Detailed per-minute log path",
    )
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.seed, args.log)
