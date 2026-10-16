"""
Script chạy thử nghiệm và kiểm chứng Môi trường Điều phối Xe cứu thương (SmartCityAmbulanceEnv).
Mô hình: 15 nút, 3 xe cứu thương, tối đa 3 cuộc gọi đồng thời và 5 tuyến đường huyết mạch.
"""

from __future__ import annotations

import sys

import numpy as np

from environment import EnvConfig, SmartCityAmbulanceEnv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def print_environment_summary(env: SmartCityAmbulanceEnv) -> None:
    """In thông tin tổng quan về đồ thị 15 nút, 5 tuyến huyết mạch và phân tích không gian trạng thái."""
    print("=" * 80)
    print("MÔI TRƯỜNG MÔ PHỎNG ĐIỀU PHỐI XE CỨU THƯƠNG TRONG THÀNH PHỐ THÔNG MINH (MDP)")
    print("=" * 80)
    print(f"- Số nút giao thông (|V|)           : {env.num_nodes} nút (đánh số 1..15)")
    print(f"- Số xe cứu thương (K)              : {env.num_ambulances} xe")
    print(f"- Số cuộc gọi chờ đồng thời tối đa  : {env.max_pending_requests} cuộc gọi (M = 3)")
    print(f"- Số tuyến đường huyết mạch         : {env.num_arterial_roads} tuyến")
    print(f"- Nút Trạm trực (Stations)          : {env.config.station_nodes}")
    print(f"- Nút Bệnh viện (Hospitals)         : {env.config.hospital_nodes}")
    print(f"- Kích thước vector trạng thái S_t  : {env.state_vector_size} chiều ([X_amb(6), X_req(9), X_traffic(5)])")
    print(f"- Kích thước không gian hành động A : {env.number_of_actions} hành động rời rạc")
    print(f"- Hệ số chiết khấu mặc định (gamma) : {env.gamma}")

    analysis = env.analyze_state_space(
        num_nodes=env.num_nodes,
        num_ambulances=env.num_ambulances,
        max_requests=env.max_pending_requests,
        num_arterials=env.num_arterial_roads,
    )
    print("\n[Phân tích tổ hợp không gian trạng thái lý thuyết (Mục 4.2.1)]")
    print(f"  + Số tổ hợp X_amb     : (15 x 5)^3            = {analysis['X_amb_states']:,}")
    print(f"  + Số tổ hợp X_req     : (1 + 15 x 3 x 15)^3   = {analysis['X_req_states']:,}")
    print(f"  + Số tổ hợp X_traffic : 3^5                   = {analysis['X_traffic_states']:,}")
    print(f"  => Tổng số trạng thái : {analysis['total_theoretical_states']:,} (~ {analysis['scientific_notation']})")

    print("\n[Danh sách 5 tuyến đường huyết mạch (Arterial Roads)]")
    for idx, edge in enumerate(env.graph.arterial_edges):
        print(
            f"  {idx + 1}. Nút {edge.u:2d} ({env.graph.nodes[edge.u].name}) "
            f"<-> Nút {edge.v:2d} ({env.graph.nodes[edge.v].name}) | Chiều dài: {edge.base_length}"
        )
    print("=" * 80)


def run_step_by_step_demo(num_demo_steps: int = 6) -> None:
    """Chạy minh họa từng bước thời gian để quan sát S_t, A_t, R_t và S_{t+1}."""
    env = SmartCityAmbulanceEnv(config=EnvConfig(max_steps=50), seed=42)
    print_environment_summary(env)

    print("\n>>> MINH HỌA QUÁ TRÌNH TƯƠNG TÁC TỪNG BƯỚC (STEP-BY-STEP TRACE):")
    env.reset(seed=42)
    env.render()

    for _ in range(num_demo_steps):
        action = env.recommend_baseline_action()
        decoded = env.decode_action(action)
        obs, reward, done, info = env.step(action)

        print(f"\n-> Thực thi hành động a = {action} [{decoded['label']}] | Phần thưởng R = {reward:+.1f}")
        for ev in info["events"]:
            print(f"   * Sự kiện: {ev}")
        env.render()
        if done:
            break


def evaluate_policy(policy_name: str, num_episodes: int = 30) -> dict:
    """Đánh giá trung bình một chính sách điều phối trên nhiều episode."""
    env = SmartCityAmbulanceEnv(config=EnvConfig(max_steps=80), seed=100)
    rewards = []
    resp_times = []
    golden_rates = []
    completed_counts = []
    expired_counts = []

    for ep in range(num_episodes):
        obs = env.reset(seed=1000 + ep)
        done = False
        while not done:
            if policy_name == "baseline":
                action = env.recommend_baseline_action()
            else:
                valid_acts = obs["valid_actions"]
                action = int(env.rng.choice(valid_acts))
            obs, _, done, _ = env.step(action)

        m = env.get_metrics()
        rewards.append(m["total_reward"])
        resp_times.append(m["avg_response_time"])
        golden_rates.append(m["golden_time_rate_pct"])
        completed_counts.append(m["completed_hospital_deliveries"])
        expired_counts.append(m["expired_requests"])

    return {
        "policy": policy_name,
        "avg_reward": float(np.mean(rewards)),
        "avg_response_time": float(np.mean(resp_times)),
        "avg_golden_rate_pct": float(np.mean(golden_rates)),
        "avg_completed": float(np.mean(completed_counts)),
        "avg_expired": float(np.mean(expired_counts)),
    }


if __name__ == "__main__":
    run_step_by_step_demo(num_demo_steps=5)

    print("\n" + "=" * 80)
    print("SO SÁNH THỬ NGHIỆM NHANH TRÊN 30 EPISODES (80 BƯỚC / EPISODE)")
    print("=" * 80)
    rand_stats = evaluate_policy("random", num_episodes=30)
    base_stats = evaluate_policy("baseline", num_episodes=30)

    print(
        f"{'Chính sách':<22} | {'Tổng Reward TB':>14} | {'TG Phản hồi TB':>14} | "
        f"{'Tỷ lệ TG Vàng':>14} | {'Chuyển viện TB':>14} | {'Ca Quá hạn TB':>13}"
    )
    print("-" * 102)
    for st in (rand_stats, base_stats):
        name = "Ngẫu nhiên (Random)" if st["policy"] == "random" else "Baseline (Gần nhất)"
        print(
            f"{name:<22} | {st['avg_reward']:>14.2f} | {st['avg_response_time']:>14.2f} | "
            f"{st['avg_golden_rate_pct']:>13.2f}% | {st['avg_completed']:>14.2f} | {st['avg_expired']:>13.2f}"
        )
    print("=" * 80)
