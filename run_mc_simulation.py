"""Chay 1 episode HIEN THI bang chinh sach on-policy MC da huan luyen,
xuat ket qua theo DUNG DINH DANG cua baseline (run_simulation.py):
- Log tung phut bang env.format_log_step() -> file log giong
    baseline_seed42.log de dat canh nhau so sanh.
- Dong tien do moi 30 phut + khoi "KET QUA TONG KET MO PHONG"
    in ra man hinh giong het ban baseline.

Khac biet duy nhat so voi baseline: baseline goi env.step(auto_dispatch=True)
(heuristic co san cua env), con script nay moi phut hoi chinh sach MC:
state -> agent.select_action(tham lam) -> adapter.step(action).

Cau hinh mac dinh GIONG HET baseline: 6 xe, lambda=0.1, seed=42,
max_time=360, max_overtime=300, max_pending=3 (lay tu DEFAULT_* cua env).

Cach chay (tai thu muc goc cua repo):
# Chua co model: train MC theo dung cau hinh baseline roi hien thi
python run_mc_simulation.py
# Da co model train cung cau hinh (so xe khop): nap lai de hien thi
python run_mc_simulation.py --model mc_model.pkl
# Doi seed giong baseline seed42:
python run_mc_simulation.py --model mc_model.pkl --seed 42 --log mc_seed42.log

LUU Y: model train voi 4 xe (13 action) KHONG dung duoc cho cau hinh
6 xe (19 action) - script se bao loi ro rang va yeu cau train lai.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List

from map_env import CityMap
from dispatch_env import (
    AmbulanceDispatchEnv,
    DEFAULT_NUM_AMBULANCES,
    DEFAULT_MAX_TIME,
    DEFAULT_MAX_OVERTIME,
    DEFAULT_POISSON_LAMBDA,
    DEFAULT_MAX_PENDING_CALLS,
    DEFAULT_SEED,
)

from group_adapter import GroupEnvAdapter
from mc_agent import MonteCarloAgent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def build_adapter(
    max_time: int,
    max_overtime: int,
    poisson_lambda: float,
    max_pending_calls: int,
    num_ambulances: int,
    seed: int,
) -> tuple[GroupEnvAdapter, CityMap]:
    """Tao env + adapter theo DUNG cach run_simulation.py tao env."""
    city_map = CityMap(seed=seed)
    env = AmbulanceDispatchEnv(
        city_map=city_map,
        num_ambulances=num_ambulances,
        max_time=max_time,
        max_overtime=max_overtime,
        poisson_lambda=poisson_lambda,
        max_pending_calls=max_pending_calls,
        seed=seed,
    )
    return GroupEnvAdapter(env=env), city_map


def train_agent(adapter: GroupEnvAdapter, episodes: int, seed: int,
                save_path: str) -> MonteCarloAgent:
    """Train on-policy MC (hoc cuoi episode) theo cau hinh cua adapter."""
    agent = MonteCarloAgent(
        number_of_actions=adapter.number_of_actions,
        discount_factor=0.95,
        epsilon=1.0,
        minimum_epsilon=0.05,
        epsilon_decay=0.995,
        seed=seed,
    )
    print(f"[*] Train on-policy MC: {episodes} episode, "
        f"{adapter.number_of_actions} action, seed={seed}")
    for episode in range(episodes):
        state = adapter.reset(seed=seed + episode)
        done = False
        episode_data: list[tuple] = []
        while not done:
            valid = adapter.get_valid_actions()
            action = agent.select_action(state, valid, training=True)
            next_state, reward, done = adapter.step(action)
            episode_data.append((state, action, reward))
            state = next_state
        agent.update_episode(episode_data)
        agent.decay_exploration()
        if (episode + 1) % 50 == 0 or episode == 0:
            s = adapter.get_summary()
            print(f"    Ep {episode + 1}/{episodes} | served {s['served']} "
                f"failed {s['failed']} | states {len(agent.q_table)} | "
                f"eps {agent.epsilon:.3f}")
    agent.save(save_path)
    print(f"[*] Da luu model: {save_path} ({len(agent.q_table)} state)")
    return agent


def run_display(
    max_time: int = DEFAULT_MAX_TIME,
    max_overtime: int = DEFAULT_MAX_OVERTIME,
    poisson_lambda: float = DEFAULT_POISSON_LAMBDA,
    max_pending_calls: int = DEFAULT_MAX_PENDING_CALLS,
    num_ambulances: int = DEFAULT_NUM_AMBULANCES,
    seed: int = DEFAULT_SEED,
    log_filename: str = "mc_simulation.log",
    model_path: str | None = None,
    train_episodes: int = 300,
) -> None:
    # --- Lay agent: nap model co san hoac train moi theo dung cau hinh ---
    if model_path:
        agent = MonteCarloAgent.load(model_path)
        print(f"[*] Da nap model MC: {model_path} "
            f"({agent.number_of_actions} action, "
            f"{len(agent.q_table)} state)")
    else:
        train_adapter, _ = build_adapter(
            max_time, max_overtime, poisson_lambda,
            max_pending_calls, num_ambulances, seed,
        )
        agent = train_agent(train_adapter, train_episodes, seed,
                            save_path="mc_model.pkl")

    # --- Tao env HIEN THI moi tinh, dung cach baseline tao env ---
    adapter, city_map = build_adapter(
        max_time, max_overtime, poisson_lambda,
        max_pending_calls, num_ambulances, seed,
    )
    env = adapter.env
    if agent.number_of_actions != adapter.number_of_actions:
        raise ValueError(
            f"Model co {agent.number_of_actions} action nhung cau hinh "
            f"hien thi can {adapter.number_of_actions} action "
            f"({num_ambulances} xe). Hay train lai voi --ambulances "
            f"{num_ambulances} (bo --model de script tu train)."
        )

    print("=" * 85)
    print(f"      KHỞI CHẠY THỬ NGHIỆM MÔ PHỎNG ĐIỀU PHỐI {num_ambulances} XE CỨU THƯƠNG (80 NÚT, 4 BV)")
    print("=" * 85)
    print(f"- Thời gian mô phỏng tối đa sinh cuộc gọi : {max_time} phút ({max_time / 60:.1f} giờ)")
    print(f"- Tốc độ sinh cuộc gọi trung bình dự kiến : {poisson_lambda} cuộc gọi / phút (Poisson lambda = {poisson_lambda})")
    print(f"- Giới hạn tối đa cuộc gọi chờ (PENDING)   : {max_pending_calls} cuộc gọi (quá {max_pending_calls} bị từ chối/thất bại)")
    print(f"- Số lượng xe cứu thương                  : {num_ambulances} xe (Trạng thái ban đầu: Rảnh)")
    print(f"- Số lượng bệnh viện                      : 4 BV (2 gần trung tâm, 2 xa trung tâm)")
    print(f"- Giới hạn thời gian quá giờ tối đa        : {max_overtime} phút (chạm ngưỡng sẽ báo Truncated và dừng)")
    print(f"- Hạt giống ngẫu nhiên (seed)             : {seed}")
    print(f"- Chính sách điều phối                    : On-policy Monte Carlo (tham lam theo Q đã học)")
    print(f"- File log ghi lại diễn biến              : {log_filename}")
    print("=" * 85)

    log_entries: List[str] = []
    # Ghi nhan trang thai khoi dau tai Phut thu 0 (giong baseline).
    state = adapter.discretize_state()
    log_entries.append(env.format_log_step())

    start_real_time = time.time()
    done = False
    step_count = 0

    while not done:
        valid_actions = adapter.get_valid_actions()
        action = agent.select_action(state, valid_actions, training=False)
        state, _reward, done = adapter.step(action)
        step_count += 1

        log_entries.append(env.format_log_step())

        if step_count % 30 == 0 or done:
            curr_m = env.current_minute
            served = env.served_calls_count
            failed = env.failed_calls_count
            perfect = env.perfect_calls_count
            pending = len(env.pending_calls)
            active = len(env.active_calls)
            status_tag = " [TRUNCATED]" if env.is_truncated else ""
            print(
                f"[+] Phút {curr_m:>3}/{max_time} | "
                f"Đang chờ: {pending} | Đang xử lý: {active} | "
                f"Thành công: {served:>3} (Hoàn hảo: {perfect:>3}) | "
                f"Thất bại: {failed:>3} | "
                f"Reward: {env.total_reward:>8.2f}{status_tag}"
            )

    elapsed_time = time.time() - start_real_time

    # Ghi toan bo log vao file (dinh dang giong baseline_seed42.log)
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("\n\n".join(log_entries))
        f.write("\n")

    print("\n" + "=" * 85)
    print("                        KẾT QUẢ TỔNG KẾT MÔ PHỎNG")
    print("=" * 85)
    if env.is_truncated:
        print("\n" + env.format_backlog_report() + "\n")
    print(f"- Tổng thời gian mô phỏng hoàn tất : {env.current_minute} phút")
    print(f"- Trạng thái kết thúc              : {'Bị dừng (TRUNCATED - quá 300p quá giờ)' if env.is_truncated else 'Hoàn thành tất cả cuộc gọi (COMPLETED)'}")
    print(f"- Thời gian chạy thực tế           : {elapsed_time:.2f} giây")
    print(f"- Tổng số cuộc gọi đã phát sinh    : {len(env.calls)}")
    print(f"- Tổng số cuộc gọi THÀNH CÔNG      : {env.served_calls_count} "
          f"({env.served_calls_count / max(1, len(env.calls)) * 100:.1f}%)")
    print(f"  + Trong đó HOÀN HẢO (trước 50%)  : {env.perfect_calls_count}")
    print(f"- Tổng số cuộc gọi THẤT BẠI        : {env.failed_calls_count} "
          f"({env.failed_calls_count / max(1, len(env.calls)) * 100:.1f}%)")
    print(f"- Tổng điểm thưởng tích lũy (R)    : {env.total_reward:.2f}")

    # Thong ke tung xe
    print(f"\n- Thống kê hoạt động của {num_ambulances} xe cứu thương:")
    for aid, amb in env.ambulances.items():
        coord = amb.get_coord(city_map)
        print(f"  * Xe {aid}: Quãng đường di chuyển = {amb.total_distance_traveled:.2f} km | "
            f"Số lần reroute tính lại đường = {amb.reroute_count} | "
            f"Vị trí cuối = ({coord[0]:.2f}, {coord[1]:.2f}) km | Trạng thái = {amb.status.value}")

    print(f"\n[+] Đã lưu toàn bộ lịch sử log chi tiết vào file: {log_filename}")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hiển thị kết quả on-policy MC theo định dạng baseline")
    parser.add_argument("--poisson", "--lambda", dest="poisson_lambda", type=float, default=DEFAULT_POISSON_LAMBDA,
                        help=f"Tốc độ sinh cuộc gọi trung bình mỗi phút (mặc định: {DEFAULT_POISSON_LAMBDA})")
    parser.add_argument("--max_time", type=int, default=DEFAULT_MAX_TIME,
                        help=f"Thời gian mô phỏng sinh cuộc gọi tối đa (mặc định: {DEFAULT_MAX_TIME})")
    parser.add_argument("--max_overtime", type=int, default=DEFAULT_MAX_OVERTIME,
                        help=f"Thời gian quá giờ tối đa trước khi báo Truncated (mặc định: {DEFAULT_MAX_OVERTIME})")
    parser.add_argument("--max_pending", dest="max_pending_calls", type=int, default=DEFAULT_MAX_PENDING_CALLS,
                        help=f"Số cuộc gọi chờ tối đa (mặc định: {DEFAULT_MAX_PENDING_CALLS})")
    parser.add_argument("--ambulances", dest="num_ambulances", type=int, default=DEFAULT_NUM_AMBULANCES,
                        help=f"Số xe cứu thương (mặc định: {DEFAULT_NUM_AMBULANCES})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Hạt giống ngẫu nhiên (mặc định: {DEFAULT_SEED})")
    parser.add_argument("--log", dest="log_filename", type=str, default="mc_simulation.log",
                        help="Tên file log lưu kết quả (mặc định: mc_simulation.log)")
    parser.add_argument("--model", type=str, default=None,
                        help="File model MC đã train cùng cấu hình (vd mc_model.pkl); bỏ trống = train mới")
    parser.add_argument("--train-episodes", type=int, default=300,
                        help="Số episode train khi chưa có --model (mặc định: 300)")

    args = parser.parse_args()

    run_display(
        max_time=args.max_time,
        max_overtime=args.max_overtime,
        poisson_lambda=args.poisson_lambda,
        max_pending_calls=args.max_pending_calls,
        num_ambulances=args.num_ambulances,
        seed=args.seed,
        log_filename=args.log_filename,
        model_path=args.model,
        train_episodes=args.train_episodes,
    )
