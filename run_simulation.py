"""
Script Chạy Thử Nghiệm Mô Phỏng Điều Phối Xe Cứu Thương và Xuất File Log.
Thực hiện mô phỏng toàn diện theo đúng cấu hình chuẩn:
1. 6 xe cứu thương (mặc định), vị trí ban đầu ngẫu nhiên, ban đầu luôn Rảnh (IDLE).
2. Cuộc gọi cấp cứu phát sinh theo Poisson (lambda = 0.1), tối đa 3 cuộc gọi PENDING.
3. 4 Bệnh viện với thời gian gửi bệnh nhân khác nhau (3 - 5 phút).
4. Vận tốc và tỉ lệ tắc đường phân biệt giữa Trung tâm và Ngoại ô.
5. Mô phỏng kéo dài tối đa 360 phút (sau đó không sinh thêm cuộc gọi, tiếp tục xử lý đến khi hết hoặc chạm ngưỡng 300 phút quá giờ).
6. Lưu toàn bộ diễn biến từng phút vào file `simulation.log` theo đúng format chuẩn.
"""

from __future__ import annotations

import sys
import time
import argparse
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_experiment(
    max_time: int = DEFAULT_MAX_TIME,
    max_overtime: int = DEFAULT_MAX_OVERTIME,
    poisson_lambda: float = DEFAULT_POISSON_LAMBDA,
    max_pending_calls: int = DEFAULT_MAX_PENDING_CALLS,
    num_ambulances: int = DEFAULT_NUM_AMBULANCES,
    seed: int = DEFAULT_SEED,
    log_filename: str = "simulation.log",
) -> None:
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
    print(f"- File log ghi lại diễn biến              : {log_filename}")
    print("=" * 85)

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

    log_entries: List[str] = []
    # Ghi nhận trạng thái khởi đầu tại Phút thứ 0 (cuộc gọi có thể sinh ra tại t=0)
    log_entries.append(env.format_log_step())

    start_real_time = time.time()
    done = False
    step_count = 0

    while not done:
        obs, reward, done, info = env.step(auto_dispatch=True)
        step_count += 1

        log_block = env.format_log_step()
        log_entries.append(log_block)

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

    # Ghi toàn bộ log vào file
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

    # Thống kê từng xe
    print(f"\n- Thống kê hoạt động của {num_ambulances} xe cứu thương:")
    for aid, amb in env.ambulances.items():
        coord = amb.get_coord(city_map)
        print(f"  * Xe {aid}: Quãng đường di chuyển = {amb.total_distance_traveled:.2f} km | "
              f"Số lần reroute tính lại đường = {amb.reroute_count} | "
              f"Vị trí cuối = ({coord[0]:.2f}, {coord[1]:.2f}) km | Trạng thái = {amb.status.value}")

    print(f"\n[+] Đã lưu toàn bộ lịch sử log chi tiết vào file: {log_filename}")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mô phỏng Điều phối Xe Cứu Thương trên bản đồ 80 nút")
    parser.add_argument("--poisson", "--lambda", dest="poisson_lambda", type=float, default=DEFAULT_POISSON_LAMBDA,
                        help=f"Tốc độ sinh cuộc gọi trung bình mỗi phút (Poisson lambda, mặc định: {DEFAULT_POISSON_LAMBDA})")
    parser.add_argument("--max_time", type=int, default=DEFAULT_MAX_TIME,
                        help=f"Thời gian mô phỏng sinh cuộc gọi tối đa (phút, mặc định: {DEFAULT_MAX_TIME})")
    parser.add_argument("--max_overtime", type=int, default=DEFAULT_MAX_OVERTIME,
                        help=f"Thời gian quá giờ tối đa trước khi báo Truncated (phút, mặc định: {DEFAULT_MAX_OVERTIME})")
    parser.add_argument("--max_pending", dest="max_pending_calls", type=int, default=DEFAULT_MAX_PENDING_CALLS,
                        help=f"Số lượng cuộc gọi chờ tối đa trong hàng đợi PENDING (mặc định: {DEFAULT_MAX_PENDING_CALLS})")
    parser.add_argument("--ambulances", dest="num_ambulances", type=int, default=DEFAULT_NUM_AMBULANCES,
                        help=f"Số lượng xe cứu thương (mặc định: {DEFAULT_NUM_AMBULANCES})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Hạt giống sinh số ngẫu nhiên (mặc định: {DEFAULT_SEED})")
    parser.add_argument("--log", dest="log_filename", type=str, default="simulation.log",
                        help="Tên file log lưu kết quả (mặc định: simulation.log)")

    args = parser.parse_args()

    run_experiment(
        max_time=args.max_time,
        max_overtime=args.max_overtime,
        poisson_lambda=args.poisson_lambda,
        max_pending_calls=args.max_pending_calls,
        num_ambulances=args.num_ambulances,
        seed=args.seed,
        log_filename=args.log_filename,
    )
