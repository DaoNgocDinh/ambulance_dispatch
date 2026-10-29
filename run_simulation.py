"""
Script Chạy Thử Nghiệm Mô Phỏng Điều Phối Xe Cứu Thương và Xuất File Log.
Thực hiện mô phỏng toàn diện theo đúng yêu cầu:
1. 4 xe cứu thương, vị trí ban đầu ngẫu nhiên, ban đầu luôn Rảnh (IDLE).
2. Cuộc gọi cấp cứu phát sinh theo Poisson (lambda = 1), tối đa 3 cuộc gọi PENDING.
3. 4 Bệnh viện với thời gian gửi bệnh nhân khác nhau (3 - 5 phút).
4. Vận tốc và tỉ lệ tắc đường phân biệt giữa Trung tâm và Ngoại ô.
5. Mô phỏng kéo dài tối đa 360 phút (sau đó không sinh thêm cuộc gọi, tiếp tục xử lý đến khi hết).
6. Lưu toàn bộ diễn biến từng phút vào file `simulation.log` theo đúng format chuẩn.
"""

from __future__ import annotations

import sys
import time
import argparse
from typing import List

from map_env import CityMap
from dispatch_env import AmbulanceDispatchEnv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_experiment(
    max_time: int = 360,
    poisson_lambda: float = 0.1,
    max_pending_calls: int = 3,
    num_ambulances: int = 6,
    seed: int = 42,
    log_filename: str = "simulation.log",
) -> None:
    print("=" * 85)
    print("      KHỞI CHẠY THỬ NGHIỆM MÔ PHỎNG ĐIỀU PHỐI 4 XE CỨU THƯƠNG (80 NÚT, 4 BV)")
    print("=" * 85)
    print(f"- Thời gian mô phỏng tối đa sinh cuộc gọi : {max_time} phút (6.0 giờ)")
    print(f"- Tốc độ sinh cuộc gọi trung bình dự kiến : {poisson_lambda} cuộc gọi / phút (Poisson lambda = {poisson_lambda})")
    print(f"- Giới hạn tối đa cuộc gọi chờ (PENDING)   : {max_pending_calls} cuộc gọi (quá {max_pending_calls} bị từ chối/thất bại)")
    print(f"- Số lượng xe cứu thương                  : {num_ambulances} xe (Trạng thái ban đầu: Rảnh)")
    print(f"- Số lượng bệnh viện                      : 4 BV (2 gần trung tâm, 2 xa trung tâm)")
    print(f"- Hạt giống ngẫu nhiên (seed)             : {seed}")
    print(f"- File log ghi lại diễn biến              : {log_filename}")
    print("=" * 85)

    city_map = CityMap(seed=seed)
    env = AmbulanceDispatchEnv(
        city_map=city_map,
        num_ambulances=num_ambulances,
        max_time=max_time,
        poisson_lambda=poisson_lambda,
        max_pending_calls=max_pending_calls,
        seed=seed,
    )

    log_entries: List[str] = []
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
            print(
                f"[+] Phút {curr_m:>3}/{max_time} | "
                f"Đang chờ: {pending} | Đang xử lý: {active} | "
                f"Thành công: {served:>3} (Hoàn hảo: {perfect:>3}) | "
                f"Thất bại: {failed:>3}"
            )

        # Biện pháp an toàn tránh vòng lặp vô hạn (tối đa max_time + 120 phút để dọn sạch các cuộc gọi còn lại)
        if env.current_minute > max_time + 120:
            print("[!] Cảnh báo: Đã vượt quá thời gian gia hạn xử lý sau 360 phút.")
            break

    elapsed_time = time.time() - start_real_time

    # Ghi toàn bộ log vào file
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("\n\n".join(log_entries))
        f.write("\n")

    print("\n" + "=" * 85)
    print("                        KẾT QUẢ TỔNG KẾT MÔ PHỎNG")
    print("=" * 85)
    print(f"- Tổng thời gian mô phỏng hoàn tất : {env.current_minute} phút")
    print(f"- Thời gian chạy thực tế           : {elapsed_time:.2f} giây")
    print(f"- Tổng số cuộc gọi đã phát sinh    : {len(env.calls)}")
    print(f"- Tổng số cuộc gọi THÀNH CÔNG      : {env.served_calls_count} "
          f"({env.served_calls_count / max(1, len(env.calls)) * 100:.1f}%)")
    print(f"  + Trong đó HOÀN HẢO (trước 50%)  : {env.perfect_calls_count}")
    print(f"- Tổng số cuộc gọi THẤT BẠI        : {env.failed_calls_count} "
          f"({env.failed_calls_count / max(1, len(env.calls)) * 100:.1f}%)")

    # Thống kê từng xe
    print("\n- Thống kê hoạt động của 4 xe cứu thương:")
    for aid, amb in env.ambulances.items():
        coord = amb.get_coord(city_map)
        print(f"  * Xe {aid}: Quãng đường di chuyển = {amb.total_distance_traveled:.2f} km | "
              f"Số lần reroute tính lại đường = {amb.reroute_count} | "
              f"Vị trí cuối = ({coord[0]:.2f}, {coord[1]:.2f}) km | Trạng thái = {amb.status.value}")

    print(f"\n[+] Đã lưu toàn bộ lịch sử log chi tiết vào file: {log_filename}")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mô phỏng Điều phối 4 Xe Cứu Thương trên bản đồ 80 nút")
    parser.add_argument("--poisson", "--lambda", dest="poisson_lambda", type=float, default=0.1,
                        help="Tốc độ sinh cuộc gọi trung bình mỗi phút (Poisson lambda, mặc định: 0.1)")
    parser.add_argument("--max_time", type=int, default=360,
                        help="Thời gian mô phỏng sinh cuộc gọi tối đa (phút, mặc định: 360)")
    parser.add_argument("--max_pending", dest="max_pending_calls", type=int, default=3,
                        help="Số lượng cuộc gọi chờ tối đa trong hàng đợi PENDING (mặc định: 3)")
    parser.add_argument("--ambulances", dest="num_ambulances", type=int, default=6,
                        help="Số lượng xe cứu thương (mặc định: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Hạt giống sinh số ngẫu nhiên (mặc định: 42)")
    parser.add_argument("--log", dest="log_filename", type=str, default="simulation.log",
                        help="Tên file log lưu kết quả (mặc định: simulation.log)")

    args = parser.parse_args()

    run_experiment(
        max_time=args.max_time,
        poisson_lambda=args.poisson_lambda,
        max_pending_calls=args.max_pending_calls,
        num_ambulances=args.num_ambulances,
        seed=args.seed,
        log_filename=args.log_filename,
    )
