"""
Script demo kiểm chứng Môi trường Bản đồ Đô thị 80 NÚT dạng cây phân nhánh.
Minh họa các tính năng theo yêu cầu:
1. ĐỦ 80 NÚT GIAO THÔNG (đánh số 1..80).
2. 4 BỆNH VIỆN:
   - 2 Bệnh viện GẦN TRUNG TÂM (mạng lưới đường dày đặc): BV 1 và BV 2.
   - 2 Bệnh viện Ở XA TRUNG TÂM (các cửa ngõ ngoại ô): BV 3 và BV 4.
3. BỆNH VIỆN CHỈ NẰM TRÊN 1 CUNG ĐƯỜNG DUY NHẤT (không nằm trên 2 đường hay 2 chiều).
4. VỊ TRÍ BỆNH VIỆN TÍNH CHÍNH XÁC BẰNG ĐỘ DÀI TỪ ĐIỂM XUẤT PHÁT (from_node).
5. ĐỘ DÀI ĐA DẠNG: Từ 1.00 km đến 6.00 km+.
6. ĐƯỜNG 2 CHIỀU ĐỘC LẬP & BIẾN THIÊN GIAO THÔNG THEO THỜI GIAN CHẠY.
7. ĐỊNH TUYẾN DIJKSTRA ĐẾN CẢ 4 BỆNH VIỆN.
"""

from __future__ import annotations

import sys
from map_env import CityMap, CONGESTION_NAMES

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def print_map_overview(city: CityMap) -> None:
    summary = city.get_summary()
    print("=" * 95)
    print("           BẢN ĐỒ THÀNH PHỐ THÔNG MINH 80 NÚT (DẠNG CÂY PHÂN NHÁNH)")
    print("=" * 95)
    print(f"- Tổng số nút giao thông        : {summary['num_nodes']} nút (đánh số từ 1 đến 80)")
    print(f"- Tổng số tuyến đường vô hướng  : {summary['num_roads_undirected']} tuyến")
    print(f"- Tổng số cung đường có hướng   : {summary['num_directed_edges']} cung (mỗi hướng đi riêng biệt)")
    print(f"- Thống kê độ dài đoạn đường    : Ngắn nhất = {summary['length_stats']['min_km']} km | "
          f"Dài nhất = {summary['length_stats']['max_km']} km | "
          f"Trung bình = {summary['length_stats']['avg_km']} km")
    print(f"- Phân loại 80 điểm giao cắt    :")
    for type_name, count in summary["intersection_types"].items():
        print(f"    + {type_name:<24}: {count} nút")

    print("\n" + "-" * 95)
    print("1. VỊ TRÍ 4 BỆNH VIỆN (2 GẦN TRUNG TÂM, 2 XA TRUNG TÂM)")
    print("-" * 95)
    for hosp in summary["hospitals"]:
        zone_str = "GẦN TRUNG TÂM (LÕI NỘI ĐÔ DÀY ĐẶC)" if hosp['id'] in (1, 2) else "XA TRUNG TÂM (NGOẠI Ô VÀNH ĐAI)"
        print(f"[*] Bệnh viện ID {hosp['id']}: {hosp['name']}")
        print(f"    - Phân vùng vị trí             : {zone_str}")
        print(f"    - Nằm trên DUY NHẤT cung đường : Xuất phát từ Nút {hosp['from_node']} ──> Nút {hosp['to_node']}")
        print(f"    - Chiều dài toàn bộ cung đường : {hosp['total_road_length_km']:.2f} km")
        print(f"    - Khoảng cách từ điểm xuất phát : {hosp['distance_from_start_km']:.2f} km tính từ Nút {hosp['from_node']}")
        print(f"    - Khoảng cách còn lại đến đích : {hosp['distance_to_end_km']:.2f} km đến Nút {hosp['to_node']}")
        print(f"    - Tọa độ địa lý trên bản đồ    : ({hosp['coord'][0]:.2f}, {hosp['coord'][1]:.2f}) km")
        print(f"    - Chú ý: Bệnh viện KHÔNG nằm trên chiều ngược lại {hosp['to_node']} ──> {hosp['from_node']}!")

    print("\n" + "-" * 95)
    print("2. DANH SÁCH 80 NÚT GIAO THÔNG (TRÍCH MẪU 25 NÚT ĐẦU VÀ CÁC NÚT ĐẶC TRƯNG)")
    print("-" * 95)
    print(f"{'ID':<4} | {'Tọa độ (x, y)':<15} | {'Bậc':<4} | {'Kiểu nút giao':<24} | {'Các nút kề kết nối'}")
    print("-" * 95)
    sample_nodes = list(range(1, 21)) + [21, 24, 41, 62, 65, 80]
    for node_id in sample_nodes:
        n = city.nodes[node_id]
        coord_str = f"({n.x:4.1f}, {n.y:4.1f}) km"
        print(f"{n.node_id:<4} | {coord_str:<15} | {n.degree:<4} | {n.intersection_type.value:<24} | {n.neighbors}")

    print("\n" + "-" * 95)
    print("3. CÁC HƯỚNG ĐI VỚI MŨI TÊN & ĐỘ TẮC ĐỘC LẬP (MẪU 12 TUYẾN ĐẦU TIÊN)")
    print("-" * 95)
    print(f"{'Tuyến đường':<12} | {'Độ dài':<9} | {'Hướng đi 1 (Xuất phát u -> v)':<32} | {'Hướng đi 2 (Xuất phát v -> u)':<32}")
    print("-" * 95)
    for u, v in city.road_pairs[:12]:
        e_uv = city.edges[(u, v)]
        e_vu = city.edges[(v, u)]
        road_str = f"N{u} <-> N{v}"
        uv_str = f"N{u} ──[{e_uv.congestion}]──> N{v} ({CONGESTION_NAMES[e_uv.congestion]})"
        vu_str = f"N{v} ──[{e_vu.congestion}]──> N{u} ({CONGESTION_NAMES[e_vu.congestion]})"
        print(f"{road_str:<12} | {e_uv.length:4.2f} km | {uv_str:<32} | {vu_str:<32}")


def demo_traffic_dynamics(city: CityMap, steps: int = 4) -> None:
    print("\n" + "=" * 95)
    print("4. MÔ PHỎNG BIẾN THIÊN ĐỘ TẮC ĐƯỜNG THEO THỜI GIAN CHẠY (DYNAMIC TRAFFIC)")
    print("=" * 95)
    sample_roads = [(1, 18), (19, 3), (4, 9), (21, 24), (62, 65), (13, 67)]
    print("Theo dõi độ tắc của 2 hướng đi riêng biệt (Xuất phát u->v vs v->u):")
    headers = [f"N{u}->N{v} vs N{v}->N{u}" for u, v in sample_roads]
    print(f"{'Thời điểm':<16} | " + " | ".join(headers))
    print("-" * 95)

    init_vals = [f"   {city.edges[(u,v)].congestion} vs {city.edges[(v,u)].congestion}   " for u, v in sample_roads]
    print(f"{'Bước t = 0':<16} | " + " | ".join(init_vals))

    for s in range(1, steps + 1):
        city.step_traffic(rush_hour=(s >= 2))
        step_vals = [f"   {city.edges[(u,v)].congestion} vs {city.edges[(v,u)].congestion}   " for u, v in sample_roads]
        rush_label = f"Bước t = {s} (Cao điểm)" if s >= 2 else f"Bước t = {s} (Thường)"
        print(f"{rush_label:<16} | " + " | ".join(step_vals))


def demo_hospital_routing(city: CityMap) -> None:
    print("\n" + "=" * 95)
    print("5. TÌM LỘ TRÌNH ĐẾN CẢ 4 BỆNH VIỆN THEO HƯỚNG ĐI ĐÚNG VÀ ĐỘ DÀI TỪ ĐIỂM XUẤT PHÁT")
    print("=" * 95)
    start_node = 32  # Nút cụt xa nhất ở đỉnh Tây Bắc
    print(f"Giả định xe cấp cứu đang tại Nút {start_node} ({city.nodes[start_node].name}):\n")

    for hid in range(1, 5):
        r_dist = city.route_to_hospital(start_node=start_node, hospital_id=hid, by_travel_time=False)
        r_time = city.route_to_hospital(start_node=start_node, hospital_id=hid, by_travel_time=True)
        hosp = city.hospitals[hid]
        zone_label = "Gần Trung Tâm" if hid in (1, 2) else "Xa Trung Tâm"

        print(f"[{chr(64 + hid)}] {hosp.name} [{zone_label} - Nằm trên cung N{hosp.from_node} ──> N{hosp.to_node}]:")
        print(f"    - Đường đi đến điểm xuất phát của BV (Nút {hosp.from_node}) : {' ──> '.join(str(n) for n in r_dist['path_to_start_node'])}")
        print(f"    - Đi thêm theo cung N{hosp.from_node} ──> N{hosp.to_node}   : {r_dist['distance_from_start_node']:.2f} km để tới cổng BV")
        print(f"    - Tổng quãng đường di chuyển                               : {r_dist['total_cost']:.2f} km")
        print(f"    - Tổng thời gian di chuyển (có tính kẹt xe)                : {r_time['total_cost']:.2f} phút\n")
    print("=" * 95)


if __name__ == "__main__":
    city = CityMap(seed=100)
    print_map_overview(city)
    demo_traffic_dynamics(city, steps=4)
    demo_hospital_routing(city)
