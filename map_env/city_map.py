"""
Quản lý Toàn bộ Bản đồ Thành phố (CityMap) gồm 80 NÚT, các cung đường 2 chiều và 4 BỆNH VIỆN.
Cấu trúc mạng lưới giao thông dạng cây phân nhánh (Tree-like hierarchical urban network):
- 80 nút giao thông (đánh số 1..80), phân loại chuẩn theo bậc:
  + Ngã 4 (Crossroads - Bậc 4)
  + Ngã 3 (T-junction - Bậc 3)
  + Đường rẽ / Nối tiếp (Turn - Bậc 2)
  + Đường cụt (Dead-end - Bậc 1)
- Tự động kết nối và sinh độ dài các cung đường theo khoảng cách tọa độ (km).
- Độ đa dạng độ dài cao: từ các ngõ nội đô ngắn (~1.0 km) đến các đại lộ và trục cao tốc ngoại ô (~6.0 - 7.5 km).
- Các đường nối là 2 chiều: mỗi đường gồm 2 cung có hướng u->v và v->u với độ tắc độc lập {1, 2, 3}.
- 4 BỆNH VIỆN (nằm trên DUY NHẤT 1 cung đường có hướng, vị trí đo từ điểm xuất phát):
  + 2 bệnh viện GẦN TRUNG TÂM (khu vực nội đô mạng lưới đường dày đặc): BV 1 và BV 2.
  + 2 bệnh viện Ở XA TRUNG TÂM (các cửa ngõ ngoại ô vành đai): BV 3 (Ngoại ô Bắc) và BV 4 (Ngoại ô Tây Nam).
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .edge import (
    CONGESTION_CLEAR,
    CONGESTION_HEAVY,
    CONGESTION_MODERATE,
    DirectedEdge,
)
from .hospital import HospitalOnRoad
from .node import IntersectionType, MapNode


class CityMap:
    """
    Lớp quản lý bản đồ giao thông đô thị 80 nút và 4 bệnh viện.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.num_nodes: int = 80
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # Danh bạ các nút: node_id -> MapNode
        self.nodes: Dict[int, MapNode] = {}

        # Danh bạ các cung đường có hướng: (u, v) -> DirectedEdge
        self.edges: Dict[Tuple[int, int], DirectedEdge] = {}

        # Danh sách kề cho đồ thị có hướng: u -> [DirectedEdge]
        self.adj: Dict[int, List[DirectedEdge]] = {i: [] for i in range(1, self.num_nodes + 1)}

        # Danh sách các cặp đường vô hướng: (min(u,v), max(u,v))
        self.road_pairs: List[Tuple[int, int]] = []

        # 4 bệnh viện nằm trên 4 cung đường có hướng riêng biệt
        self.hospitals: Dict[int, HospitalOnRoad] = {}

        # Bước thời gian mô phỏng biến thiên giao thông
        self.time_step: int = 0
        self.last_traffic_transitions: Dict[str, List[str]] = {
            "tac_to_binh_thuong": [],
            "thong_thoang_to_binh_thuong": [],
            "binh_thuong_to_tac": [],
            "binh_thuong_to_thong_thoang": [],
            "tac_to_thong_thoang": [],
            "thong_thoang_to_tac": [],
        }

        # Khởi tạo bản đồ
        self._build_80_nodes()
        self._build_roads()
        self._place_hospitals()
        self.randomize_traffic(initial=True)

    # ========================================================================
    # 1. KHỞI TẠO 80 NÚT VỚI TỌA ĐỘ VÀ PHÂN BỐ ĐÔ THỊ
    # ========================================================================
    def _build_80_nodes(self) -> None:
        raw_node_specs = [
            # ----------------- KHU VỰC 1: NỘI ĐÔ TRUNG TÂM (MẠNG LƯỚI DÀY ĐẶC) -----------------
            (1, 18.0, 16.0, "Ngã 4 Trung tâm Lõi Thành phố"),
            (2, 18.0, 18.0, "Ngã 4 Bắc Lõi Nội đô"),
            (3, 21.0, 16.0, "Ngã 4 Đông Lõi Nội đô"),
            (4, 18.0, 14.0, "Ngã 4 Nam Lõi Nội đô"),
            (5, 15.0, 16.0, "Ngã 4 Tây Lõi Nội đô"),
            (6, 15.5, 18.0, "Ngã 4 Tây Bắc Nội đô"),
            (7, 20.5, 18.0, "Ngã 4 Đông Bắc Nội đô"),
            (8, 20.5, 14.0, "Ngã 4 Đông Nam Nội đô"),
            (9, 15.5, 14.0, "Ngã 4 Tây Nam Nội đô"),
            (10, 18.0, 20.5, "Ngã 4 Cửa ngõ Bắc Nội đô"),
            (11, 23.5, 16.0, "Ngã 4 Cửa ngõ Đông Nội đô"),
            (12, 18.0, 11.5, "Ngã 4 Cửa ngõ Nam Nội đô"),
            (13, 12.5, 16.0, "Ngã 4 Cửa ngõ Tây Nội đô"),
            (14, 13.5, 19.5, "Ngã 3 Nội đô Tây Bắc"),
            (15, 22.5, 19.5, "Ngã 3 Nội đô Đông Bắc"),
            (16, 22.5, 12.5, "Ngã 3 Nội đô Đông Nam"),
            (17, 13.5, 12.5, "Ngã 3 Nội đô Tây Nam"),
            (18, 18.0, 17.0, "Ngã 3 Phố đi bộ Trung tâm"),
            (19, 19.5, 16.0, "Ngã 3 Phố tài chính Trung tâm"),
            (20, 18.0, 15.0, "Ngã 3 Phố thương mại Nam Lõi"),

            # ----------------- KHU VỰC 2: PHÍA BẮC & ĐÔNG BẮC -----------------
            (21, 18.0, 23.5, "Ngã 4 Trục Bắc Vành đai 2"),
            (22, 14.5, 22.0, "Đường rẽ Cửa ngõ Tây Bắc 2"),
            (23, 21.5, 22.5, "Đường rẽ Cửa ngõ Đông Bắc 2"),
            (24, 18.0, 27.0, "Ngã 4 Trục Bắc Vành đai 3"),
            (25, 11.5, 24.5, "Ngã 3 Khu Dân cư Bắc"),
            (26, 24.5, 24.5, "Ngã 3 KCN Công nghệ cao Đông Bắc"),
            (27, 8.5, 26.5, "Đường rẽ Làng sinh thái Tây Bắc"),
            (28, 27.5, 26.5, "Đường rẽ Cụm Logistics Cảng Đông Bắc"),
            (29, 18.0, 30.5, "Đường cụt Đỉnh Cực Bắc Đô thị"),
            (30, 15.0, 29.0, "Đường rẽ Thung Lũng Sinh Thái"),
            (31, 21.0, 29.0, "Đường rẽ Khu Lâm Viên Bắc"),
            (32, 5.5, 28.5, "Đường cụt Khu Bảo Tồn Tây Bắc"),
            (33, 10.0, 29.0, "Đường cụt Trạm Kiểm Lâm Bắc"),
            (34, 30.0, 28.5, "Đường cụt Mũi Đông Cảng Biển"),
            (35, 26.5, 29.0, "Đường cụt Khu Công Nghiệp Biển"),
            (36, 23.0, 21.5, "Đường cụt Khu Nhà Ở Cán Bộ Bắc"),
            (37, 12.5, 21.0, "Đường cụt Cụm Chung Cư Bắc"),
            (38, 16.5, 25.0, "Đường cụt Khu Nghiên Cứu Bắc"),
            (39, 19.5, 25.0, "Đường cụt Viện Công Nghệ Môi Trường"),
            (40, 11.5, 27.5, "Đường cụt Nông Trường Bắc"),

            # ----------------- KHU VỰC 3: PHÍA ĐÔNG & ĐÔNG NAM -----------------
            (41, 26.0, 16.0, "Ngã 4 Trục Cao tốc Đông"),
            (42, 24.5, 19.0, "Đường rẽ Cảng Cạn Đông Bắc"),
            (43, 24.5, 13.0, "Đường rẽ Vành đai Cảng Đông Nam"),
            (44, 29.5, 16.0, "Ngã 4 Cửa ngõ Đại dương"),
            (45, 27.5, 13.0, "Ngã 4 Đô thị Cảng Đông Nam"),
            (46, 27.5, 19.0, "Ngã 3 Khu Chế Xuất Đông"),
            (47, 33.0, 16.0, "Đường rẽ Cảng Quốc Tế Đông"),
            (48, 30.5, 19.5, "Đường rẽ Hải Đăng Đông"),
            (49, 30.5, 12.5, "Đường rẽ Bến Phà Đông"),
            (50, 29.0, 10.0, "Đường cụt Cảng Cá Đông Nam"),
            (51, 25.5, 10.0, "Đường rẽ Vành đai Nam Cảng"),
            (52, 22.5, 9.5, "Ngã 3 Trục Đông Nam"),
            (53, 25.0, 7.0, "Đường rẽ Cửa ngõ Nam - Đông"),
            (54, 28.0, 7.0, "Đường cụt Kho Ngoại Quan Đông Nam"),
            (55, 24.0, 4.5, "Đường cụt Cụm Đóng Tàu Nam"),
            (56, 21.0, 6.5, "Đường cụt Cụm Dân cư Đông Nam"),
            (57, 28.0, 14.5, "Đường cụt Trạm Hoa Tiêu"),
            (58, 28.5, 17.5, "Đường cụt Khu Hải Quan Đông"),
            (59, 31.5, 16.0, "Đường cụt Đảo Nổi Công Nghệ"),
            (60, 23.5, 14.5, "Đường cụt Khu Cư xá Thủy Thủ"),

            # ----------------- KHU VỰC 4: PHÍA NAM & TÂY NAM -----------------
            (61, 18.0, 8.5, "Ngã 3 Trục Nam Vành đai 2"),
            (62, 14.5, 9.0, "Đường rẽ Cửa ngõ Tây Nam 2"),
            (63, 20.0, 9.0, "Đường rẽ Nam - Đông Nam"),
            (64, 18.0, 4.5, "Ngã 4 Trục Cao tốc Nam"),
            (65, 11.5, 7.0, "Ngã 3 Đô thị Mới Tây Nam"),
            (66, 10.0, 12.0, "Ngã 3 Khu Công nghệ Tây"),
            (67, 6.5, 16.0, "Ngã 3 Trục Cao tốc Tây Vành đai"),
            (68, 6.5, 11.5, "Đường rẽ KCN Tây"),
            (69, 18.0, 1.0, "Đường cụt Trạm Thu phí Cực Nam"),
            (70, 14.5, 3.5, "Đường rẽ Làng Hoa Sa Đéc Nam"),
            (71, 21.0, 3.0, "Đường rẽ Khu Nông Nghiệp Nam"),
            (72, 7.5, 5.5, "Đường rẽ KCN Nặng Tây Nam"),
            (73, 4.0, 4.5, "Đường cụt Cụm Cơ khí Tây Nam"),
            (74, 2.5, 16.0, "Đường cụt Cửa khẩu Tây"),
            (75, 4.5, 19.0, "Đường cụt Đồi Thông Tây"),
            (76, 6.5, 21.0, "Đường cụt Khu Quân sự Tây Bắc"),
            (77, 6.5, 8.5, "Đường cụt Kho Dự trữ Tây"),
            (78, 12.5, 5.0, "Đường cụt Cụm Gốm Sứ Nam"),
            (79, 15.0, 2.0, "Đường cụt Trạm Biến áp 500kV"),
            (80, 20.0, 5.5, "Đường cụt Khu Tái định cư Nam"),
        ]

        for node_id, x, y, name in raw_node_specs:
            self.nodes[node_id] = MapNode(node_id=node_id, x=x, y=y, name=name)

    # ========================================================================
    # 2. TỰ ĐỘNG NỐI ĐƯỜNG VÀ TÍNH ĐỘ DÀI ĐA DẠNG
    # ========================================================================
    def _build_roads(self) -> None:
        connections = [
            # ----------------- NỘI ĐÔ DÀY ĐẶC (DOWNTOWN GRID & RADIALS) -----------------
            (1, 18, "Đoạn phố đi bộ Bắc"),                      # 1.00 km (Ngắn)
            (18, 2, "Đoạn phố đi bộ Bắc mở rộng"),               # 1.00 km (Ngắn)
            (1, 19, "Phố tài chính Đông"),                      # 1.50 km (Ngắn)
            (19, 3, "Đại lộ Đông - Chứa BV1"),                  # 1.50 km (Ngắn, Chứa BV1)
            (1, 20, "Phố thương mại Nam"),                      # 1.00 km (Ngắn)
            (20, 4, "Phố thương mại Nam mở rộng"),              # 1.00 km (Ngắn)
            (1, 5, "Đại lộ Tây Trung tâm"),                     # 3.00 km (Trung bình)

            (2, 6, "Đường liên khu Bắc - Tây Bắc"),             # ~2.50 km
            (2, 7, "Đường liên khu Bắc - Đông Bắc"),             # ~2.50 km
            (2, 10, "Đại lộ Cửa ngõ Bắc"),                      # 2.50 km

            (3, 7, "Đường vành đai Đông Bắc Lõi"),              # ~2.06 km
            (3, 8, "Đường vành đai Đông Nam Lõi"),              # ~2.06 km
            (3, 11, "Đại lộ Cửa ngõ Đông"),                     # 2.50 km

            (4, 8, "Đường liên khu Nam - Đông Nam"),            # ~2.50 km
            (4, 9, "Đường liên khu Nam - Tây Nam - Chứa BV2"),  # ~2.50 km (Chứa BV2)
            (4, 12, "Đại lộ Cửa ngõ Nam"),                      # 2.50 km

            (5, 6, "Đường vành đai Tây Bắc Lõi"),               # ~2.06 km
            (5, 9, "Đường vành đai Tây Nam Lõi"),               # ~2.06 km
            (5, 13, "Đại lộ Cửa ngõ Tây"),                      # 2.50 km

            (6, 14, "Đường nhánh Tây Bắc"),                     # ~2.50 km
            (7, 15, "Đường nhánh Đông Bắc"),                    # ~2.50 km
            (8, 16, "Đường nhánh Đông Nam"),                    # ~2.50 km
            (9, 17, "Đường nhánh Tây Nam"),                     # ~2.50 km

            # ----------------- NGOẠI Ô BẮC (SUBURBAN NORTH) -----------------
            (10, 21, "Trục Bắc Vành đai 2"),                    # 3.00 km
            (10, 22, "Đường Cửa ngõ Tây Bắc 2"),                # ~3.81 km
            (10, 23, "Đường Cửa ngõ Đông Bắc 2"),               # ~3.91 km

            (21, 24, "Trục Bắc Vành đai 3 - Chứa BV3"),         # 3.50 km (Chứa BV3)
            (21, 38, "Ngõ Viện Nghiên Cứu Bắc"),                # ~2.12 km
            (21, 39, "Ngõ Viện Môi Trường"),                    # ~2.12 km

            (22, 25, "Đường liên xã Bắc"),                      # ~3.91 km
            (22, 37, "Đường vào Chung cư Bắc"),                 # ~2.24 km

            (23, 26, "Đường KCN Đông Bắc"),                     # ~3.61 km
            (23, 36, "Đường Cư xá Cán bộ"),                     # ~1.80 km

            (24, 29, "Trục Cực Bắc Đô thị"),                    # 3.50 km
            (24, 30, "Đường Thung Lũng Sinh Thái"),              # ~3.61 km
            (24, 31, "Đường Lâm Viên Bắc"),                     # ~3.61 km
            (30, 31, "Đường vành đai Đỉnh Bắc"),                # 6.00 km (Vành đai liên kết)

            (25, 27, "Đường Làng sinh thái Tây Bắc"),           # ~3.61 km
            (25, 40, "Đường Nông Trường Bắc"),                  # 3.00 km
            (27, 32, "Đường Khu Bảo tồn Tây Bắc"),              # ~3.61 km
            (27, 33, "Đường Trạm Kiểm lâm"),                    # ~2.92 km

            (26, 28, "Đường Cụm Logistics Cảng"),               # ~3.61 km
            (28, 34, "Đường ra Mũi Đông"),                      # ~3.20 km
            (28, 35, "Đường vào KCN Biển"),                     # ~2.69 km

            # ----------------- NGOẠI Ô ĐÔNG (SUBURBAN EAST) -----------------
            (11, 41, "Trục Cao tốc Đông"),                      # 2.50 km
            (11, 42, "Đường Cảng Cạn Đông Bắc"),                # ~3.16 km
            (11, 43, "Đường Cảng Đông Nam"),                    # ~3.16 km

            (41, 44, "Đại lộ Cửa ngõ Đại dương"),               # 3.50 km
            (41, 45, "Đường Đô thị Cảng Đông Nam"),             # ~3.35 km
            (41, 46, "Đường Khu Chế Xuất Đông"),                # ~3.35 km

            (44, 47, "Trục Cảng Quốc Tế Đông"),                 # 3.50 km
            (44, 48, "Đường ra Hải Đăng Đông"),                 # ~3.64 km
            (44, 49, "Đường ra Bến Phà Đông"),                  # ~3.64 km
            (47, 59, "Đường Đảo Nổi Công Nghệ"),                # 1.50 km (Ngắn)
            (48, 49, "Đường ven biển Đông"),                    # 7.00 km (Dài - Vành đai ven biển)

            (45, 50, "Đường Cảng Cá Đông Nam"),                 # ~3.35 km
            (45, 51, "Đường Vành đai Nam Cảng"),                # ~3.61 km
            (45, 57, "Đường Trạm Hoa Tiêu"),                    # ~1.58 km

            (46, 58, "Đường Khu Hải Quan Đông"),                # ~1.80 km

            (43, 52, "Trục Đông Nam Liên khu"),                 # ~4.03 km
            (52, 53, "Đường Cửa ngõ Nam - Đông"),               # ~3.54 km
            (52, 56, "Đường Cụm Dân cư Đông Nam"),              # ~3.35 km
            (52, 60, "Đường Cư xá Thủy Thủ"),                   # ~5.10 km

            (53, 54, "Đường Kho Ngoại Quan"),                   # 3.00 km
            (53, 55, "Đường Đóng Tàu Nam"),                     # ~2.69 km

            # ----------------- NGOẠI Ô NAM & TÂY NAM (SUBURBAN SOUTH & WEST) -----------------
            (12, 61, "Trục Nam Vành đai 2"),                    # 3.00 km
            (12, 62, "Đường Cửa ngõ Tây Nam 2"),                # ~4.30 km
            (12, 63, "Đường Nam - Đông Nam"),                   # ~3.20 km

            (61, 64, "Trục Cao tốc Nam"),                       # 4.00 km
            (62, 65, "Đường Đô thị Mới Tây Nam - Chứa BV4"),    # ~3.61 km (Chứa BV4)

            (64, 69, "Đại lộ Trạm Thu phí Cực Nam"),            # 3.50 km
            (64, 70, "Đường Làng Hoa Sa Đéc"),                  # ~3.64 km
            (64, 71, "Đường Nông Nghiệp Công Nghệ Cao"),        # ~3.35 km
            (70, 71, "Đường Vành đai Nông nghiệp Nam"),         # ~6.52 km (Dài)
            (70, 79, "Đường Trạm Biến áp 500kV"),               # ~1.58 km (Ngắn)
            (71, 80, "Đường Tái định cư Nam"),                  # ~2.69 km

            (65, 72, "Đường KCN Nặng Tây Nam"),                 # ~4.27 km
            (65, 78, "Đường Cụm Gốm Sứ Nam"),                   # ~2.24 km
            (72, 73, "Đường Cụm Cơ khí Tây Nam"),               # ~3.64 km

            (13, 66, "Đường Khu Công nghệ Tây"),                # ~4.72 km
            (13, 67, "Trục Cao tốc Tây Vành đai"),              # 6.00 km (Rất dài)

            (66, 68, "Đường KCN Tây"),                          # ~3.54 km
            (66, 77, "Đường Kho Dự trữ Tây"),                   # ~4.95 km

            (67, 74, "Đường ra Cửa khẩu Tây"),                  # 4.00 km
            (67, 75, "Đường Đồi Thông Tây"),                    # ~3.61 km
            (67, 76, "Đường Khu Quân sự Tây Bắc"),              # 5.00 km
        ]

        for u, v, road_name in connections:
            pair = (min(u, v), max(u, v))
            self.road_pairs.append(pair)

            self.nodes[u].add_neighbor(v)
            self.nodes[v].add_neighbor(u)

            node_u = self.nodes[u]
            node_v = self.nodes[v]
            raw_dist = math.hypot(node_u.x - node_v.x, node_u.y - node_v.y)
            length_km = round(float(raw_dist), 2)

            is_center_edge = (u <= 20 and v <= 20)
            edge_uv = DirectedEdge(
                from_node=u,
                to_node=v,
                length=length_km,
                congestion=CONGESTION_CLEAR,
                name=f"{road_name} ({u}->{v})",
                is_center=is_center_edge,
            )
            edge_vu = DirectedEdge(
                from_node=v,
                to_node=u,
                length=length_km,
                congestion=CONGESTION_CLEAR,
                name=f"{road_name} ({v}->{u})",
                is_center=is_center_edge,
            )

            self.edges[(u, v)] = edge_uv
            self.edges[(v, u)] = edge_vu

            self.adj[u].append(edge_uv)
            self.adj[v].append(edge_vu)

    # ========================================================================
    # 3. SINH RA 4 BỆNH VIỆN: 2 GẦN TRUNG TÂM, 2 Ở XA TRUNG TÂM
    # ========================================================================
    def _place_hospitals(self) -> None:
        # ==================== BỆNH VIỆN 1 (GẦN TRUNG TÂM) ====================
        edge_19_3 = self.edges[(19, 3)]
        dist_1 = 0.80  # km từ điểm đi Nút 19
        ratio_1 = dist_1 / edge_19_3.length
        h1_x = self.nodes[19].x + ratio_1 * (self.nodes[3].x - self.nodes[19].x)
        h1_y = self.nodes[19].y + ratio_1 * (self.nodes[3].y - self.nodes[19].y)

        self.hospitals[1] = HospitalOnRoad(
            hospital_id=1,
            name="Bệnh viện Đa khoa Trung tâm Thành phố (BV 1)",
            from_node=19,
            to_node=3,
            road_length=edge_19_3.length,
            distance_from_start=dist_1,
            coord=(round(h1_x, 2), round(h1_y, 2)),
            handover_time=3.0,
        )

        # ==================== BỆNH VIỆN 2 (GẦN TRUNG TÂM) ====================
        edge_4_9 = self.edges[(4, 9)]
        dist_2 = 1.00  # km từ điểm đi Nút 4
        ratio_2 = dist_2 / edge_4_9.length
        h2_x = self.nodes[4].x + ratio_2 * (self.nodes[9].x - self.nodes[4].x)
        h2_y = self.nodes[4].y + ratio_2 * (self.nodes[9].y - self.nodes[4].y)

        self.hospitals[2] = HospitalOnRoad(
            hospital_id=2,
            name="Bệnh viện Cấp cứu Tim mạch Nội đô (BV 2)",
            from_node=4,
            to_node=9,
            road_length=edge_4_9.length,
            distance_from_start=dist_2,
            coord=(round(h2_x, 2), round(h2_y, 2)),
            handover_time=4.0,
        )

        # ==================== BỆNH VIỆN 3 (XA TRUNG TÂM - PHÍA BẮC) ====================
        edge_21_24 = self.edges[(21, 24)]
        dist_3 = 1.80  # km từ điểm đi Nút 21
        ratio_3 = dist_3 / edge_21_24.length
        h3_x = self.nodes[21].x + ratio_3 * (self.nodes[24].x - self.nodes[21].x)
        h3_y = self.nodes[21].y + ratio_3 * (self.nodes[24].y - self.nodes[21].y)

        self.hospitals[3] = HospitalOnRoad(
            hospital_id=3,
            name="Bệnh viện Đa khoa Khu vực Bắc Đô thị (BV 3)",
            from_node=21,
            to_node=24,
            road_length=edge_21_24.length,
            distance_from_start=dist_3,
            coord=(round(h3_x, 2), round(h3_y, 2)),
            handover_time=5.0,
        )

        # ==================== BỆNH VIỆN 4 (XA TRUNG TÂM - PHÍA TÂY NAM) ====================
        edge_62_65 = self.edges[(62, 65)]
        dist_4 = 1.80  # km từ điểm đi Nút 62
        ratio_4 = dist_4 / edge_62_65.length
        h4_x = self.nodes[62].x + ratio_4 * (self.nodes[65].x - self.nodes[62].x)
        h4_y = self.nodes[62].y + ratio_4 * (self.nodes[65].y - self.nodes[62].y)

        self.hospitals[4] = HospitalOnRoad(
            hospital_id=4,
            name="Bệnh viện Cấp cứu Ngoại ô Tây Nam (BV 4)",
            from_node=62,
            to_node=65,
            road_length=edge_62_65.length,
            distance_from_start=dist_4,
            coord=(round(h4_x, 2), round(h4_y, 2)),
            handover_time=4.0,
        )

    # ========================================================================
    # 4. BIẾN THIÊN ĐỘ TẮC CÁC ĐƯỜNG THEO THỜI GIAN CHẠY
    # ========================================================================
    def randomize_traffic(self, initial: bool = False) -> None:
        """Khởi tạo độ tắc ngẫu nhiên in {1, 2, 3} cho tất cả các cung đường."""
        probs = [0.60, 0.30, 0.10] if initial else [0.45, 0.35, 0.20]
        for edge in self.edges.values():
            level = int(self.rng.choice([CONGESTION_CLEAR, CONGESTION_MODERATE, CONGESTION_HEAVY], p=probs))
            edge.congestion = level

    def step_traffic(self, rush_hour: Optional[bool] = None) -> Dict[Tuple[int, int], int]:
        """
        Cập nhật độ tắc đường ngẫu nhiên theo chuỗi Markov và thống kê biến thiên.
        Thống kê 6 kiểu chuyển đổi phục vụ ghi log theo yêu cầu:
        - Từ tắc -> bình thường (3 -> 2)
        - Từ thông thoáng -> bình thường (1 -> 2)
        - Từ bình thường -> tắc (2 -> 3)
        - Từ bình thường -> thông thoáng (2 -> 1)
        - Từ tắc -> thông thoáng (3 -> 1)
        - Từ thông thoáng -> tắc (1 -> 3)
        """
        self.time_step += 1
        if rush_hour is None:
            rush_hour = 10 <= (self.time_step % 40) <= 25

        transitions: Dict[str, List[str]] = {
            "tac_to_binh_thuong": [],
            "thong_thoang_to_binh_thuong": [],
            "binh_thuong_to_tac": [],
            "binh_thuong_to_thong_thoang": [],
            "tac_to_thong_thoang": [],
            "thong_thoang_to_tac": [],
        }

        current_traffic: Dict[Tuple[int, int], int] = {}
        for key, edge in self.edges.items():
            old_lvl = edge.congestion
            new_lvl = edge.update_congestion_markov(self.rng, rush_hour=rush_hour)
            current_traffic[key] = new_lvl

            u, v = key
            road_lbl = f"N{u}->N{v}"
            if old_lvl == 3 and new_lvl == 2:
                transitions["tac_to_binh_thuong"].append(road_lbl)
            elif old_lvl == 1 and new_lvl == 2:
                transitions["thong_thoang_to_binh_thuong"].append(road_lbl)
            elif old_lvl == 2 and new_lvl == 3:
                transitions["binh_thuong_to_tac"].append(road_lbl)
            elif old_lvl == 2 and new_lvl == 1:
                transitions["binh_thuong_to_thong_thoang"].append(road_lbl)
            elif old_lvl == 3 and new_lvl == 1:
                transitions["tac_to_thong_thoang"].append(road_lbl)
            elif old_lvl == 1 and new_lvl == 3:
                transitions["thong_thoang_to_tac"].append(road_lbl)

        self.last_traffic_transitions = transitions
        return current_traffic

    def set_edge_congestion(self, u: int, v: int, level: int) -> None:
        """Đặt độ tắc đường thủ công cho hướng (u -> v)."""
        if (u, v) not in self.edges:
            raise ValueError(f"Không tồn tại cung đường có hướng từ {u} đến {v}.")
        self.edges[(u, v)].set_congestion(level)

    # ========================================================================
    # 5. TÌM ĐƯỜNG TỐI ƯU ĐẾN BỆNH VIỆN (DIJKSTRA)
    # ========================================================================
    def shortest_path(
        self,
        start_node: int,
        target_node: int,
        by_travel_time: bool = True,
    ) -> Tuple[List[int], float]:
        """Tìm đường đi ngắn nhất giữa 2 nút bằng Dijkstra."""
        if start_node == target_node:
            return [start_node], 0.0

        dist: Dict[int, float] = {i: float("inf") for i in range(1, self.num_nodes + 1)}
        parent: Dict[int, Optional[int]] = {i: None for i in range(1, self.num_nodes + 1)}
        dist[start_node] = 0.0

        pq: List[Tuple[float, int]] = [(0.0, start_node)]

        while pq:
            d, u = heapq.heappop(pq)
            if u == target_node:
                break
            if d > dist[u]:
                continue

            for edge in self.adj[u]:
                v = edge.to_node
                cost = edge.travel_time_minutes if by_travel_time else edge.length
                nd = d + cost
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))

        if dist[target_node] == float("inf"):
            return [], float("inf")

        path: List[int] = []
        curr: Optional[int] = target_node
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()
        return path, round(dist[target_node], 3)

    def route_to_hospital(
        self,
        start_node: int,
        hospital_id: int,
        by_travel_time: bool = True,
    ) -> Dict[str, any]:
        """
        Tìm lộ trình từ start_node đến một trong 4 Bệnh viện.
        Bệnh viện nằm trên DUY NHẤT 1 cung đường có hướng (from_node -> to_node).
        Xe cứu thương di chuyển đến điểm đi (from_node) của cung đó,
        sau đó đi thêm một đoạn bằng distance_from_start theo hướng đó để đến bệnh viện.
        """
        hosp = self.hospitals.get(hospital_id)
        if hosp is None:
            raise ValueError(f"Không tìm thấy bệnh viện có ID = {hospital_id}")

        start_pt = hosp.from_node
        path_to_start_node, cost_to_start = self.shortest_path(start_node, start_pt, by_travel_time=by_travel_time)

        edge = self.edges[(hosp.from_node, hosp.to_node)]

        if by_travel_time:
            time_entry = (hosp.distance_from_start / edge.speed) * 60.0
            total_cost = cost_to_start + time_entry
        else:
            total_cost = cost_to_start + hosp.distance_from_start

        return {
            "hospital_id": hospital_id,
            "hospital_name": hosp.name,
            "on_directed_edge": f"{hosp.from_node} -> {hosp.to_node}",
            "start_node": hosp.from_node,
            "path_to_start_node": path_to_start_node,
            "distance_from_start_node": hosp.distance_from_start,
            "total_cost": round(total_cost, 3),
            "cost_unit": "phút" if by_travel_time else "km",
        }

    def route_from_road_to_road(
        self,
        start_edge: Tuple[int, int],
        start_dist: float,
        dest_edge: Tuple[int, int],
        dest_dist: float,
        by_travel_time: bool = True,
    ) -> Dict[str, Any]:
        """
        Tìm lộ trình chi tiết từ một vị trí trên cung đường start_edge
        đến một vị trí trên cung đường dest_edge.
        Các phương án di chuyển tuân thủ yêu cầu:
        - Đi thẳng trên cùng cung đường (nếu start_edge == dest_edge và start_dist <= dest_dist).
        - Rẽ sang đường mới tại các nút giao.
        - Tới cuối đường quay đầu chọn tuyến đường khác (U-turn qua nút v_start).
        """
        u_start, v_start = start_edge
        u_dest, v_dest = dest_edge

        e_start = self.edges[start_edge]
        e_dest = self.edges[dest_edge]

        # Trường hợp 1: Cùng trên 1 cung đường và điểm đến ở phía trước
        if start_edge == dest_edge and start_dist <= dest_dist:
            dist_leg = dest_dist - start_dist
            speed = e_start.speed
            time_leg = (dist_leg / speed) * 60.0
            return {
                "edges_path": [start_edge],
                "total_time_minutes": round(time_leg, 3),
                "total_distance_km": round(dist_leg, 3),
                "start_edge": start_edge,
                "dest_edge": dest_edge,
                "dest_dist": dest_dist,
            }

        # Trường hợp 2: Đi tiếp tới cuối cung đường hiện tại (nút v_start),
        # sau đó dùng Dijkstra từ v_start tới u_dest, rồi đi dest_dist trên dest_edge
        rem_start_dist = max(0.0, e_start.length - start_dist)
        rem_start_time = (rem_start_dist / e_start.speed) * 60.0

        node_path, _ = self.shortest_path(v_start, u_dest, by_travel_time=by_travel_time)

        intermediate_edges: List[Tuple[int, int]] = []
        inter_dist = 0.0
        inter_time = 0.0
        for i in range(len(node_path) - 1):
            ek = (node_path[i], node_path[i + 1])
            intermediate_edges.append(ek)
            ed = self.edges[ek]
            inter_dist += ed.length
            inter_time += ed.travel_time_minutes

        # Đoạn cuối trên dest_edge (từ nút u_dest đến dest_dist)
        dest_time = (dest_dist / e_dest.speed) * 60.0

        edges_path = [start_edge] + intermediate_edges + [dest_edge]
        total_time = rem_start_time + inter_time + dest_time
        total_dist = rem_start_dist + inter_dist + dest_dist

        return {
            "edges_path": edges_path,
            "total_time_minutes": round(total_time, 3),
            "total_distance_km": round(total_dist, 3),
            "start_edge": start_edge,
            "dest_edge": dest_edge,
            "dest_dist": dest_dist,
        }

    # ========================================================================
    # 6. TỔNG KẾT & THỐNG KÊ
    # ========================================================================
    def get_summary(self) -> Dict[str, any]:
        type_counts: Dict[str, int] = {}
        for node in self.nodes.values():
            t_name = node.intersection_type.value
            type_counts[t_name] = type_counts.get(t_name, 0) + 1

        congestions = [e.congestion for e in self.edges.values()]
        lengths = [e.length for e in self.edges.values()]

        return {
            "num_nodes": self.num_nodes,
            "num_roads_undirected": len(self.road_pairs),
            "num_directed_edges": len(self.edges),
            "length_stats": {
                "min_km": min(lengths),
                "max_km": max(lengths),
                "avg_km": round(float(np.mean(lengths)), 2),
            },
            "intersection_types": type_counts,
            "hospitals": [
                {
                    "id": h.hospital_id,
                    "name": h.name,
                    "directed_road": f"{h.from_node} -> {h.to_node}",
                    "from_node": h.from_node,
                    "to_node": h.to_node,
                    "distance_from_start_km": h.distance_from_start,
                    "distance_to_end_km": h.distance_to_end,
                    "total_road_length_km": h.road_length,
                    "coord": h.coord,
                }
                for h in self.hospitals.values()
            ],
            "traffic_distribution": {
                "clear_level_1": congestions.count(CONGESTION_CLEAR),
                "moderate_level_2": congestions.count(CONGESTION_MODERATE),
                "heavy_level_3": congestions.count(CONGESTION_HEAVY),
            },
        }
