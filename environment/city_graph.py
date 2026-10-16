"""
Mô hình hóa Đồ thị Mạng lưới Giao thông Thành phố Thông minh G = (V, E).
- |V| = 15 nút giao thông (đánh số 1..15)
- 3 Trạm trực xe cứu thương: Nút 1, Nút 8, Nút 15
- 2 Bệnh viện đa khoa: Nút 3, Nút 13
- 5 Tuyến đường huyết mạch (Arterial Roads) có trạng thái lưu thông động c_e in {0, 1, 2}
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    ALPHA_CONGESTION,
    HOSPITAL_NODES,
    NOISE_STD,
    NUM_ARTERIAL_ROADS,
    NUM_NODES,
    STATION_NODES,
    TRAFFIC_BUSY,
    TRAFFIC_CLEAR,
    TRAFFIC_CONGESTED,
    V_MAX,
)


@dataclass(frozen=True)
class NodeInfo:
    """Thông tin chi tiết của một nút giao thông trong đô thị."""

    node_id: int              # 1..15
    name: str                 # Tên khu vực / nút giao
    coord: Tuple[float, float]# Tọa độ (x, y) trên bản đồ đô thị (km)
    zone_type: str            # "station", "hospital", "commercial", "residential"
    demand_weight: float      # Trọng số mật độ dân cư (dùng cho cường độ Poisson lambda(v, t))


@dataclass(frozen=True)
class EdgeInfo:
    """Thông tin một cung đường hai chiều trong mạng lưới đô thị."""

    u: int                    # Nút đầu (1..15)
    v: int                    # Nút cuối (1..15)
    base_length: float        # Chiều dài chuẩn (đơn vị chuẩn hóa ~ 1 bước di chuyển khi thông thoáng)
    is_arterial: bool = False # Có phải tuyến đường huyết mạch không
    arterial_idx: int = -1    # Chỉ số tuyến huyết mạch (0..4 nếu is_arterial=True)
    name: str = ""            # Tên tuyến đường


class SmartCityGraph:
    """
    Đồ thị mạng lưới giao thông đô thị 15 nút và 5 tuyến đường huyết mạch.

    Sơ đồ bố trí 15 nút (3 hàng x 5 cột + đường chéo trung tâm):
        [1:Trạm ĐB] --- [2:Khu Dân Cư B] ===(HM1)=== [3:BỆNH VIỆN BẮC] --- [4:Khu Đô Thị B] --- [5:Cửa Ngõ ĐB]
             |                 |                           || (HM2)               |                    |
        [6:Khu CN Tây] -- [7:TT Thương Mại T] ==(HM3)== [8:TRẠM TRUNG TÂM] ==(HM4)== [9:TT Thương Mại Đ] -- [10:Khu Dân Cư Đ]
             |                 |                           || (HM5)               |                    |
        [11:Khu Dân Cư TN]-[12:Ngã Tư Nam] ---------- [13:BỆNH VIỆN NAM] --- [14:Khu Dân Cư ĐN] -- [15:Trạm ĐN]

    Chú thích: === (HM1..HM5) là 5 tuyến đường huyết mạch (Arterial Roads).
    """

    def __init__(
        self,
        v_max: float = V_MAX,
        alpha_congestion: float = ALPHA_CONGESTION,
        noise_std: float = NOISE_STD,
        noise_distribution: str = "gaussian",
    ) -> None:
        self.num_nodes = NUM_NODES
        self.num_arterial_roads = NUM_ARTERIAL_ROADS
        self.v_max = v_max
        self.alpha_congestion = alpha_congestion
        self.noise_std = noise_std
        self.noise_distribution = noise_distribution

        self.stations: List[int] = list(STATION_NODES)
        self.hospitals: List[int] = list(HOSPITAL_NODES)

        # Khởi tạo 15 nút đô thị
        self.nodes: Dict[int, NodeInfo] = self._build_nodes()

        # Khởi tạo các cung đường và 5 tuyến huyết mạch
        self.edges: List[EdgeInfo] = []
        self.arterial_edges: List[EdgeInfo] = []
        self.adj: Dict[int, List[EdgeInfo]] = {i: [] for i in range(1, self.num_nodes + 1)}
        self._edge_lookup: Dict[Tuple[int, int], EdgeInfo] = {}
        self._build_edges()

        # Trạng thái lưu thông của 5 tuyến đường huyết mạch: X_traffic = [c_1, c_2, c_3, c_4, c_5]
        self.arterial_traffic: np.ndarray = np.zeros(self.num_arterial_roads, dtype=np.int64)

        # Tiền tính toán ma trận khoảng cách tĩnh (số bước tối thiểu khi đường thông thoáng)
        self.static_dist_matrix: np.ndarray = self._compute_all_pairs_static_distance()

    def _build_nodes(self) -> Dict[int, NodeInfo]:
        """Định nghĩa 15 nút giao thông với tọa độ và mật độ dân cư đặc trưng."""
        raw_nodes = [
            # Hàng Bắc (1..5)
            NodeInfo(1, "Trạm Cứu Hộ Tây Bắc", (0.0, 2.0), "station", 0.05),
            NodeInfo(2, "Khu Dân Cư Bắc", (1.0, 2.0), "residential", 0.08),
            NodeInfo(3, "Bệnh Viện Đa Khoa Bắc", (2.0, 2.0), "hospital", 0.04),
            NodeInfo(4, "Khu Đô Thị Mới Bắc", (3.0, 2.0), "residential", 0.07),
            NodeInfo(5, "Khu Công Nghệ Cao Đông Bắc", (4.0, 2.0), "commercial", 0.05),
            # Hàng Trung tâm (6..10)
            NodeInfo(6, "Khu Công Nghiệp Tây", (0.0, 1.0), "commercial", 0.06),
            NodeInfo(7, "Trung Tâm Thương Mại Tây", (1.0, 1.0), "commercial", 0.11),
            NodeInfo(8, "Trạm Trung Tâm Thành Phố", (2.0, 1.0), "station", 0.10),
            NodeInfo(9, "Trung Tâm Thương Mại Đông", (3.0, 1.0), "commercial", 0.11),
            NodeInfo(10, "Khu Dân Cư Đông", (4.0, 1.0), "residential", 0.07),
            # Hàng Nam (11..15)
            NodeInfo(11, "Khu Dân Cư Tây Nam", (0.0, 0.0), "residential", 0.06),
            NodeInfo(12, "Ngã Tư Dân Cư Nam", (1.0, 0.0), "residential", 0.07),
            NodeInfo(13, "Bệnh Viện Đa Khoa Nam", (2.0, 0.0), "hospital", 0.04),
            NodeInfo(14, "Khu Dân Cư Đông Nam", (3.0, 0.0), "residential", 0.05),
            NodeInfo(15, "Trạm Cứu Hộ Đông Nam", (4.0, 0.0), "station", 0.04),
        ]
        # Chuẩn hóa trọng số cầu cấp cứu để tổng bằng 1.0
        total_w = sum(n.demand_weight for n in raw_nodes)
        nodes_dict: Dict[int, NodeInfo] = {}
        for n in raw_nodes:
            nodes_dict[n.node_id] = NodeInfo(
                node_id=n.node_id,
                name=n.name,
                coord=n.coord,
                zone_type=n.zone_type,
                demand_weight=n.demand_weight / total_w,
            )
        return nodes_dict

    def _build_edges(self) -> None:
        """
        Thiết lập các cung đường nối 15 nút, trong đó đánh dấu rõ 5 tuyến đường huyết mạch.
        5 Tuyến đường huyết mạch (Arterial Roads):
            0: (2, 3)  - Đường huyết mạch Cửa ngõ Bắc -> Bệnh viện Bắc
            1: (3, 8)  - Đại lộ Bắc - Trung tâm (Bệnh viện Bắc <-> Trạm Trung tâm)
            2: (7, 8)  - Trục huyết mạch Thương mại Tây <-> Trung tâm
            3: (8, 9)  - Trục huyết mạch Trung tâm <-> Thương mại Đông
            4: (8, 13) - Đại lộ Trung tâm - Nam (Trạm Trung tâm <-> Bệnh viện Nam)
        """
        arterial_specs: Dict[Tuple[int, int], Tuple[int, str]] = {
            (2, 3): (0, "Tuyến huyết mạch 1 (KDC Bắc - BV Bắc)"),
            (3, 8): (1, "Tuyến huyết mạch 2 (Đại lộ BV Bắc - Trung tâm)"),
            (7, 8): (2, "Tuyến huyết mạch 3 (Trục TM Tây - Trung tâm)"),
            (8, 9): (3, "Tuyến huyết mạch 4 (Trục Trung tâm - TM Đông)"),
            (8, 13): (4, "Tuyến huyết mạch 5 (Đại lộ Trung tâm - BV Nam)"),
        }

        # Các cạnh cơ sở của lưới 3x5 + 2 đường nhánh chéo giúp xe có lộ trình thay thế khi tránh kẹt xe
        raw_pairs = [
            # Hàng ngang 1 (Bắc)
            (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0),
            # Hàng ngang 2 (Trung tâm)
            (6, 7, 1.0), (7, 8, 1.0), (8, 9, 1.0), (9, 10, 1.0),
            # Hàng ngang 3 (Nam)
            (11, 12, 1.0), (12, 13, 1.0), (13, 14, 1.0), (14, 15, 1.0),
            # Cột dọc
            (1, 6, 1.0), (6, 11, 1.0),
            (2, 7, 1.0), (7, 12, 1.0),
            (3, 8, 1.0), (8, 13, 1.0),
            (4, 9, 1.0), (9, 14, 1.0),
            (5, 10, 1.0), (10, 15, 1.0),
            # Đường nhánh vành đai / chéo giúp định tuyến tránh kẹt xe trên trục huyết mạch
            (2, 8, 1.35), (4, 8, 1.35), (7, 13, 1.35), (9, 13, 1.35),
        ]

        arterial_list: List[Optional[EdgeInfo]] = [None] * self.num_arterial_roads

        for u, v, length in raw_pairs:
            pair_key = (min(u, v), max(u, v))
            if pair_key in arterial_specs:
                art_idx, art_name = arterial_specs[pair_key]
                edge = EdgeInfo(
                    u=pair_key[0],
                    v=pair_key[1],
                    base_length=length,
                    is_arterial=True,
                    arterial_idx=art_idx,
                    name=art_name,
                )
                arterial_list[art_idx] = edge
            else:
                edge = EdgeInfo(
                    u=pair_key[0],
                    v=pair_key[1],
                    base_length=length,
                    is_arterial=False,
                    arterial_idx=-1,
                    name=f"Đường nội đô ({pair_key[0]}-{pair_key[1]})",
                )

            self.edges.append(edge)
            self._edge_lookup[(u, v)] = edge
            self._edge_lookup[(v, u)] = edge
            self.adj[u].append(edge)
            self.adj[v].append(edge)

        self.arterial_edges = [e for e in arterial_list if e is not None]
        assert len(self.arterial_edges) == self.num_arterial_roads, (
            f"Expected {self.num_arterial_roads} arterial roads, got {len(self.arterial_edges)}"
        )

    def reset_traffic(self, rng: np.random.Generator, initial_rush_hour: bool = False) -> np.ndarray:
        """Khởi tạo trạng thái giao thông cho 5 tuyến đường huyết mạch."""
        if initial_rush_hour:
            probs = [0.25, 0.45, 0.30]
        else:
            probs = [0.60, 0.30, 0.10]

        self.arterial_traffic = rng.choice(
            [TRAFFIC_CLEAR, TRAFFIC_BUSY, TRAFFIC_CONGESTED],
            size=self.num_arterial_roads,
            p=probs,
        ).astype(np.int64)
        return self.arterial_traffic.copy()

    def step_traffic(self, rng: np.random.Generator, step_count: int) -> np.ndarray:
        """
        Cập nhật trạng thái giao thông của 5 tuyến đường huyết mạch theo chuỗi Markov thời gian thực.
        Mô phỏng chu kỳ giờ bình thường và giờ cao điểm (Rush Hour).
        """
        is_rush_hour = (20 <= (step_count % 50) <= 38)

        # Ma trận chuyển trạng thái Markov P(c_{t+1} | c_t)
        if is_rush_hour:
            transition_matrix = np.array([
                [0.50, 0.35, 0.15],  # Từ Thông thoáng (0)
                [0.20, 0.50, 0.30],  # Từ Đông (1)
                [0.10, 0.35, 0.55],  # Từ Ùn tắc (2)
            ])
        else:
            transition_matrix = np.array([
                [0.75, 0.20, 0.05],  # Từ Thông thoáng (0)
                [0.40, 0.45, 0.15],  # Từ Đông (1)
                [0.25, 0.50, 0.25],  # Từ Ùn tắc (2)
            ])

        for i in range(self.num_arterial_roads):
            curr_state = int(self.arterial_traffic[i])
            self.arterial_traffic[i] = int(
                rng.choice([TRAFFIC_CLEAR, TRAFFIC_BUSY, TRAFFIC_CONGESTED], p=transition_matrix[curr_state])
            )

        return self.arterial_traffic.copy()

    def get_edge(self, u: int, v: int) -> Optional[EdgeInfo]:
        """Lấy thông tin cung đường giữa 2 nút kề nhau u và v."""
        return self._edge_lookup.get((u, v))

    def get_neighbors(self, node_id: int) -> List[int]:
        """Trả về danh sách các nút kề của node_id (đã sắp xếp tăng dần)."""
        neighbors = []
        for edge in self.adj[node_id]:
            other = edge.v if edge.u == node_id else edge.u
            neighbors.append(other)
        return sorted(neighbors)

    def get_edge_congestion(self, u: int, v: int) -> int:
        """
        Trả về mức độ lưu thông c_e in {0, 1, 2} trên cạnh (u, v).
        - Nếu là 1 trong 5 tuyến đường huyết mạch: lấy từ vector X_traffic.
        - Nếu là đường nội đô thường: mặc định thông thoáng (0).
        """
        edge = self.get_edge(u, v)
        if edge is None:
            raise ValueError(f"Không tồn tại cạnh nối giữa nút {u} và nút {v}.")
        if edge.is_arterial:
            return int(self.arterial_traffic[edge.arterial_idx])
        return TRAFFIC_CLEAR

    def compute_edge_velocity(
        self,
        u: int,
        v: int,
        rng: Optional[np.random.Generator] = None,
        deterministic: bool = False,
    ) -> float:
        """
        Công thức động học giao thông (Mục 4.2.4):
            v(e, t) = v_max * (1 - alpha * cong(e, t))
        với độ trễ ngẫu nhiên tuân theo phân phối Gaussian hoặc Log-Normal.
        """
        cong = self.get_edge_congestion(u, v)
        base_velocity = self.v_max * (1.0 - self.alpha_congestion * float(cong))

        if deterministic or rng is None or self.noise_distribution == "none":
            return max(0.15, base_velocity)

        if self.noise_distribution == "lognormal":
            # Độ trễ nhân tính theo Log-Normal có kỳ vọng xấp xỉ 1
            sigma = self.noise_std
            factor = float(rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma))
            vel = base_velocity * factor
        else:
            # Nhiễu cộng tính theo Gaussian
            noise = float(rng.normal(loc=0.0, scale=self.noise_std))
            vel = base_velocity + noise

        return float(np.clip(vel, 0.15, self.v_max * 1.2))

    def compute_edge_travel_time(
        self,
        u: int,
        v: int,
        rng: Optional[np.random.Generator] = None,
        deterministic: bool = True,
    ) -> float:
        """
        Thời gian di chuyển qua cạnh (u, v) = chiều_dài_cạnh / vận_tốc_thực_tế v(e, t).
        """
        edge = self.get_edge(u, v)
        if edge is None:
            return float("inf")
        vel = self.compute_edge_velocity(u, v, rng=rng, deterministic=deterministic)
        return edge.base_length / vel

    def _compute_all_pairs_static_distance(self) -> np.ndarray:
        """Tính khoảng cách ngắn nhất tĩnh giữa mọi cặp nút (1..15) bằng thuật toán Dijkstra."""
        dist = np.full((self.num_nodes + 1, self.num_nodes + 1), np.inf, dtype=np.float64)
        for src in range(1, self.num_nodes + 1):
            dist[src, src] = 0.0
            pq: List[Tuple[float, int]] = [(0.0, src)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[src, u]:
                    continue
                for edge in self.adj[u]:
                    v = edge.v if edge.u == u else edge.u
                    nd = d + edge.base_length
                    if nd < dist[src, v]:
                        dist[src, v] = nd
                        heapq.heappush(pq, (nd, v))
        return dist

    def get_static_distance(self, u: int, v: int) -> float:
        """Khoảng cách đồ thị tĩnh ngắn nhất giữa nút u và nút v."""
        return float(self.static_dist_matrix[u, v])

    def shortest_path(
        self,
        src: int,
        dst: int,
        consider_traffic: bool = True,
    ) -> Tuple[List[int], float]:
        """
        Tìm lộ trình tối ưu từ nút src đến nút dst bằng thuật toán Dijkstra.
        - Nếu consider_traffic=True: tối ưu hóa thời gian di chuyển theo trạng thái kẹt xe hiện tại.
        - Nếu consider_traffic=False: tối ưu hóa khoảng cách hình học tĩnh.
        """
        if src == dst:
            return [src], 0.0

        dist: Dict[int, float] = {i: float("inf") for i in range(1, self.num_nodes + 1)}
        parent: Dict[int, Optional[int]] = {i: None for i in range(1, self.num_nodes + 1)}
        dist[src] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, src)]

        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist[u]:
                continue
            for edge in self.adj[u]:
                v = edge.v if edge.u == u else edge.u
                cost = (
                    self.compute_edge_travel_time(u, v, deterministic=True)
                    if consider_traffic
                    else edge.base_length
                )
                nd = d + cost
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))

        path: List[int] = []
        curr: Optional[int] = dst
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()
        return path, dist[dst]

    def get_ranked_next_hops(
        self,
        src: int,
        dst: int,
        consider_traffic: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        Xếp hạng các nút kề u của src theo tổng chi phí dự kiến từ src -> u -> dst.
        Hỗ trợ trực tiếp cho hành động a_route(k, u) định tuyến tránh kẹt xe.
        """
        if src == dst:
            return [(src, 0.0)]

        candidates: List[Tuple[int, float]] = []
        for neighbor in self.get_neighbors(src):
            step_cost = (
                self.compute_edge_travel_time(src, neighbor, deterministic=True)
                if consider_traffic
                else self.get_edge(src, neighbor).base_length  # type: ignore[union-attr]
            )
            _, rem_cost = self.shortest_path(neighbor, dst, consider_traffic=consider_traffic)
            candidates.append((neighbor, step_cost + rem_cost))

        candidates.sort(key=lambda item: item[1])
        return candidates

    def nearest_hospital(self, node_id: int, consider_traffic: bool = True) -> Tuple[int, float]:
        """Tìm bệnh viện có thời gian di chuyển ngắn nhất từ node_id."""
        best_hosp = self.hospitals[0]
        best_cost = float("inf")
        for hosp in self.hospitals:
            _, cost = self.shortest_path(node_id, hosp, consider_traffic=consider_traffic)
            if cost < best_cost:
                best_cost = cost
                best_hosp = hosp
        return best_hosp, best_cost
