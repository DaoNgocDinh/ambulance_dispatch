"""Adapter chuyển môi trường cứu thương sang giao diện state/action Q-learning.

Mọi hiểu biết riêng về xe, cuộc gọi, bản đồ và API môi trường được giữ ở đây.
Nhờ vậy, lõi trong q_learning.py không cần biết cấu trúc của dự án cứu thương.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, List, Tuple

from dispatch_env import (
    AmbulanceDispatchEnv,
    AmbulanceStatus,
    DISTANCE_PENALTY_PER_KM,
    REWARD_CONFIG,
    SEVERITY_CONFIG,
)
from map_env.edge import SPEEDS_CENTER, SPEEDS_OUTER


class AmbulanceDispatchAdapter:
    """Biến trạng thái môi trường thành state key và chuyển tiếp action tới env."""

    @staticmethod
    def _bucket(value: float, limits: Tuple[float, ...]) -> int:
        """Đổi số liên tục thành chỉ số khoảng để state có số giá trị hữu hạn.

        Ví dụ limits=(5, 15, 30): giá trị 4 thuộc khoảng 0, 10 thuộc khoảng 1,
        còn giá trị lớn hơn 30 thuộc khoảng cuối cùng (chỉ số 3).
        """
        for index, limit in enumerate(limits):
            if value <= limit:
                return index
        return len(limits)

    @staticmethod
    def _environment_signature(env: AmbulanceDispatchEnv) -> str:
        """Tạo fingerprint để tách Q-table của các cấu hình môi trường khác nhau.

        Fingerprint bao gồm bản đồ, bệnh viện, tập action và tham số ảnh hưởng
        đến luật chuyển trạng thái/phần thưởng. Đây không phải cơ chế sửa hoặc
        kiểm tra tính hợp lệ của bản đồ.
        """
        city_map = env.city_map
        # Lưu thông tin nút, tọa độ và kết nối để phát hiện thay đổi topology.
        nodes = [
            (node_id, round(node.x, 6), round(node.y, 6), tuple(sorted(node.neighbors)))
            for node_id, node in sorted(city_map.nodes.items())
        ]
        # Cung đường có hướng: chiều đi, chiều đến, chiều dài và vùng trung tâm.
        edges = [
            (
                edge_id,
                edge.from_node,
                edge.to_node,
                round(edge.length, 6),
                edge.is_center,
            )
            for edge_id, edge in sorted(city_map.edges_by_id.items())
        ]
        # Vị trí và thời gian bàn giao tại bệnh viện cũng ảnh hưởng quá trình mô phỏng.
        hospitals = [
            (
                hospital_id,
                hospital.directed_edge_key,
                round(hospital.distance_from_start, 6),
                round(hospital.handover_time, 6),
            )
            for hospital_id, hospital in sorted(city_map.hospitals.items())
        ]
        # Action space phụ thuộc số xe/số slot và quy tắc ghép xe với cuộc gọi.
        action_space = [tuple(action) for action in env.potential_actions]
        # Ghép cấu hình thành dữ liệu ổn định trước khi băm thành chuỗi ngắn.
        configuration = (
            city_map.num_nodes,
            len(city_map.road_pairs),
            nodes,
            edges,
            hospitals,
            action_space,
            env.num_ambulances,
            env.max_pending_calls,
            env.max_time,
            env.max_overtime,
            env.poisson_lambda,
            sorted((key, tuple(value)) for key, value in SEVERITY_CONFIG.items()),
            sorted((severity, sorted(config.items())) for severity, config in REWARD_CONFIG.items()),
            DISTANCE_PENALTY_PER_KM,
            sorted(SPEEDS_CENTER.items()),
            sorted(SPEEDS_OUTER.items()),
        )
        serialized = json.dumps(configuration, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def encode_state(self, env: AmbulanceDispatchEnv) -> str:
        """Mã hóa trạng thái hiện tại thành khóa chuỗi dùng trong Q-table.

        Các giá trị như thời gian chờ được chia khoảng để giảm số lượng state.
        Fingerprint được đặt đầu state để không dùng nhầm Q-value giữa cấu hình.
        """
        statuses = list(AmbulanceStatus)
        ambulance_features = []

        # Tóm tắt tình trạng từng xe: trạng thái, vùng đường, tắc nghẽn,
        # mức độ cuộc gọi đang xử lý và thời gian xử lý còn lại.
        for ambulance_id in sorted(env.ambulances):
            ambulance = env.ambulances[ambulance_id]
            edge = env.city_map.edges[ambulance.current_edge]
            current_call = env.calls.get(ambulance.current_call_id)
            target_zone = 0
            if ambulance.target_edge is not None:
                target_zone = int(env.city_map.edges[ambulance.target_edge].is_center) + 1

            ambulance_features.append((
                statuses.index(ambulance.status),
                int(edge.is_center),
                edge.congestion,
                0 if current_call is None else current_call.severity,
                target_zone,
                self._bucket(ambulance.scene_time_remaining, (1.0, 4.0, 8.0)),
                self._bucket(ambulance.hospital_time_remaining, (1.0, 3.0, 5.0)),
            ))

        # Môi trường sắp xếp các slot; giữ nguyên thứ tự để action slot khớp state.
        slots = env.get_call_slots()
        call_features = []
        # Tóm tắt mức độ, thời gian chờ/còn đến deadline và vị trí từng cuộc gọi.
        for call in slots:
            if call is None:
                call_features.append(None)
                continue

            edge = env.city_map.edges[call.edge]
            call_features.append((
                call.severity,
                self._bucket(call.wait_time, (5.0, 15.0, 30.0)),
                self._bucket(call.deadline - env.current_minute, (5.0, 15.0, 30.0)),
                int(edge.is_center),
                edge.congestion,
            ))

        # Khoảng cách đường thẳng là đặc trưng gần đúng giúp so sánh xe-call.
        # Môi trường vẫn dùng thuật toán định tuyến của nó để tính đường thực tế.
        proximity = []
        for ambulance_id in sorted(env.ambulances):
            ambulance = env.ambulances[ambulance_id]
            row = []
            for call in slots:
                if call is None or ambulance.status != AmbulanceStatus.IDLE:
                    row.append(-1)
                    continue
                x1, y1 = ambulance.get_coord(env.city_map)
                x2, y2 = call.coord
                distance = math.hypot(x1 - x2, y1 - y2)
                row.append(self._bucket(distance, (5.0, 10.0, 20.0)))
            proximity.append(tuple(row))

        # Gộp tình trạng tắc đường thành tỷ lệ đường ùn tắc ở trung tâm/ngoại vi.
        traffic_features = []
        for is_center in (True, False):
            zone_edges = [
                edge for edge in env.city_map.edges.values()
                if edge.is_center == is_center
            ]
            heavy_ratio = sum(edge.congestion == 3 for edge in zone_edges) / max(1, len(zone_edges))
            traffic_features.append(self._bucket(heavy_ratio, (0.2, 0.5)))

        # Gom thời gian thành khoảng 30 phút và đánh dấu đã qua thời điểm ngừng sinh call.
        time_state = (
            min(env.current_minute // 30, (env.max_time + env.max_overtime) // 30),
            int(env.current_minute >= env.max_time),
        )
        # Tạo tuple có thứ tự ổn định rồi chuyển thành chuỗi JSON làm khóa Q-table.
        state = (
            self._environment_signature(env),
            time_state,
            tuple(ambulance_features),
            tuple(call_features),
            tuple(proximity),
            tuple(traffic_features),
        )
        return json.dumps(state, separators=(",", ":"))

    @staticmethod
    def valid_actions(env: AmbulanceDispatchEnv) -> List[int]:
        """Lấy các action ID được phép ở state hiện tại từ action mask môi trường."""
        return env.get_valid_action_ids()

    @staticmethod
    def step(env: AmbulanceDispatchEnv, action_id: int) -> Tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Gửi action ID tới API môi trường; môi trường tự thực hiện luật mô phỏng."""
        return env.step(action=action_id)
