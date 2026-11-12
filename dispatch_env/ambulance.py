"""
Định nghĩa thực thể Xe cứu thương (Ambulance).
Mô hình hóa di chuyển liên tục trong khoảng thời gian [t, t+1),
xử lý các sự kiện đến hiện trường, chăm sóc y tế, di chuyển đến bệnh viện và bàn giao bệnh nhân.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
from dispatch_env.types import AmbulanceStatus, CallStatus
from dispatch_env.call import EmergencyCall
from map_env.city_map import CityMap


@dataclass
class Ambulance:
    """
    Thực thể Xe cứu thương.
    - ambulance_id: Mã định danh xe (1, 2, 3, 4).
    - current_edge: Cung đường có hướng hiện tại (u, v).
    - distance_from_start: Vị trí hiện tại trên cung đường tính từ điểm đi u (km).
    - status: Trạng thái xe (Rảnh, Đang đến cuộc gọi, Xử lý tại cuộc gọi, Đang tới bệnh viện, Xử lý tại bệnh viện).
    - current_call_id: Mã cuộc gọi đang tiếp nhận xử lý (hoặc None nếu rảnh).
    - target_type: Loại đích đến ('call', 'hospital' hoặc None).
    - target_id: ID của đích đến (call_id hoặc hospital_id).
    - target_edge: Cung đường chứa đích đến (u_dest, v_dest).
    - target_dist: Khoảng cách từ đầu cung đường đến đích (km).
    - target_name: Tên hiển thị của đích đến.
    - route_edges: Danh sách các cung đường trong lộ trình còn lại cần đi qua.
    - scene_time_remaining: Thời gian sơ cứu tại hiện trường còn lại (phút).
    - hospital_time_remaining: Thời gian bàn giao tại bệnh viện còn lại (phút).
    - assigned_hospital_id: Mã bệnh viện đang chuyển bệnh nhân tới.
    - total_distance_traveled: Tổng quãng đường xe đã di chuyển (km).
    - reroute_count: Số lần xe đã tính toán lại tuyến đường do biến động giao thông.
    - last_rerouted_this_step: Cờ báo hiệu xe vừa đổi đường trong phút vừa qua.
    """
    ambulance_id: int
    current_edge: Tuple[int, int]
    distance_from_start: float = 0.0
    status: AmbulanceStatus = AmbulanceStatus.IDLE
    current_call_id: Optional[int] = None

    target_type: Optional[str] = None
    target_id: Optional[int] = None
    target_edge: Optional[Tuple[int, int]] = None
    target_dist: Optional[float] = None
    target_name: str = ""

    route_edges: List[Tuple[int, int]] = field(default_factory=list)
    scene_time_remaining: float = 0.0
    hospital_time_remaining: float = 0.0
    assigned_hospital_id: Optional[int] = None
    total_distance_traveled: float = 0.0
    last_step_distance: float = 0.0
    reroute_count: int = 0
    last_rerouted_this_step: bool = False

    def __post_init__(self) -> None:
        if not self.route_edges:
            self.route_edges = [self.current_edge]

    def get_coord(self, city_map: CityMap) -> Tuple[float, float]:
        """Tọa độ hình học (x, y) hiện tại của xe cứu thương trên bản đồ."""
        u, v = self.current_edge
        nu = city_map.nodes[u]
        nv = city_map.nodes[v]
        edge = city_map.edges[self.current_edge]
        ratio = 0.0
        if edge.length > 0:
            ratio = min(1.0, max(0.0, self.distance_from_start / edge.length))
        x = nu.x + ratio * (nv.x - nu.x)
        y = nu.y + ratio * (nv.y - nu.y)
        return (round(float(x), 2), round(float(y), 2))

    def get_road_speed(self, city_map: CityMap) -> float:
        """Vận tốc cho phép trên cung đường hiện tại (km/h) dựa trên mức độ tắc và phân vùng."""
        edge = city_map.edges.get(self.current_edge)
        return edge.speed if edge is not None else 30.0

    def get_current_speed(self, city_map: CityMap) -> float:
        """
        Vận tốc thực tế hiện tại của xe cứu thương (km/h).
        - Khi đang di chuyển (EN_ROUTE_TO_CALL, EN_ROUTE_TO_HOSPITAL): bằng vận tốc cho phép của đường.
        - Khi đang dừng (IDLE, AT_SCENE, AT_HOSPITAL): 0.0 km/h.
        """
        if self.status in (AmbulanceStatus.EN_ROUTE_TO_CALL, AmbulanceStatus.EN_ROUTE_TO_HOSPITAL):
            return self.get_road_speed(city_map)
        return 0.0

    @property
    def targer_type(self) -> Optional[str]:
        """Bí danh cho target_type hỗ trợ trường hợp gõ theo prompt."""
        return self.target_type

    def to_dict(self, city_map: CityMap) -> Dict[str, Any]:
        """
        Xuất toàn bộ thông tin chi tiết của xe cứu thương dưới dạng dictionary,
        đáp ứng đầy đủ các trường đích đến, bệnh viện bàn giao, thời gian còn lại và cung đường.
        """
        curr_edge_id = city_map.edge_to_id.get(self.current_edge, 0)
        target_edge_id = city_map.edge_to_id.get(self.target_edge) if self.target_edge else None
        handover_time = (
            city_map.hospitals[self.assigned_hospital_id].handover_time
            if self.assigned_hospital_id and self.assigned_hospital_id in city_map.hospitals
            else 0.0
        )
        return {
            "id": self.ambulance_id,
            "ambulance_id": self.ambulance_id,
            "status": self.status.value,
            "current_edge": self.current_edge,
            "current_edge_id": curr_edge_id,
            "distance_from_start": self.distance_from_start,
            "coord": self.get_coord(city_map),
            "current_speed": self.get_current_speed(city_map),
            "road_speed": self.get_road_speed(city_map),
            "current_call_id": self.current_call_id,
            "target_type": self.target_type,
            "targer_type": self.target_type,
            "target_id": self.target_id,
            "target_edge": self.target_edge,
            "target_edge_id": target_edge_id,
            "target_dist": self.target_dist,
            "target_name": self.target_name,
            "assigned_hospital_id": self.assigned_hospital_id,
            "hospital_handover_time": handover_time,
            "scene_time_remaining": round(self.scene_time_remaining, 2),
            "hospital_time_remaining": round(self.hospital_time_remaining, 2),
            "route": self.route_edges,
            "route_edge_ids": [city_map.edge_to_id.get(e, 0) for e in self.route_edges],
            "total_distance_traveled": round(self.total_distance_traveled, 2),
        }

    def dispatch_to_call(self, call: EmergencyCall, city_map: CityMap) -> None:
        """Phân công xe cứu thương đi tiếp nhận một cuộc gọi cấp cứu mới."""
        self.status = AmbulanceStatus.EN_ROUTE_TO_CALL
        self.current_call_id = call.call_id
        self.target_type = "call"
        self.target_id = call.call_id
        self.target_edge = call.edge
        self.target_dist = call.distance_from_start
        self.target_name = f"Cuộc gọi #{call.call_id} (Mức {call.severity})"
        call.status = CallStatus.WAITING_FOR_AMBULANCE
        call.assigned_ambulance_id = self.ambulance_id

        # Tìm tuyến đường tối ưu bằng Dijkstra
        route_info = city_map.route_from_road_to_road(
            start_edge=self.current_edge,
            start_dist=self.distance_from_start,
            dest_edge=call.edge,
            dest_dist=call.distance_from_start,
            by_travel_time=True,
        )
        self.route_edges = list(route_info["edges_path"])
        self.last_rerouted_this_step = False

    def abort_mission_to_idle(self) -> None:
        """
        Khi cuộc gọi bị xoá (thất bại/hết hạn), xe lập tức về trạng thái rảnh
        và sẵn sàng tìm cuộc gọi xử lý mới.
        """
        self.status = AmbulanceStatus.IDLE
        self.current_call_id = None
        self.target_type = None
        self.target_id = None
        self.target_edge = None
        self.target_dist = None
        self.target_name = ""
        self.scene_time_remaining = 0.0
        self.hospital_time_remaining = 0.0
        self.assigned_hospital_id = None
        self.route_edges = [self.current_edge]

    def select_best_hospital_and_route(self, city_map: CityMap) -> int:
        """
        Sau khi sơ cứu xong tại hiện trường, chọn bệnh viện tối ưu nhất
        (thời gian di chuyển ngắn nhất) trong 4 bệnh viện.
        """
        best_hid: Optional[int] = None
        best_time = float("inf")
        best_route: Optional[Dict[str, Any]] = None

        for hid, hosp in city_map.hospitals.items():
            r = city_map.route_from_road_to_road(
                start_edge=self.current_edge,
                start_dist=self.distance_from_start,
                dest_edge=hosp.directed_edge_key,
                dest_dist=hosp.distance_from_start,
                by_travel_time=True,
            )
            if r["total_time_minutes"] < best_time:
                best_time = r["total_time_minutes"]
                best_hid = hid
                best_route = r

        assert best_hid is not None and best_route is not None
        hosp = city_map.hospitals[best_hid]
        self.assigned_hospital_id = best_hid
        self.target_type = "hospital"
        self.target_id = best_hid
        self.target_edge = hosp.directed_edge_key
        self.target_dist = hosp.distance_from_start
        self.target_name = hosp.name
        self.route_edges = list(best_route["edges_path"])
        self.status = AmbulanceStatus.EN_ROUTE_TO_HOSPITAL
        return best_hid

    def reroute_if_beneficial(self, city_map: CityMap) -> bool:
        """
        Tính toán lại tuyến đường bằng Dijkstra khi môi trường giao thông thay đổi.
        Nếu tìm thấy tuyến đường mới khác tuyến hiện tại, cập nhật và tăng số lần reroute.
        """
        if self.status not in (AmbulanceStatus.EN_ROUTE_TO_CALL, AmbulanceStatus.EN_ROUTE_TO_HOSPITAL):
            self.last_rerouted_this_step = False
            return False

        if not self.target_edge or self.target_dist is None:
            self.last_rerouted_this_step = False
            return False

        new_route = city_map.route_from_road_to_road(
            start_edge=self.current_edge,
            start_dist=self.distance_from_start,
            dest_edge=self.target_edge,
            dest_dist=self.target_dist,
            by_travel_time=True,
        )

        new_edges = list(new_route["edges_path"])
        if new_edges != self.route_edges:
            self.route_edges = new_edges
            self.reroute_count += 1
            self.last_rerouted_this_step = True
            return True

        self.last_rerouted_this_step = False
        return False

    def step_continuous(
        self,
        dt_minute: float,
        current_minute: float,
        city_map: CityMap,
        calls_dict: Dict[int, EmergencyCall],
    ) -> None:
        """
        Mô phỏng di chuyển liên tục trong khoảng dt_minute (thường là 1.0 phút):
        - Nếu chuyển đổi trạng thái, đến đích khi chưa hết phút:
          + Tính thời gian đi đến cuối đoạn.
          + Chuyển sang đoạn tiếp theo trong tuyến đã được chọn.
          + Dùng thời gian còn lại để đi tiếp.
          + Nếu đến vị trí bệnh nhân hoặc bệnh viện giữa phút, xử lý sự kiện tương ứng.
        """
        time_budget = dt_minute
        start_total_dist = self.total_distance_traveled

        while time_budget > 1e-6:
            # 1. Trạng thái rảnh (IDLE)
            if self.status == AmbulanceStatus.IDLE:
                break

            # 2. Đang di chuyển đến cuộc gọi (EN_ROUTE_TO_CALL)
            elif self.status == AmbulanceStatus.EN_ROUTE_TO_CALL:
                call = calls_dict.get(self.current_call_id or -1)
                # Kiểm tra nếu cuộc gọi đã bị huỷ/thất bại trước đó
                if call is None or call.status == CallStatus.FAILED:
                    self.abort_mission_to_idle()
                    break

                edge = city_map.edges[self.current_edge]
                v_min = edge.speed / 60.0  # km / phút

                # Xe đang ở trên cung đường đích cuối cùng
                if len(self.route_edges) <= 1:
                    target_dist = self.target_dist if self.target_dist is not None else edge.length
                    rem_dist = target_dist - self.distance_from_start

                    if rem_dist <= 1e-6:
                        # Đã tới hiện trường!
                        arrival_time = current_minute + (dt_minute - time_budget)
                        if arrival_time <= call.deadline + 1e-6:
                            # Đến đúng hoặc trước deadline: tính là THÀNH CÔNG
                            self.distance_from_start = target_dist
                            self.status = AmbulanceStatus.AT_SCENE
                            call.mark_at_scene(arrival_time=arrival_time)
                            self.scene_time_remaining = call.processing_time
                            continue
                        else:
                            # Đến trễ sau deadline
                            call.status = CallStatus.FAILED
                            call.failure_type = "timeout"
                            call.failure_reason = (
                                f"Quá hạn thời gian chờ cấp cứu (đã chờ {call.wait_time:.1f}/{call.failure_time:.0f} phút) "
                                f"trước khi Xe {self.ambulance_id} kịp tiếp cận hiện trường"
                            )
                            self.abort_mission_to_idle()
                            break

                    time_to_reach = rem_dist / max(v_min, 1e-4)
                    if time_budget >= time_to_reach:
                        arrival_time = current_minute + (dt_minute - time_budget + time_to_reach)
                        if arrival_time <= call.deadline + 1e-6:
                            # Đến đúng hoặc trước deadline: tính là THÀNH CÔNG
                            self.distance_from_start = target_dist
                            self.total_distance_traveled += rem_dist
                            time_budget -= time_to_reach
                            self.status = AmbulanceStatus.AT_SCENE
                            call.mark_at_scene(arrival_time=arrival_time)
                            self.scene_time_remaining = call.processing_time
                            continue
                        else:
                            # Đến trễ sau deadline
                            self.distance_from_start = target_dist
                            self.total_distance_traveled += rem_dist
                            call.status = CallStatus.FAILED
                            call.failure_type = "timeout"
                            call.failure_reason = (
                                f"Quá hạn thời gian chờ cấp cứu (đã chờ {call.wait_time:.1f}/{call.failure_time:.0f} phút) "
                                f"trước khi Xe {self.ambulance_id} kịp tiếp cận hiện trường"
                            )
                            self.abort_mission_to_idle()
                            break
                    else:
                        moved = time_budget * v_min
                        self.distance_from_start += moved
                        self.total_distance_traveled += moved
                        time_budget = 0.0
                        break

                # Xe đang ở cung đường trung gian trong tuyến
                else:
                    rem_dist = edge.length - self.distance_from_start
                    time_to_end = rem_dist / max(v_min, 1e-4)

                    if time_budget >= time_to_end:
                        self.total_distance_traveled += rem_dist
                        time_budget -= time_to_end
                        self.route_edges.pop(0)
                        self.current_edge = self.route_edges[0]
                        self.distance_from_start = 0.0
                        continue
                    else:
                        moved = time_budget * v_min
                        self.distance_from_start += moved
                        self.total_distance_traveled += moved
                        time_budget = 0.0
                        break

            # 3. Đang xử lý sơ cứu tại hiện trường (AT_SCENE)
            elif self.status == AmbulanceStatus.AT_SCENE:
                call = calls_dict.get(self.current_call_id or -1)
                if time_budget >= self.scene_time_remaining:
                    time_budget -= self.scene_time_remaining
                    self.scene_time_remaining = 0.0

                    # Hoàn thành sơ cứu cuộc gọi: bắt đầu xuất phát đi bệnh viện
                    if call is not None:
                        call.mark_completed(complete_time=current_minute + (dt_minute - time_budget))

                    # Chọn bệnh viện tối ưu và lập tuyến đường
                    self.select_best_hospital_and_route(city_map)
                    continue
                else:
                    self.scene_time_remaining -= time_budget
                    time_budget = 0.0
                    break

            # 4. Đang di chuyển đến bệnh viện (EN_ROUTE_TO_HOSPITAL)
            elif self.status == AmbulanceStatus.EN_ROUTE_TO_HOSPITAL:
                edge = city_map.edges[self.current_edge]
                v_min = edge.speed / 60.0

                if len(self.route_edges) <= 1:
                    target_dist = self.target_dist if self.target_dist is not None else edge.length
                    rem_dist = target_dist - self.distance_from_start

                    if rem_dist <= 1e-6:
                        # Đã tới cổng bệnh viện!
                        self.distance_from_start = target_dist
                        self.status = AmbulanceStatus.AT_HOSPITAL
                        hosp = city_map.hospitals[self.assigned_hospital_id or 1]
                        self.hospital_time_remaining = hosp.handover_time
                        continue

                    time_to_reach = rem_dist / max(v_min, 1e-4)
                    if time_budget >= time_to_reach:
                        self.distance_from_start = target_dist
                        self.total_distance_traveled += rem_dist
                        time_budget -= time_to_reach
                        self.status = AmbulanceStatus.AT_HOSPITAL
                        hosp = city_map.hospitals[self.assigned_hospital_id or 1]
                        self.hospital_time_remaining = hosp.handover_time
                        continue
                    else:
                        moved = time_budget * v_min
                        self.distance_from_start += moved
                        self.total_distance_traveled += moved
                        time_budget = 0.0
                        break
                else:
                    rem_dist = edge.length - self.distance_from_start
                    time_to_end = rem_dist / max(v_min, 1e-4)

                    if time_budget >= time_to_end:
                        self.total_distance_traveled += rem_dist
                        time_budget -= time_to_end
                        self.route_edges.pop(0)
                        self.current_edge = self.route_edges[0]
                        self.distance_from_start = 0.0
                        continue
                    else:
                        moved = time_budget * v_min
                        self.distance_from_start += moved
                        self.total_distance_traveled += moved
                        time_budget = 0.0
                        break

            # 5. Đang xử lý bàn giao bệnh nhân tại bệnh viện (AT_HOSPITAL)
            elif self.status == AmbulanceStatus.AT_HOSPITAL:
                if time_budget >= self.hospital_time_remaining:
                    time_budget -= self.hospital_time_remaining
                    self.hospital_time_remaining = 0.0

                    # Bàn giao hoàn tất: xe trở về trạng thái rảnh sẵn sàng nhiệm vụ mới
                    self.abort_mission_to_idle()
                    break
                else:
                    self.hospital_time_remaining -= time_budget
                    time_budget = 0.0
                    break

        self.last_step_distance = self.total_distance_traveled - start_total_dist
