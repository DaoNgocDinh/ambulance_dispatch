"""
Môi trường Mô phỏng Điều phối Xe Cứu thương (Ambulance Dispatch Environment).
Quy trình mô phỏng từng phút [t, t+1) với MDP step,
sinh cuộc gọi Poisson, giới hạn tối đa 3 cuộc gọi yêu cầu xử lý,
và ghi log chi tiết theo định dạng yêu cầu.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np

from map_env.city_map import CityMap
from dispatch_env.types import AmbulanceStatus, CallStatus, SEVERITY_CONFIG
from dispatch_env.call import EmergencyCall
from dispatch_env.ambulance import Ambulance


class AmbulanceDispatchEnv:
    """
    Môi trường Điều phối Xe Cứu thương trên mạng lưới 80 nút và 4 bệnh viện.
    """
    def __init__(
        self,
        city_map: Optional[CityMap] = None,
        num_ambulances: int = 4,
        max_time: int = 360,
        poisson_lambda: float = 0.05,
        max_pending_calls: int = 3,
        seed: Optional[int] = 42,
    ) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.city_map = city_map if city_map is not None else CityMap(seed=seed)

        self.num_ambulances = num_ambulances
        self.max_time = max_time
        self.poisson_lambda = poisson_lambda
        self.max_pending_calls = max_pending_calls

        # Thời gian mô phỏng (phút)
        self.current_minute: int = 0
        self.next_call_id: int = 1

        # Danh sách xe cứu thương: id -> Ambulance
        self.ambulances: Dict[int, Ambulance] = {}

        # Danh bạ toàn bộ các cuộc gọi đã phát sinh: id -> EmergencyCall
        self.calls: Dict[int, EmergencyCall] = {}

        # Danh sách cuộc gọi mới sinh ra trong phút vừa qua
        self.calls_generated_this_step: List[EmergencyCall] = []

        # Danh sách các cuộc gọi bị thất bại CHỈ trong phút này (để ghi log 1 lần duy nhất)
        self.calls_failed_this_step: List[EmergencyCall] = []

        # Thống kê tổng kết
        self.served_calls_count: int = 0
        self.failed_calls_count: int = 0
        self.perfect_calls_count: int = 0

        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Khởi tạo lại môi trường:
        - 4 xe cứu thương ở các vị trí ban đầu ngẫu nhiên trên mạng lưới.
        - Trạng thái ban đầu các xe luôn RẢNH (IDLE).
        - Thời gian mô phỏng bắt đầu từ 0 phút.
        """
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            self.city_map = CityMap(seed=seed)

        self.current_minute = 0
        self.next_call_id = 1
        self.calls.clear()
        self.calls_generated_this_step.clear()
        self.calls_failed_this_step.clear()
        self.served_calls_count = 0
        self.failed_calls_count = 0
        self.perfect_calls_count = 0

        # Khởi tạo 4 xe cứu thương tại vị trí ngẫu nhiên trên các cung đường
        all_edges = list(self.city_map.edges.keys())
        self.ambulances.clear()

        # Chọn 4 cung đường ngẫu nhiên riêng biệt cho 4 xe
        chosen_indices = self.rng.choice(len(all_edges), size=self.num_ambulances, replace=False)
        for amb_id in range(1, self.num_ambulances + 1):
            edge_key = all_edges[chosen_indices[amb_id - 1]]
            edge = self.city_map.edges[edge_key]
            # Vị trí ngẫu nhiên dọc theo cung đường [0.0, length]
            init_dist = round(float(self.rng.uniform(0.1, max(0.2, edge.length - 0.1))), 2)

            self.ambulances[amb_id] = Ambulance(
                ambulance_id=amb_id,
                current_edge=edge_key,
                distance_from_start=init_dist,
                status=AmbulanceStatus.IDLE,
                current_call_id=None,
                route_edges=[edge_key],
            )

        return self.get_observation()

    @property
    def pending_calls(self) -> List[EmergencyCall]:
        """Danh sách các cuộc gọi đang ở trạng thái Yêu cầu xử lý (PENDING)."""
        return [c for c in self.calls.values() if c.status == CallStatus.PENDING]

    @property
    def active_calls(self) -> List[EmergencyCall]:
        """Các cuộc gọi đang chờ hoặc đang được xe cứu thương xử lý."""
        return [
            c for c in self.calls.values()
            if c.status in (CallStatus.PENDING, CallStatus.WAITING_FOR_AMBULANCE, CallStatus.AT_SCENE)
        ]

    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """
        Xác định danh sách các hành động phân công hợp lệ:
        A = (Xe cứu thương rảnh, Cuộc gọi đang yêu cầu xử lý).
        """
        idle_ambs = [a.ambulance_id for a in self.ambulances.values() if a.status == AmbulanceStatus.IDLE]
        pending_call_ids = [c.call_id for c in self.pending_calls]

        valid_actions: List[Tuple[int, int]] = []
        for aid in idle_ambs:
            for cid in pending_call_ids:
                valid_actions.append((aid, cid))
        return valid_actions

    def get_observation(self) -> Dict[str, Any]:
        """Quan sát trạng thái hiện tại S_t của môi trường."""
        amb_obs = {}
        for aid, amb in self.ambulances.items():
            amb_obs[aid] = {
                "id": aid,
                "status": amb.status.value,
                "current_edge": amb.current_edge,
                "distance_from_start": amb.distance_from_start,
                "coord": amb.get_coord(self.city_map),
                "current_speed": amb.get_current_speed(self.city_map),
                "road_speed": amb.get_road_speed(self.city_map),
                "current_call_id": amb.current_call_id,
                "target_name": amb.target_name,
                "route": amb.route_edges,
            }

        call_obs = [
            {
                "id": c.call_id,
                "severity": c.severity,
                "edge": c.edge,
                "dist": c.distance_from_start,
                "status": c.status.value,
                "wait_time": c.wait_time,
                "deadline": c.deadline,
            }
            for c in self.active_calls
        ]

        return {
            "current_minute": self.current_minute,
            "ambulances": amb_obs,
            "pending_call_count": len(self.pending_calls),
            "active_calls": call_obs,
            "served_calls": self.served_calls_count,
            "failed_calls": self.failed_calls_count,
            "perfect_calls": self.perfect_calls_count,
        }

    def _generate_new_calls(self) -> List[EmergencyCall]:
        """
        Sinh cuộc gọi mới theo phân phối Poisson (k ~ Poisson(0.05)).
        - Nếu thời gian >= 360 phút: không sinh thêm cuộc gọi.
        - Giới hạn: Chỉ cho phép tối đa 3 cuộc gọi trong trạng thái 'Yêu cầu xử lý' (PENDING).
        - Cuộc gọi vượt quá bị từ chối và tính là Thất bại.
        """
        if self.current_minute >= self.max_time:
            return []

        # Số cuộc gọi sinh ra theo Poisson(0.05)
        num_new_calls = int(self.rng.poisson(lam=self.poisson_lambda))
        new_calls: List[EmergencyCall] = []

        all_edges = list(self.city_map.edges.keys())

        for _ in range(num_new_calls):
            edge_key = all_edges[int(self.rng.integers(0, len(all_edges)))]
            edge = self.city_map.edges[edge_key]
            # Vị trí trên 1 hướng đường
            dist = round(float(self.rng.uniform(0.1, max(0.2, edge.length - 0.1))), 2)

            # Tọa độ địa lý
            nu = self.city_map.nodes[edge_key[0]]
            nv = self.city_map.nodes[edge_key[1]]
            ratio = dist / edge.length
            cx = nu.x + ratio * (nv.x - nu.x)
            cy = nu.y + ratio * (nv.y - nu.y)

            # Mức độ nghiêm trọng (1, 2, 3)
            severity = int(self.rng.choice([1, 2, 3], p=[0.40, 0.35, 0.25]))

            call = EmergencyCall(
                call_id=self.next_call_id,
                edge=edge_key,
                distance_from_start=dist,
                coord=(round(cx, 2), round(cy, 2)),
                severity=severity,
                created_time=float(self.current_minute),
            )
            self.next_call_id += 1

            # Kiểm tra giới hạn 3 cuộc gọi PENDING
            current_pending = [c for c in self.calls.values() if c.status == CallStatus.PENDING]
            if len(current_pending) >= self.max_pending_calls:
                # Bị từ chối ngay lập tức -> Thất bại
                call.mark_rejected()
                self.failed_calls_count += 1
                self.calls_failed_this_step.append(call)
            else:
                call.status = CallStatus.PENDING

            self.calls[call.call_id] = call
            new_calls.append(call)

        return new_calls

    def step(
        self,
        action: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
        auto_dispatch: bool = False,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Thực hiện một bước mô phỏng 1 phút: [t, t+1).
        
        Quy trình chuẩn theo yêu cầu:
        1. Quan sát trạng thái S_t.
        2. Xác định các hành động hợp lệ.
        3. Chọn hành động A_t = (Xe cứu thương, Cuộc gọi).
        4. Thực hiện phân công và lập tuyến đường cho xe mới nhận nhiệm vụ.
        5. Trong khoảng [t, t+1):
           - Các xe di chuyển theo tuyến hiện tại (di chuyển liên tục sub-minute).
           - Xử lý sự kiện đến hiện trường, chăm sóc sơ cứu, đến BV và bàn giao.
           - Cập nhật thời gian chờ và kiểm tra mốc thời gian thất bại.
           - Nếu cuộc gọi thất bại trong khi xe đang tiếp cận, xe lập tức về rảnh!
        6. Tại t+1:
           - Cập nhật trạng thái giao thông.
           - Tính toán lại tuyến đường (reroute) nếu có thay đổi.
           - Sinh cuộc gọi mới (Poisson).
           - Kiểm tra giới hạn 3 cuộc gọi PENDING.
        7. Tính phần thưởng R_{t+1}.
        8. Trả về trạng thái mới S_{t+1}.
        """
        t0 = float(self.current_minute)
        step_reward = 0.0
        self.calls_failed_this_step.clear()

        # --------------------------------------------------------------------
        # 3 & 4. XỬ LÝ HÀNH ĐỘNG PHÂN CÔNG (DISPATCH ACTIONS)
        # --------------------------------------------------------------------
        dispatch_pairs: List[Tuple[int, int]] = []
        if action is not None:
            if isinstance(action, tuple):
                dispatch_pairs = [action]
            elif isinstance(action, list):
                dispatch_pairs = action

        # Nếu chọn chế độ auto_dispatch và chưa có action cụ thể
        if auto_dispatch and not dispatch_pairs:
            # Phân công xe rảnh gần nhất cho cuộc gọi ưu tiên (mức độ nghiêm trọng cao hoặc chờ lâu nhất)
            idle_ambs = [a for a in self.ambulances.values() if a.status == AmbulanceStatus.IDLE]
            sorted_pending = sorted(
                self.pending_calls,
                key=lambda c: (-c.severity, c.created_time),
            )
            for c in sorted_pending:
                if not idle_ambs:
                    break
                # Tìm xe có thời gian tiếp cận nhanh nhất
                best_amb = None
                best_eta = float("inf")
                for a in idle_ambs:
                    r = self.city_map.route_from_road_to_road(
                        start_edge=a.current_edge,
                        start_dist=a.distance_from_start,
                        dest_edge=c.edge,
                        dest_dist=c.distance_from_start,
                        by_travel_time=True,
                    )
                    if r["total_time_minutes"] < best_eta:
                        best_eta = r["total_time_minutes"]
                        best_amb = a
                if best_amb is not None:
                    dispatch_pairs.append((best_amb.ambulance_id, c.call_id))
                    idle_ambs.remove(best_amb)

        # Áp dụng các phân công hợp lệ
        for aid, cid in dispatch_pairs:
            amb = self.ambulances.get(aid)
            call = self.calls.get(cid)
            if amb and call and amb.status == AmbulanceStatus.IDLE and call.status == CallStatus.PENDING:
                amb.dispatch_to_call(call, self.city_map)

        # --------------------------------------------------------------------
        # 5. TRONG KHOẢNG [t, t+1): CÁC XE DI CHUYỂN LIÊN TỤC & XỬ LÝ SỰ KIỆN
        # --------------------------------------------------------------------
        for amb in self.ambulances.values():
            amb.step_continuous(
                dt_minute=1.0,
                current_minute=t0,
                city_map=self.city_map,
                calls_dict=self.calls,
            )

        # Cập nhật thời gian chờ và kiểm tra mốc thất bại
        for call in list(self.calls.values()):
            if call.status in (CallStatus.PENDING, CallStatus.WAITING_FOR_AMBULANCE):
                call.update_wait_time(1.0)
                # Phạt thời gian chờ tỷ lệ với mức độ nghiêm trọng
                step_reward -= 1.0 * call.severity

                if call.check_failure(current_time=t0 + 1.0):
                    # Cuộc gọi bị thất bại do quá hạn
                    self.failed_calls_count += 1
                    self.calls_failed_this_step.append(call)
                    step_reward -= 50.0 * call.severity

                    # "Khi cuộc gọi bị xoá mà trong lúc đó có xe đang tiếp cận thì xe sẽ lập tức về trạng thái rảnh"
                    if call.assigned_ambulance_id is not None:
                        assigned_amb = self.ambulances.get(call.assigned_ambulance_id)
                        if assigned_amb and assigned_amb.status == AmbulanceStatus.EN_ROUTE_TO_CALL:
                            assigned_amb.abort_mission_to_idle()

        # Kiểm tra phần thưởng cho các cuộc gọi vừa hoàn thành
        for call in self.calls.values():
            if call.status == CallStatus.COMPLETED and call.completed_time is not None:
                if t0 <= call.completed_time < t0 + 1.0:
                    self.served_calls_count += 1
                    reward_add = 100.0 * call.severity
                    if call.is_perfect:
                        self.perfect_calls_count += 1
                        reward_add += 50.0 * call.severity  # Thưởng thêm hoàn thành hoàn hảo
                    step_reward += reward_add

        # --------------------------------------------------------------------
        # 6. TẠI t+1: CẬP NHẬT GIAO THÔNG, REROUTE, SINH CUỘC GỌI MỚI
        # --------------------------------------------------------------------
        self.current_minute += 1

        # Cập nhật trạng thái giao thông (Markov transition)
        self.city_map.step_traffic()

        # Reroute xe nếu có thay đổi trên lộ trình
        for amb in self.ambulances.values():
            amb.reroute_if_beneficial(self.city_map)

        # Sinh cuộc gọi mới tại đầu phút t+1
        new_calls = self._generate_new_calls()
        self.calls_generated_this_step = new_calls

        # --------------------------------------------------------------------
        # 7 & 8. KIỂM TRA ĐIỀU KIỆN KẾT THÚC & TRẢ VỀ
        # --------------------------------------------------------------------
        # Mô phỏng kết thúc khi đã vượt qua max_time (360 phút)
        # VÀ không còn cuộc gọi nào đang hoạt động (tất cả xe đã hoàn thành nhiệm vụ)
        done = False
        if self.current_minute >= self.max_time:
            active_count = len(self.active_calls)
            all_ambs_idle = all(a.status == AmbulanceStatus.IDLE for a in self.ambulances.values())
            if active_count == 0 or all_ambs_idle:
                done = True

        info = {
            "minute": self.current_minute,
            "generated_calls": len(new_calls),
            "served": self.served_calls_count,
            "failed": self.failed_calls_count,
            "perfect": self.perfect_calls_count,
        }

        return self.get_observation(), step_reward, done, info

    def format_log_step(self) -> str:
        """
        Định dạng chuỗi log cho phút hiện tại theo ĐÚNG mẫu yêu cầu của người dùng:
        -------------------------------------------------------------
        Phút thứ t:
        Số cuộc gọi sinh ra: …
        Thông tin cuộc gọi …: 
        —--------------------------------------------
        Xe 1 đang di chuyển trên tuyến đường :... Hiện đang ở vị trí …. đang xử lý cuộc gọi: ….. Trạng thái: ………….
        Xe N đang di chuyển trên tuyến đường :... Hiện đang ở vị trí …. đang xử lý cuộc gọi: ….. Trạng thái: ………….
        —--------------------------------------------
        Cách tuyến đường chuyển trạng thái:
        Từ tắc -> bình thường: ………..
        Từ thông thoáng -> bình thường: ………..
        Từ bình thường -> tắc: ………..
        Từ bình thường -> thông thoáng: ………..
        Từ tắc -> thông thoáng: ………..
        Từ thông thoáng -> tắc: ………..
        —--------------------------------------------
        Số cuộc gọi thành công: 
        Số cuộc gọi thất bại:
        —---------------------------------------------
        Xe … đã tính toán lại đoạn đường
        """
        t = self.current_minute
        lines: List[str] = []

        lines.append(f"Phút thứ {t}:")
        lines.append(f"Số cuộc gọi sinh ra: {len(self.calls_generated_this_step)}")

        if not self.calls_generated_this_step:
            lines.append("Thông tin cuộc gọi: Không có cuộc gọi mới sinh ra.")
        else:
            for c in self.calls_generated_this_step:
                status_str = c.status.value
                if c.status == CallStatus.FAILED and c.failure_reason:
                    status_str += f" - Lý do: {c.failure_reason}"
                c_info = (
                    f"Thông tin cuộc gọi #{c.call_id}: "
                    f"Vị trí: N{c.edge[0]}->N{c.edge[1]} (+{c.distance_from_start:.2f}km), "
                    f"Mức độ nghiêm trọng: {c.severity}, "
                    f"Thời gian thất bại: {c.failure_time:.0f}p, "
                    f"Thời gian cần xử lý: {c.processing_time:.0f}p, "
                    f"Trạng thái: {status_str}"
                )
                lines.append(c_info)

        lines.append("—--------------------------------------------")
        # Thông tin từng xe
        for aid in range(1, self.num_ambulances + 1):
            amb = self.ambulances[aid]
            route_str = " -> ".join(f"N{u}->N{v}" for u, v in amb.route_edges) if amb.route_edges else f"N{amb.current_edge[0]}->N{amb.current_edge[1]}"
            u, v = amb.current_edge
            coord = amb.get_coord(self.city_map)
            pos_str = f"({coord[0]:.2f}, {coord[1]:.2f}) km trên cung N{u}->N{v} (+{amb.distance_from_start:.2f}km)"
            call_str = f"#{amb.current_call_id}" if amb.current_call_id is not None else "Không"
            
            cur_speed = amb.get_current_speed(self.city_map)
            road_speed = amb.get_road_speed(self.city_map)
            if amb.status in (AmbulanceStatus.EN_ROUTE_TO_CALL, AmbulanceStatus.EN_ROUTE_TO_HOSPITAL):
                speed_str = f"{cur_speed:.1f} km/h"
            else:
                speed_str = f"0.0 km/h (Tốc độ tuyến: {road_speed:.1f} km/h)"

            amb_line = (
                f"Xe {amb.ambulance_id} đang di chuyển trên tuyến đường : {route_str} "
                f"Hiện đang ở vị trí {pos_str} "
                f"Vận tốc: {speed_str} "
                f"đang xử lý cuộc gọi: {call_str} "
                f"Trạng thái: {amb.status.value}"
            )
            lines.append(amb_line)

        lines.append("—--------------------------------------------")
        # Thống kê biến thiên giao thông
        tr = self.city_map.last_traffic_transitions
        lines.append("Cách tuyến đường chuyển trạng thái:")
        lines.append(f"Từ tắc -> bình thường: {', '.join(tr['tac_to_binh_thuong']) if tr['tac_to_binh_thuong'] else 'Không có'}")
        lines.append(f"Từ thông thoáng -> bình thường: {', '.join(tr['thong_thoang_to_binh_thuong']) if tr['thong_thoang_to_binh_thuong'] else 'Không có'}")
        lines.append(f"Từ bình thường -> tắc: {', '.join(tr['binh_thuong_to_tac']) if tr['binh_thuong_to_tac'] else 'Không có'}")
        lines.append(f"Từ bình thường -> thông thoáng: {', '.join(tr['binh_thuong_to_thong_thoang']) if tr['binh_thuong_to_thong_thoang'] else 'Không có'}")
        lines.append(f"Từ tắc -> thông thoáng: {', '.join(tr['tac_to_thong_thoang']) if tr['tac_to_thong_thoang'] else 'Không có'}")
        lines.append(f"Từ thông thoáng -> tắc: {', '.join(tr['thong_thoang_to_tac']) if tr['thong_thoang_to_tac'] else 'Không có'}")

        lines.append("—--------------------------------------------")
        lines.append(f"Số cuộc gọi thành công: {self.served_calls_count}")
        lines.append(f"Số cuộc gọi thất bại: {self.failed_calls_count}")
        if self.calls_failed_this_step:
            lines.append("Lý do cuộc gọi thất bại trong phút này:")
            for fc in self.calls_failed_this_step:
                lines.append(f"• Cuộc gọi #{fc.call_id} (Mức {fc.severity}): {fc.failure_reason}")
        else:
            lines.append("Lý do cuộc gọi thất bại trong phút này: Không có")
        lines.append("—---------------------------------------------")

        # Thông báo xe tính toán lại tuyến đường
        rerouted_ambs = [a.ambulance_id for a in self.ambulances.values() if a.last_rerouted_this_step]
        for aid in rerouted_ambs:
            lines.append(f"Xe {aid} đã tính toán lại đoạn đường")

        return "\n".join(lines)
