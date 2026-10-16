"""
Môi trường Mô phỏng Điều phối Xe cứu thương Khẩn cấp trong Thành phố Thông minh (MDP).
Thiết kế bám sát Mục III và Mục IV (Thẻ 1 của báo cáo) với thông số:
- 15 nút giao thông (|V| = 15)
- 3 xe cứu thương (K = 3)
- Tối đa 3 cuộc gọi đồng thời (M = 3)
- 5 tuyến đường huyết mạch (|E_arterial| = 5)

Hỗ trợ đầy đủ cho 4 thuật toán:
1. Dynamic Programming (Value Iteration)
2. Monte Carlo
3. Tabular Q-Learning
4. PPO (Proximal Policy Optimization)
cùng thuật toán Baseline (Nearest-Ambulance / Priority-Greedy).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .city_graph import SmartCityGraph
from .config import (
    AMB_STATE_IDLE,
    AMB_STATE_ON_SCENE,
    AMB_STATE_RELOCATING,
    AMB_STATE_TO_HOSPITAL,
    AMB_STATE_TO_SCENE,
    DEFAULT_GAMMA_PPO,
    DEFAULT_GAMMA_TABULAR,
    EnvConfig,
    PENALTY_CONGESTED_ROAD,
    PENALTY_EMPTY_STATIONS,
    PENALTY_INVALID_ACTION,
    PENALTY_MAX_WAIT_EXCEEDED,
    PENALTY_WAIT_PER_STEP,
    PRIO_GREEN,
    PRIO_NONE,
    PRIO_PROBABILITIES,
    PRIO_RED,
    PRIO_YELLOW,
    REWARD_GOLDEN_TIME,
    REWARD_HOSPITAL_DELIVERY,
    REWARD_OPTIMAL_DIRECTION,
     TRAFFIC_CONGESTED,
    TRAFFIC_NAMES,
)
from .entities import Ambulance, EmergencyRequest


class SmartCityAmbulanceEnv:
    """
    Môi trường MDP điều phối xe cứu thương đô thị thông minh: M = <S, A, P, R, gamma>.

    Không gian trạng thái S_t = [X_amb, X_req, X_traffic] (20 chiều):
        - X_amb (6 chiều): [p_1, st_1, p_2, st_2, p_3, st_3]
            + p_k in {1, ..., 15}: Nút hiện tại của xe k
            + st_k in {0, 1, 2, 3, 4}: Trạng thái hoạt động của xe k
        - X_req (9 chiều): [loc_1, prio_1, wait_1, loc_2, prio_2, wait_2, loc_3, prio_3, wait_3]
            + loc_m in {0, 1, ..., 15} (0 nếu slot trống)
            + prio_m in {0, 1, 2, 3} (1: Đỏ, 2: Vàng, 3: Xanh, 0: Trống)
            + wait_m >= 0: Thời gian đã chờ (bước)
        - X_traffic (5 chiều): [c_1, c_2, c_3, c_4, c_5]
            + c_e in {0: Thông thoáng, 1: Đông, 2: Ùn tắc} trên 5 tuyến đường huyết mạch

    Không gian hành động A (25 hành động rời rạc):
        - 0..8   (9 hành động) : a_dispatch(k, m) - Phân bổ xe rảnh k in {0,1,2} phục vụ ca chờ m in {0,1,2}
        - 9..17  (9 hành động) : a_relocate(k, j) - Tái định vị xe rảnh k in {0,1,2} về trạm j in {0,1,2}
        - 18..23 (6 hành động) : a_route(k, u_opt)- Chỉ định xe đang chạy k chọn hướng tránh kẹt xe (u_opt in {0,1})
        - 24     (1 hành động) : a_standby        - Giữ nguyên trạng thái / chờ tại trạm / tiếp tục lộ trình
    """

    STANDBY_ACTION: int = 24

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config if config is not None else EnvConfig()

        self.num_nodes: int = self.config.num_nodes                    # 15
        self.num_ambulances: int = self.config.num_ambulances          # 3
        self.max_pending_requests: int = self.config.max_pending_requests  # 3
        self.num_arterial_roads: int = self.config.num_arterial_roads  # 5
        self.max_steps: int = self.config.max_steps
        self.gamma: float = self.config.gamma

        self.rng: np.random.Generator = np.random.default_rng(seed)

        # Khởi tạo đồ thị giao thông 15 nút và 5 tuyến đường huyết mạch
        self.graph = SmartCityGraph(
            noise_distribution=self.config.noise_distribution,
        )

        # Số lượng hành động: 9 (dispatch) + 9 (relocate) + 6 (route) + 1 (standby) = 25
        self.number_of_actions: int = 25

        # Khởi tạo đội xe K = 3 tại 3 trạm trực [1, 8, 15]
        self.ambulances: List[Ambulance] = [
            Ambulance(
                amb_id=k,
                home_station=self.config.station_nodes[k],
                node=self.config.station_nodes[k],
            )
            for k in range(self.num_ambulances)
        ]

        # Danh sách tối đa M = 3 cuộc gọi đang chờ phân bổ xe
        self.pending_requests: List[Optional[EmergencyRequest]] = [
            None for _ in range(self.max_pending_requests)
        ]

        # Bộ đếm & thống kê đánh giá hiệu năng (Mục 3.5)
        self.step_count: int = 0
        self.next_request_id: int = 1
        self.total_reward: float = 0.0

        self.total_spawned_requests: int = 0
        self.dispatched_requests: int = 0
        self.golden_time_reached: int = 0
        self.completed_hospital_deliveries: int = 0
        self.expired_requests: int = 0
        self.invalid_actions: int = 0
        self.congested_road_steps: int = 0
        self.empty_station_penalties: int = 0
        self.response_times: List[int] = []

    # ========================================================================
    # THUỘC TÍNH KÍCH THƯỚC KHÔNG GIAN TRẠNG THÁI & HÀNH ĐỘNG
    # ========================================================================
    @property
    def state_vector_size(self) -> int:
        """
        Kích thước vector trạng thái S_t = [X_amb, X_req, X_traffic]:
            2 * K + 3 * M + |E_arterial| = 2*3 + 3*3 + 5 = 20 chiều.
        """
        return (2 * self.num_ambulances) + (3 * self.max_pending_requests) + self.num_arterial_roads

    @staticmethod
    def analyze_state_space(
        num_nodes: int = 15,
        num_ambulances: int = 3,
        max_requests: int = 3,
        num_arterials: int = 5,
        max_wait_steps: int = 14,
    ) -> Dict[str, Any]:
        """
        Phân tích số lượng trạng thái lý thuyết theo tổ hợp (Mục 4.2.1):
        - X_amb: (num_nodes * 5)^K
        - X_req: (1 + num_nodes * 3 * (max_wait_steps + 1))^M
        - X_traffic: 3^|E_arterial|
        """
        amb_combinations = (num_nodes * 5) ** num_ambulances
        req_slot_states = 1 + (num_nodes * 3 * (max_wait_steps + 1))
        req_combinations = req_slot_states ** max_requests
        traffic_combinations = 3 ** num_arterials
        total_combinations = amb_combinations * req_combinations * traffic_combinations
        return {
            "num_nodes": num_nodes,
            "num_ambulances": num_ambulances,
            "max_requests": max_requests,
            "num_arterials": num_arterials,
            "X_amb_states": amb_combinations,
            "X_req_states": req_combinations,
            "X_traffic_states": traffic_combinations,
            "total_theoretical_states": total_combinations,
            "scientific_notation": f"{total_combinations:.3e}",
        }

    # ========================================================================
    # MÃ HÓA & GIẢI MÃ HÀNH ĐỘNG (MỤC 4.2.2)
    # ========================================================================
    def decode_action(self, action: int) -> Dict[str, Any]:
        """
        Giải mã chỉ số hành động nguyên (0..24) sang cấu trúc ngữ nghĩa MDP:
            - 0..8   : a_dispatch(k, m)
            - 9..17  : a_relocate(k, j)
            - 18..23 : a_route(k, u_opt)
            - 24     : a_standby
        """
        if not isinstance(action, (int, np.integer)) or action < 0 or action >= self.number_of_actions:
            raise ValueError(f"Hành động phải là số nguyên trong khoảng [0, {self.number_of_actions - 1}].")

        action = int(action)
        if 0 <= action < 9:
            k = action // 3
            m = action % 3
            return {
                "type": "dispatch",
                "amb_id": k,
                "req_slot": m,
                "label": f"a_dispatch(k={k + 1}, m={m + 1})",
            }
        if 9 <= action < 18:
            idx = action - 9
            k = idx // 3
            j = idx % 3
            station_node = self.config.station_nodes[j]
            return {
                "type": "relocate",
                "amb_id": k,
                "station_idx": j,
                "station_node": station_node,
                "label": f"a_relocate(k={k + 1}, station={station_node})",
            }
        if 18 <= action < 24:
            idx = action - 18
            k = idx // 2
            u_opt = idx % 2
            mode_str = "nhánh nhanh nhất tránh tắc" if u_opt == 0 else "nhánh thay thế thứ 2"
            return {
                "type": "route",
                "amb_id": k,
                "route_option": u_opt,
                "label": f"a_route(k={k + 1}, u_opt={u_opt}:{mode_str})",
            }
        return {
            "type": "standby",
            "label": "a_standby",
        }

    @staticmethod
    def encode_dispatch_action(amb_id: int, req_slot: int) -> int:
        """Mã hóa hành động a_dispatch(k, m) -> 0..8."""
        return int(amb_id * 3 + req_slot)

    @staticmethod
    def encode_relocate_action(amb_id: int, station_idx: int) -> int:
        """Mã hóa hành động a_relocate(k, j) -> 9..17."""
        return int(9 + amb_id * 3 + station_idx)

    @staticmethod
    def encode_route_action(amb_id: int, route_option: int = 0) -> int:
        """Mã hóa hành động a_route(k, u_opt) -> 18..23."""
        return int(18 + amb_id * 2 + route_option)

    def get_action_mask(self) -> np.ndarray:
        """
        Trả về vector nhị phân (25 chiều) đánh dấu các hành động hợp lệ tại trạng thái hiện tại.
        1 = Hợp lệ, 0 = Không hợp lệ.
        """
        mask = np.zeros(self.number_of_actions, dtype=np.int8)

        # 1. Kiểm tra a_dispatch(k, m): Xe k phải rảnh (st_k == 0) và slot m phải có cuộc gọi chờ
        for k in range(self.num_ambulances):
            if self.ambulances[k].can_be_dispatched:
                for m in range(self.max_pending_requests):
                    if self.pending_requests[m] is not None:
                        mask[self.encode_dispatch_action(k, m)] = 1

        # 2. Kiểm tra a_relocate(k, j): Xe k phải rảnh (st_k == 0) và chưa đứng tại trạm j
        for k in range(self.num_ambulances):
            if self.ambulances[k].can_be_dispatched:
                for j, station_node in enumerate(self.config.station_nodes):
                    if self.ambulances[k].node != station_node:
                        mask[self.encode_relocate_action(k, j)] = 1

        # 3. Kiểm tra a_route(k, u_opt): Xe k đang di chuyển (st_k in {1, 3, 4}), đang ở nút giao (edge_progress == 0)
        for k in range(self.num_ambulances):
            amb = self.ambulances[k]
            if amb.status in (AMB_STATE_TO_SCENE, AMB_STATE_TO_HOSPITAL, AMB_STATE_RELOCATING):
                if amb.edge_progress == 0.0 and amb.target_node is not None and amb.node != amb.target_node:
                    neighbors = self.graph.get_neighbors(amb.node)
                    if len(neighbors) >= 1:
                        mask[self.encode_route_action(k, 0)] = 1
                    if len(neighbors) >= 2:
                        mask[self.encode_route_action(k, 1)] = 1

        # 4. Hành động a_standby: hợp lệ khi không thể dispatch ngay hoặc dùng làm lựa chọn giữ nguyên lộ trình
        has_dispatchable_pair = bool(mask[:9].any())
        if not has_dispatchable_pair:
            mask[self.STANDBY_ACTION] = 1
        else:
            # Vẫn cho phép standby trong không gian hành động nhưng ưu tiên dispatch khi có ca nguy kịch
            mask[self.STANDBY_ACTION] = 1

        return mask

    def get_valid_actions(self) -> List[int]:
        """Danh sách các chỉ số hành động hợp lệ tại bước hiện tại."""
        mask = self.get_action_mask()
        return [int(i) for i, v in enumerate(mask) if v == 1]

    # ========================================================================
    # KHỞI TẠO LẠI MÔI TRƯỜNG (RESET) & SINH CUỘC GỌI POISSON (MỤC 4.2.4)
    # ========================================================================
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Khởi tạo lại môi trường cho một episode mới."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Đặt 3 xe cứu thương về 3 trạm trực [1, 8, 15]
        for k, amb in enumerate(self.ambulances):
            amb.reset(initial_node=self.config.station_nodes[k])

        # Xóa danh sách cuộc gọi chờ
        self.pending_requests = [None for _ in range(self.max_pending_requests)]

        # Khởi tạo trạng thái giao thông trên 5 tuyến đường huyết mạch
        self.graph.reset_traffic(self.rng, initial_rush_hour=False)

        # Đặt lại các biến thống kê
        self.step_count = 0
        self.next_request_id = 1
        self.total_reward = 0.0
        self.total_spawned_requests = 0
        self.dispatched_requests = 0
        self.golden_time_reached = 0
        self.completed_hospital_deliveries = 0
        self.expired_requests = 0
        self.invalid_actions = 0
        self.congested_road_steps = 0
        self.empty_station_penalties = 0
        self.response_times = []

        # Đảm bảo ở thời điểm t = 0 có ít nhất 1 cuộc gọi cấp cứu khởi đầu để tác tử ra quyết định
        self._spawn_poisson_requests(force_at_least_one=True)

        return self.get_observation()

    def _spawn_poisson_requests(self, force_at_least_one: bool = False) -> List[EmergencyRequest]:
        """
        Sinh cuộc gọi cấp cứu mới theo quá trình Poisson không đồng nhất (Mục 4.2.4):
            P(k cuộc gọi mới) = e^(-lambda * Delta_t) * (lambda * Delta_t)^k / k!
        với cường độ lambda(v, t) phụ thuộc vào mật độ dân cư của nút v và hệ số giờ cao điểm.
        """
        spawned: List[EmergencyRequest] = []
        empty_slots = [m for m in range(self.max_pending_requests) if self.pending_requests[m] is None]
        if not empty_slots:
            return spawned

        # Hệ số giờ cao điểm làm tăng nhu cầu cấp cứu
        is_rush_hour = (20 <= (self.step_count % 50) <= 38)
        rush_multiplier = 1.35 if is_rush_hour else 1.0

        # Cường độ Poisson tại từng nút: lambda(v, t) = base_rate * rush_multiplier * demand_weight(v)
        node_ids = list(range(1, self.num_nodes + 1))
        node_lambdas = np.array([
            self.config.base_arrival_rate * rush_multiplier * self.graph.nodes[n].demand_weight
            for n in node_ids
        ])

        # Sinh số lượng cuộc gọi tại mỗi nút theo phân phối Poisson
        counts_per_node = self.rng.poisson(lam=node_lambdas * self.config.delta_t)
        candidate_nodes: List[int] = []
        for node_id, count in zip(node_ids, counts_per_node):
            for _ in range(int(count)):
                candidate_nodes.append(node_id)

        if force_at_least_one and not candidate_nodes:
            weights = np.array([self.graph.nodes[n].demand_weight for n in node_ids])
            chosen_node = int(self.rng.choice(node_ids, p=weights))
            candidate_nodes.append(chosen_node)

        # Trộn ngẫu nhiên thứ tự nếu có nhiều cuộc gọi cùng lúc
        if len(candidate_nodes) > 1:
            self.rng.shuffle(candidate_nodes)

        prio_choices = [PRIO_RED, PRIO_YELLOW, PRIO_GREEN]
        prio_probs = [
            PRIO_PROBABILITIES[PRIO_RED],
            PRIO_PROBABILITIES[PRIO_YELLOW],
            PRIO_PROBABILITIES[PRIO_GREEN],
        ]

        for loc_node in candidate_nodes:
            if not empty_slots:
                break
            slot_idx = empty_slots.pop(0)
            prio = int(self.rng.choice(prio_choices, p=prio_probs))
            req = EmergencyRequest(
                req_id=self.next_request_id,
                loc=int(loc_node),
                prio=prio,
                wait_time=0,
                spawn_step=self.step_count,
            )
            self.next_request_id += 1
            self.total_spawned_requests += 1
            self.pending_requests[slot_idx] = req
            spawned.append(req)

        return spawned

    # ========================================================================
    # THỰC THI BƯỚC THỜI GIAN MDP: step(action) (MỤC 4.2.2, 4.2.3, 4.2.4)
    # ========================================================================
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Thực thi hành động a_t tại trạng thái S_t, cập nhật động học hệ thống sang S_{t+1}
        và tính toán phần thưởng đa mục tiêu R(S_t, a_t).

        Returns
        -------
        next_obs : dict
            Quan sát tại trạng thái kế tiếp S_{t+1}.
        reward : float
            Tổng phần thưởng ở bước thời gian t.
        done : bool
            Cờ kết thúc episode (khi đạt max_steps).
        info : dict
            Thông tin chi tiết về các sự kiện và thành phần phần thưởng trong bước.
        """
        action_info = self.decode_action(action)
        reward: float = 0.0
        reward_breakdown: Dict[str, float] = {
            "golden_time_bonus": 0.0,
            "hospital_delivery_bonus": 0.0,
            "optimal_direction_bonus": 0.0,
            "waiting_penalty": 0.0,
            "max_wait_exceeded_penalty": 0.0,
            "congested_road_penalty": 0.0,
            "empty_station_penalty": 0.0,
            "invalid_action_penalty": 0.0,
        }
        events: List[str] = []

        # --------------------------------------------------------------------
        # BƯỚC 1: XỬ LÝ HÀNH ĐỘNG ĐIỀU PHỐI CỦA AGENT (A)
        # --------------------------------------------------------------------
        act_type = action_info["type"]

        if act_type == "dispatch":
            k = action_info["amb_id"]
            m = action_info["req_slot"]
            amb = self.ambulances[k]
            req = self.pending_requests[m]

            if not amb.can_be_dispatched or req is None:
                reward += PENALTY_INVALID_ACTION
                reward_breakdown["invalid_action_penalty"] += PENALTY_INVALID_ACTION
                self.invalid_actions += 1
                events.append(f"Hành động không hợp lệ: {action_info['label']}")
            else:
                # Phân bổ xe k phục vụ ca cấp cứu m
                self.pending_requests[m] = None
                req.assigned_amb_id = k
                amb.active_request = req
                amb.target_node = req.loc
                amb.edge_progress = 0.0
                amb.moving_to_node = None
                amb.forced_next_node = None
                self.dispatched_requests += 1

                if amb.node == req.loc:
                    # Xe đang ở ngay tại nút xảy ra sự cố -> tiếp cận tức thì!
                    amb.status = AMB_STATE_ON_SCENE
                    req.response_time = req.wait_time
                    self.response_times.append(req.response_time)
                    if req.response_time <= req.golden_time:
                        reward += REWARD_GOLDEN_TIME
                        reward_breakdown["golden_time_bonus"] += REWARD_GOLDEN_TIME
                        self.golden_time_reached += 1
                        events.append(
                            f"Xe {k + 1} tiếp cận tức thì ca #{req.req_id} tại nút {req.loc} "
                            f"trong Thời gian vàng (T_resp={req.response_time} <= {req.golden_time}) [+50]"
                        )
                    else:
                        events.append(
                            f"Xe {k + 1} tiếp cận ca #{req.req_id} tại nút {req.loc} (T_resp={req.response_time})"
                        )
                else:
                    amb.status = AMB_STATE_TO_SCENE
                    events.append(
                        f"Phân bổ Xe {k + 1} (từ nút {amb.node}) đi đón ca #{req.req_id} "
                        f"({req.prio_name}) tại nút {req.loc}"
                    )

        elif act_type == "relocate":
            k = action_info["amb_id"]
            station_node = action_info["station_node"]
            amb = self.ambulances[k]

            if not amb.can_be_dispatched or amb.node == station_node:
                reward += PENALTY_INVALID_ACTION
                reward_breakdown["invalid_action_penalty"] += PENALTY_INVALID_ACTION
                self.invalid_actions += 1
                events.append(f"Tái định vị không hợp lệ: {action_info['label']}")
            else:
                amb.status = AMB_STATE_RELOCATING
                amb.target_node = station_node
                amb.edge_progress = 0.0
                amb.moving_to_node = None
                amb.forced_next_node = None
                events.append(f"Xe {k + 1} tái định vị từ nút {amb.node} về Trạm nút {station_node}")

        elif act_type == "route":
            k = action_info["amb_id"]
            u_opt = action_info["route_option"]
            amb = self.ambulances[k]

            if (
                amb.status in (AMB_STATE_TO_SCENE, AMB_STATE_TO_HOSPITAL, AMB_STATE_RELOCATING)
                and amb.edge_progress == 0.0
                and amb.target_node is not None
                and amb.node != amb.target_node
            ):
                ranked_hops = self.graph.get_ranked_next_hops(
                    amb.node, amb.target_node, consider_traffic=True
                )
                if u_opt < len(ranked_hops):
                    chosen_next_node = ranked_hops[u_opt][0]
                    amb.forced_next_node = chosen_next_node
                    events.append(
                        f"Định tuyến Xe {k + 1} tại nút {amb.node} rẽ sang nút {chosen_next_node} "
                        f"(đích: nút {amb.target_node})"
                    )
                else:
                    reward += PENALTY_INVALID_ACTION
                    reward_breakdown["invalid_action_penalty"] += PENALTY_INVALID_ACTION
                    self.invalid_actions += 1
            else:
                reward += PENALTY_INVALID_ACTION
                reward_breakdown["invalid_action_penalty"] += PENALTY_INVALID_ACTION
                self.invalid_actions += 1
                events.append(f"Định tuyến không hợp lệ: {action_info['label']}")

        elif act_type == "standby":
            # Nếu có xe rảnh và có cuộc gọi đang chờ mà Agent lại chọn đứng chờ -> nhắc trong log
            events.append("Bộ điều phối chọn giữ nguyên trạng thái (a_standby)")

        # --------------------------------------------------------------------
        # BƯỚC 2: CẬP NHẬT DI CHUYỂN & VÒNG ĐỜI HOẠT ĐỘNG CỦA 3 XE CỨU THƯƠNG
        # --------------------------------------------------------------------
        for k, amb in enumerate(self.ambulances):
            if amb.status == AMB_STATE_IDLE:
                continue

            # Trường hợp xe đang sơ cứu tại hiện trường (st_k == 2) -> sau 1 bước bắt đầu chuyển viện (st_k == 3)
            if amb.status == AMB_STATE_ON_SCENE:
                best_hosp, _ = self.graph.nearest_hospital(amb.node, consider_traffic=True)
                amb.target_node = best_hosp
                amb.edge_progress = 0.0
                amb.moving_to_node = None
                amb.forced_next_node = None

                if amb.node == best_hosp:
                    # Hiện trường trùng ngay nút Bệnh viện -> hoàn thành chuyển viện ngay!
                    amb.status = AMB_STATE_IDLE
                    amb.target_node = None
                    amb.active_request = None
                    reward += REWARD_HOSPITAL_DELIVERY
                    reward_breakdown["hospital_delivery_bonus"] += REWARD_HOSPITAL_DELIVERY
                    self.completed_hospital_deliveries += 1
                    events.append(f"Xe {k + 1} hoàn tất chuyển viện ngay tại Bệnh viện nút {best_hosp} [+20]")
                else:
                    amb.status = AMB_STATE_TO_HOSPITAL
                    events.append(
                        f"Xe {k + 1} sơ cứu xong tại nút {amb.node}, chuyển bệnh nhân tới Bệnh viện nút {best_hosp}"
                    )
                continue

            # Các trạng thái di chuyển trên đồ thị: TO_SCENE (1), TO_HOSPITAL (3), RELOCATING (4)
            if amb.target_node is None or amb.node == amb.target_node:
                self._handle_ambulance_arrival(amb, reward_breakdown, events)
                continue

            # Chọn nút kế tiếp trên lộ trình nếu xe đang đứng tại nút (edge_progress == 0)
            if amb.moving_to_node is None:
                if amb.forced_next_node is not None and amb.forced_next_node in self.graph.get_neighbors(amb.node):
                    amb.moving_to_node = amb.forced_next_node
                    amb.forced_next_node = None
                else:
                    # Mặc định nếu không dùng a_route: xe đi theo đường ngắn nhất hình học tĩnh
                    path, _ = self.graph.shortest_path(amb.node, amb.target_node, consider_traffic=False)
                    amb.moving_to_node = path[1] if len(path) > 1 else amb.target_node

            next_node = amb.moving_to_node
            assert next_node is not None

            # Kiểm tra xem xe có đi vào cung đường đang ùn tắc (c_e == 2) không (Mục 4.2.3: phạt -2/bước)
            cong = self.graph.get_edge_congestion(amb.node, next_node)
            if cong == TRAFFIC_CONGESTED:
                reward += PENALTY_CONGESTED_ROAD
                reward_breakdown["congested_road_penalty"] += PENALTY_CONGESTED_ROAD
                self.congested_road_steps += 1
                events.append(
                    f"Xe {k + 1} đi vào đường ùn tắc ({amb.node} <-> {next_node}) [-2]"
                )

            # Kiểm tra thưởng di chuyển đúng hướng tối ưu (+1) (Mục 4.2.3)
            dist_before = self.graph.get_static_distance(amb.node, amb.target_node)
            dist_after = self.graph.get_static_distance(next_node, amb.target_node)
            if amb.status in (AMB_STATE_TO_SCENE, AMB_STATE_TO_HOSPITAL) and dist_after < dist_before:
                reward += REWARD_OPTIMAL_DIRECTION
                reward_breakdown["optimal_direction_bonus"] += REWARD_OPTIMAL_DIRECTION

            # Tính vận tốc thực tế v(e, t) = v_max * (1 - alpha * cong(e, t)) + nhiễu
            edge = self.graph.get_edge(amb.node, next_node)
            edge_len = edge.base_length if edge is not None else 1.0
            vel = self.graph.compute_edge_velocity(amb.node, next_node, rng=self.rng, deterministic=False)
            amb.edge_progress += (vel * self.config.delta_t) / edge_len

            # Nếu xe đang đi đón bệnh nhân (st_k == 1), thời gian phản hồi của ca đó tăng thêm 1 bước
            if amb.status == AMB_STATE_TO_SCENE and amb.active_request is not None:
                amb.active_request.wait_time += 1

            # Kiểm tra xe đã hoàn thành cung đường để tới nút next_node chưa
            if amb.edge_progress >= 1.0 - 1e-9:
                amb.node = next_node
                amb.edge_progress = 0.0
                amb.moving_to_node = None

                if amb.node == amb.target_node:
                    arr_reward = self._handle_ambulance_arrival(amb, reward_breakdown, events)
                    reward += arr_reward

        # --------------------------------------------------------------------
        # BƯỚC 3: CẬP NHẬT THỜI GIAN CHỜ & PHẠT CHỜ ĐỢI CHO CÁC CUỘC GỌI (MỤC 4.2.3)
        # --------------------------------------------------------------------
        # 3a. Các cuộc gọi đang chờ trong hàng đợi X_req (chưa có xe tiếp nhận)
        for m in range(self.max_pending_requests):
            req = self.pending_requests[m]
            if req is None:
                continue

            req.wait_time += 1
            step_wait_pen = PENALTY_WAIT_PER_STEP[req.prio]
            reward += step_wait_pen
            reward_breakdown["waiting_penalty"] += step_wait_pen

            # Kiểm tra vượt quá thời gian chờ tối đa (T_wait > T_max -> Phạt nặng -100)
            if req.wait_time > req.max_wait_time:
                reward += PENALTY_MAX_WAIT_EXCEEDED
                reward_breakdown["max_wait_exceeded_penalty"] += PENALTY_MAX_WAIT_EXCEEDED
                self.expired_requests += 1
                events.append(
                    f"CẢNH BÁO: Ca #{req.req_id} ({req.prio_name}) tại nút {req.loc} "
                    f"chờ quá T_max ({req.wait_time} > {req.max_wait_time}) [-100]"
                )
                # Hủy cuộc gọi đã quá hạn tối đa để giải phóng slot
                self.pending_requests[m] = None

        # 3b. Các ca đang chờ xe di chuyển tới hiện trường (xe đang ở trạng thái TO_SCENE)
        for amb in self.ambulances:
            if amb.status == AMB_STATE_TO_SCENE and amb.active_request is not None:
                req = amb.active_request
                step_wait_pen = PENALTY_WAIT_PER_STEP[req.prio]
                reward += step_wait_pen
                reward_breakdown["waiting_penalty"] += step_wait_pen

                if req.wait_time > req.max_wait_time:
                    reward += PENALTY_MAX_WAIT_EXCEEDED
                    reward_breakdown["max_wait_exceeded_penalty"] += PENALTY_MAX_WAIT_EXCEEDED
                    self.expired_requests += 1
                    events.append(
                        f"CẢNH BÁO: Xe {amb.amb_id + 1} đến trễ quá T_max cho ca #{req.req_id} "
                        f"({req.wait_time} > {req.max_wait_time}) [-100]"
                    )
                    amb.status = AMB_STATE_IDLE
                    amb.target_node = None
                    amb.moving_to_node = None
                    amb.edge_progress = 0.0
                    amb.active_request = None

        # --------------------------------------------------------------------
        # BƯỚC 4: KIỂM TRA ĐỘ PHỦ AN TOÀN TẠI CÁC TRẠM TRỰC (MỤC 4.2.3: PHẠT -5)
        # --------------------------------------------------------------------
        # Phạt -5 khi tất cả các trạm trực đều bị bỏ trống hoàn toàn (không có xe nào đang trực tại trạm)
        stations_covered = sum(
            1
            for amb in self.ambulances
            if amb.node in self.config.station_nodes and amb.status == AMB_STATE_IDLE
        )
        if stations_covered == 0:
            reward += PENALTY_EMPTY_STATIONS
            reward_breakdown["empty_station_penalty"] += PENALTY_EMPTY_STATIONS
            self.empty_station_penalties += 1
            events.append("Mất độ phủ an toàn: Tất cả 3 trạm trực đều đang trống [-5]")

        # --------------------------------------------------------------------
        # BƯỚC 5: CẬP NHẬT ĐỘNG HỌC GIAO THÔNG & SINH CUỘC GỌI MỚI (MỤC 4.2.4)
        # --------------------------------------------------------------------
        self.step_count += 1
        self.graph.step_traffic(self.rng, self.step_count)
        new_requests = self._spawn_poisson_requests(force_at_least_one=False)
        for nreq in new_requests:
            events.append(
                f"Cuộc gọi mới #{nreq.req_id}: Mức {nreq.prio_name} tại Nút {nreq.loc} "
                f"({self.graph.nodes[nreq.loc].name})"
            )

        self.total_reward += reward
        done = self.step_count >= self.max_steps

        info = {
            "step": self.step_count,
            "action_decoded": action_info,
            "reward_breakdown": reward_breakdown,
            "events": events,
            "metrics": self.get_metrics(),
        }

        return self.get_observation(), float(reward), bool(done), info

    def _handle_ambulance_arrival(
        self,
        amb: Ambulance,
        reward_breakdown: Dict[str, float],
        events: List[str],
    ) -> float:
        """Xử lý sự kiện khi xe cứu thương vừa đặt chân tới nút đích target_node."""
        bonus = 0.0
        k = amb.amb_id

        if amb.status == AMB_STATE_TO_SCENE and amb.active_request is not None:
            req = amb.active_request
            req.response_time = req.wait_time
            self.response_times.append(req.response_time)
            amb.status = AMB_STATE_ON_SCENE
            amb.target_node = None

            if req.response_time <= req.golden_time:
                bonus += REWARD_GOLDEN_TIME
                reward_breakdown["golden_time_bonus"] += REWARD_GOLDEN_TIME
                self.golden_time_reached += 1
                events.append(
                    f"Xe {k + 1} tiếp cận ca #{req.req_id} tại nút {amb.node} trong Thời gian vàng "
                    f"(T_resp={req.response_time} <= {req.golden_time}) [+50]"
                )
            else:
                events.append(
                    f"Xe {k + 1} tiếp cận hiện trường ca #{req.req_id} tại nút {amb.node} "
                    f"(T_resp={req.response_time} > T_golden={req.golden_time})"
                )

        elif amb.status == AMB_STATE_TO_HOSPITAL:
            req_id_str = f"#{amb.active_request.req_id}" if amb.active_request else ""
            amb.status = AMB_STATE_IDLE
            amb.target_node = None
            amb.active_request = None
            bonus += REWARD_HOSPITAL_DELIVERY
            reward_breakdown["hospital_delivery_bonus"] += REWARD_HOSPITAL_DELIVERY
            self.completed_hospital_deliveries += 1
            events.append(
                f"Xe {k + 1} chuyển viện thành công ca {req_id_str} tới Bệnh viện nút {amb.node} [+20]"
            )

        elif amb.status == AMB_STATE_RELOCATING:
            amb.status = AMB_STATE_IDLE
            amb.target_node = None
            events.append(f"Xe {k + 1} hoàn tất tái định vị về Trạm trực nút {amb.node}")

        return bonus

    # ========================================================================
    # BIỂU DIỄN TRẠNG THÁI (STATE REPRESENTATIONS) CHO CÁC THUẬT TOÁN
    # ========================================================================
    def get_state_vector(self) -> np.ndarray:
        """
        Trả về vector trạng thái gốc 20 chiều đúng theo công thức Mục 4.2.1:
            S_t = [X_amb, X_req, X_traffic]
        - X_amb (6 phần tử)    : [p_1, st_1, p_2, st_2, p_3, st_3]
        - X_req (9 phần tử)    : [loc_1, prio_1, wait_1, loc_2, prio_2, wait_2, loc_3, prio_3, wait_3]
        - X_traffic (5 phần tử): [c_1, c_2, c_3, c_4, c_5]
        """
        x_amb: List[int] = []
        for amb in self.ambulances:
            x_amb.extend(amb.to_state_pair())

        x_req: List[int] = []
        for req in self.pending_requests:
            if req is None:
                x_req.extend([0, PRIO_NONE, 0])
            else:
                x_req.extend(req.to_state_triplet())

        x_traffic: List[int] = [int(c) for c in self.graph.arterial_traffic]

        return np.array(x_amb + x_req + x_traffic, dtype=np.int64)

    def get_normalized_vector(self) -> np.ndarray:
        """
        Trả về vector trạng thái thực chuẩn hóa trong đoạn [0, 1] (20 chiều)
        dành riêng cho mạng thần kinh của thuật toán PPO / DQN.
        """
        raw = self.get_state_vector().astype(np.float32)
        norm = np.zeros_like(raw, dtype=np.float32)

        # Chuẩn hóa X_amb (6 phần tử đầu)
        for k in range(self.num_ambulances):
            norm[2 * k] = raw[2 * k] / float(self.num_nodes)       # p_k / 15
            norm[2 * k + 1] = raw[2 * k + 1] / 4.0                 # st_k / 4

        # Chuẩn hóa X_req (9 phần tử tiếp theo)
        offset = 2 * self.num_ambulances
        for m in range(self.max_pending_requests):
            base = offset + 3 * m
            norm[base] = raw[base] / float(self.num_nodes)         # loc_m / 15
            norm[base + 1] = raw[base + 1] / 3.0                   # prio_m / 3
            norm[base + 2] = min(1.0, raw[base + 2] / 15.0)        # wait_time_m / 15

        # Chuẩn hóa X_traffic (5 phần tử cuối)
        t_offset = offset + 3 * self.max_pending_requests
        for e in range(self.num_arterial_roads):
            norm[t_offset + e] = raw[t_offset + e] / 2.0           # c_e / 2

        return norm

    def get_tabular_state(self) -> Tuple[int, ...]:
        """
        Trả về trạng thái rời rạc thu gọn (Compact Discrete State Tuple) dành cho
        các thuật toán dạng bảng: Dynamic Programming, Monte Carlo, Tabular Q-Learning.

        Cấu trúc tuple gồm 7 thành phần:
            (avail_1, avail_2, avail_3, urgent_slot_plus1, urgent_prio, nearest_idle_amb, traffic_level)
        Tổng số trạng thái rút gọn: 2^3 * 4 * 4 * 4 * 3 = 1,536 trạng thái (rất phù hợp để hội tụ bảng Q/V).
        """
        avail_tuple = tuple(1 if amb.can_be_dispatched else 0 for amb in self.ambulances)

        # Tìm cuộc gọi đang chờ có mức độ ưu tiên cao nhất (prio nhỏ nhất: 1=Đỏ > 2=Vàng > 3=Xanh)
        best_slot = -1
        best_prio = PRIO_NONE
        best_wait = -1
        for m, req in enumerate(self.pending_requests):
            if req is not None:
                if best_prio == PRIO_NONE or req.prio < best_prio or (
                    req.prio == best_prio and req.wait_time > best_wait
                ):
                    best_slot = m
                    best_prio = req.prio
                    best_wait = req.wait_time

        # Tìm xe rảnh gần nhất với ca cấp cứu ưu tiên nhất
        nearest_amb = 3  # 3 nghĩa là không có xe rảnh hoặc không có ca cấp cứu
        if best_slot != -1:
            req_loc = self.pending_requests[best_slot].loc  # type: ignore[union-attr]
            best_dist = float("inf")
            for k, amb in enumerate(self.ambulances):
                if amb.can_be_dispatched:
                    _, cost = self.graph.shortest_path(amb.node, req_loc, consider_traffic=True)
                    if cost < best_dist:
                        best_dist = cost
                        nearest_amb = k

        # Mức độ ùn tắc chung trên 5 tuyến huyết mạch: 0 (Thông thoáng), 1 (Trung bình), 2 (Cao)
        mean_cong = float(np.mean(self.graph.arterial_traffic))
        if mean_cong < 0.6:
            traffic_level = 0
        elif mean_cong < 1.3:
            traffic_level = 1
        else:
            traffic_level = 2

        return (
            avail_tuple[0],
            avail_tuple[1],
            avail_tuple[2],
            best_slot + 1,
            best_prio,
            nearest_amb,
            traffic_level,
        )

    def get_observation(self) -> Dict[str, Any]:
        """Trả về dictionary chứa đầy đủ mọi biểu diễn trạng thái và mặt nạ hành động."""
        state_vec = self.get_state_vector()
        return {
            "state_vector": state_vec,
            "X_amb": state_vec[: 2 * self.num_ambulances].copy(),
            "X_req": state_vec[
                2 * self.num_ambulances : 2 * self.num_ambulances + 3 * self.max_pending_requests
            ].copy(),
            "X_traffic": state_vec[-self.num_arterial_roads :].copy(),
            "normalized_vector": self.get_normalized_vector(),
            "tabular_state": self.get_tabular_state(),
            "action_mask": self.get_action_mask(),
            "valid_actions": self.get_valid_actions(),
            "step": self.step_count,
        }

    # ========================================================================
    # CHÍNH SÁCH BASELINE (NEAREST AMBULANCE + PRIORITY TRIAGE)
    # ========================================================================
    def recommend_baseline_action(self) -> int:
        """
        Thuật toán Baseline truyền thống (Mục 3.4):
        - Ưu tiên phục vụ cuộc gọi có mức độ khẩn cấp cao nhất (Đỏ -> Vàng -> Xanh) và chờ lâu nhất.
        - Chọn xe cứu thương đang rảnh có thời gian di chuyển ngắn nhất đến hiện trường.
        - Nếu không có cuộc gọi chờ hoặc không có xe rảnh: chọn a_standby (24).
        """
        best_slot: Optional[int] = None
        best_req: Optional[EmergencyRequest] = None

        for m, req in enumerate(self.pending_requests):
            if req is None:
                continue
            if (
                best_req is None
                or req.prio < best_req.prio
                or (req.prio == best_req.prio and req.wait_time > best_req.wait_time)
            ):
                best_req = req
                best_slot = m

        if best_req is not None and best_slot is not None:
            best_amb: Optional[int] = None
            best_cost = float("inf")
            for k, amb in enumerate(self.ambulances):
                if amb.can_be_dispatched:
                    _, cost = self.graph.shortest_path(amb.node, best_req.loc, consider_traffic=True)
                    if cost < best_cost:
                        best_cost = cost
                        best_amb = k

            if best_amb is not None:
                return self.encode_dispatch_action(best_amb, best_slot)

        return self.STANDBY_ACTION

    # ========================================================================
    # THỐNG KÊ ĐÁNH GIÁ & HIỂN THỊ TRạng THÁI (MỤC 3.5)
    # ========================================================================
    def get_metrics(self) -> Dict[str, Any]:
        """Tổng hợp các chỉ số đánh giá hiệu năng (Mục 3.5)."""
        avg_resp = float(np.mean(self.response_times)) if self.response_times else 0.0
        approached_count = len(self.response_times)
        survival_rate = (
            (self.golden_time_reached / approached_count) * 100.0 if approached_count > 0 else 0.0
        )
        completion_rate = (
            (self.completed_hospital_deliveries / self.total_spawned_requests) * 100.0
            if self.total_spawned_requests > 0
            else 0.0
        )
        return {
            "total_reward": round(self.total_reward, 2),
            "total_spawned_requests": self.total_spawned_requests,
            "dispatched_requests": self.dispatched_requests,
            "approached_requests": approached_count,
            "golden_time_reached": self.golden_time_reached,
            "completed_hospital_deliveries": self.completed_hospital_deliveries,
            "expired_requests": self.expired_requests,
            "avg_response_time": round(avg_resp, 3),
            "golden_time_rate_pct": round(survival_rate, 2),
            "completion_rate_pct": round(completion_rate, 2),
            "congested_road_steps": self.congested_road_steps,
            "empty_station_penalties": self.empty_station_penalties,
            "invalid_actions": self.invalid_actions,
        }

    def render(self) -> str:
        """Hiển thị trực quan trạng thái hệ thống đô thị tại bước hiện tại."""
        lines = [
            f"================ BƯỚC THỜI GIAN t = {self.step_count}/{self.max_steps} ================",
            f"1. Vector trạng thái S_t (20 chiều): {self.get_state_vector().tolist()}",
            "2. Đội xe cứu thương (X_amb):",
        ]
        for amb in self.ambulances:
            target_str = f" -> Đích: Nút {amb.target_node}" if amb.target_node else ""
            req_str = f" (Ca #{amb.active_request.req_id})" if amb.active_request else ""
            lines.append(
                f"   - Xe {amb.amb_id + 1}: Nút {amb.node:2d} ({self.graph.nodes[amb.node].name}) | "
                f"Trạng thái st={amb.status} [{amb.status_name}]{target_str}{req_str}"
            )

        lines.append("3. Danh sách cuộc gọi đang chờ (X_req - tối đa 3 ca):")
        for m, req in enumerate(self.pending_requests):
            if req is None:
                lines.append(f"   - Slot {m + 1}: [Trống] (loc=0, prio=0, wait=0)")
            else:
                lines.append(
                    f"   - Slot {m + 1}: Ca #{req.req_id} tại Nút {req.loc:2d} ({self.graph.nodes[req.loc].name}) | "
                    f"Mức {req.prio} [{req.prio_name}] | Đã chờ: {req.wait_time}/{req.max_wait_time} bước "
                    f"(T_golden={req.golden_time})"
                )

        lines.append("4. Tình trạng 5 tuyến đường huyết mạch (X_traffic):")
        for idx, edge in enumerate(self.graph.arterial_edges):
            c_val = int(self.graph.arterial_traffic[idx])
            lines.append(
                f"   - Tuyến {idx + 1} (Nút {edge.u:2d} <-> Nút {edge.v:2d}): "
                f"c_{idx + 1} = {c_val} [{TRAFFIC_NAMES[c_val]}]"
            )

        text = "\n".join(lines)
        try:
            print(text)
        except UnicodeEncodeError:
            import sys
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            print(text)
        return text
