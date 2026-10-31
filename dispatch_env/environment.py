"""
Môi trường Mô phỏng Điều phối Xe Cứu thương (Ambulance Dispatch Environment).
Quy trình mô phỏng từng phút [t, t+1) với MDP step,
sinh cuộc gọi Poisson, giới hạn tối đa 3 cuộc gọi yêu cầu xử lý,
và ghi log chi tiết theo định dạng yêu cầu.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Tuple, Optional, Any, Union, Sequence, Set
import numpy as np

from map_env.city_map import CityMap
from dispatch_env.types import (
    AmbulanceStatus,
    CallStatus,
    DEFAULT_NUM_AMBULANCES,
    DEFAULT_MAX_TIME,
    DEFAULT_MAX_OVERTIME,
    DEFAULT_POISSON_LAMBDA,
    DEFAULT_MAX_PENDING_CALLS,
    DEFAULT_SEED,
    SEVERITY_CONFIG,
    REWARD_CONFIG,
    DISTANCE_PENALTY_PER_KM,
)
from dispatch_env.call import EmergencyCall
from dispatch_env.ambulance import Ambulance


class AmbulanceDispatchEnv:
    """
    Môi trường Điều phối Xe Cứu thương trên mạng lưới 80 nút và 4 bệnh viện.
    Các tham số mặc định đồng bộ hoàn toàn với run_simulation.py:
    - num_ambulances = 6
    - max_time = 360
    - max_overtime = 300
    - poisson_lambda = 0.1
    - max_pending_calls = 3
    - seed = 42
    """
    def __init__(
        self,
        city_map: Optional[CityMap] = None,
        num_ambulances: int = DEFAULT_NUM_AMBULANCES,
        max_time: int = DEFAULT_MAX_TIME,
        max_overtime: int = DEFAULT_MAX_OVERTIME,
        poisson_lambda: float = DEFAULT_POISSON_LAMBDA,
        max_pending_calls: int = DEFAULT_MAX_PENDING_CALLS,
        seed: Optional[int] = DEFAULT_SEED,
    ) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.city_map = city_map if city_map is not None else CityMap(seed=seed)

        self.num_ambulances = num_ambulances
        self.max_time = max_time
        self.max_overtime = max_overtime  # Số phút tối đa khi vượt quá thời gian mô phỏng là 300p
        self.poisson_lambda = poisson_lambda
        self.max_pending_calls = max_pending_calls
        self.episode_count: int = 0

        # Thời gian mô phỏng (phút) và bước thời gian giao thông chạy song song
        self.current_minute: int = 0
        self.traffic_time_step: int = 0
        self.edge_keys: List[Tuple[int, int]] = sorted(self.city_map.edges.keys())
        self.next_call_id: int = 1
        self.is_truncated: bool = False

        # Danh sách xe cứu thương: id -> Ambulance
        self.ambulances: Dict[int, Ambulance] = {}

        # Danh bạ toàn bộ các cuộc gọi đã phát sinh: id -> EmergencyCall
        self.calls: Dict[int, EmergencyCall] = {}

        # Danh sách cuộc gọi mới sinh ra trong phút vừa qua (đầu phút)
        self.calls_generated_this_step: List[EmergencyCall] = []

        # Danh sách các cuộc gọi bị thất bại CHỈ trong phút này (để ghi log 1 lần duy nhất)
        self.calls_failed_this_step: List[EmergencyCall] = []

        # Danh sách các cuộc gọi vừa hoàn thành trong bước này (C_t)
        self.calls_completed_this_step: List[EmergencyCall] = []

        # Thống kê tổng kết
        self.served_calls_count: int = 0
        self.failed_calls_count: int = 0
        self.perfect_calls_count: int = 0
        self.last_step_reward: float = 0.0
        self.total_reward: float = 0.0
        self.last_reward_breakdown: Dict[str, float] = {
            "wait_penalty": 0.0,
            "failure_penalty": 0.0,
            "distance_penalty": 0.0,
            "completion_bonus": 0.0,
            "total": 0.0,
        }

        # Xây dựng tập tất cả các cách ghép hành động tiềm năng (gồm cả không phân công)
        # Mỗi phần tử là 1 tuple các cặp (amb_id, slot_idx) đã chuẩn hóa theo amb_id tăng dần
        self.potential_actions: List[Tuple[Tuple[int, int], ...]] = self._build_potential_actions()
        self.num_potential_actions: int = len(self.potential_actions)
        self.slot_action_to_id: Dict[Tuple[Tuple[int, int], ...], int] = {
            act: idx for idx, act in enumerate(self.potential_actions)
        }

        # Lưu thông tin hành động của bước vừa thực hiện (để ghi log và theo dõi)
        self.last_action_id: int = 0
        self.last_decoded_action: List[Tuple[int, int]] = []
        self.last_valid_action_count: int = 1
        self.last_slot_summary: str = ""

        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        generate_initial_calls: bool = True,
    ) -> Dict[str, Any]:
        """
        Khởi tạo lại môi trường cho một episode mới:
        - Nếu truyền seed != None: đặt lại toàn bộ bộ sinh số ngẫu nhiên theo seed đó (tái lập kết quả).
        - Nếu seed == None (chạy qua nhiều episode liên tiếp):
          + Luồng sinh số ngẫu nhiên (self.rng và self.city_map.rng) tiếp tục tiến lên tự nhiên.
          + Tái khởi tạo ngẫu nhiên độ tắc đường ban đầu trên toàn bộ 176 cung đường (reset_traffic).
          + Sinh mới ngẫu nhiên vị trí ban đầu (cung đường & khoảng cách) cho tất cả các xe cứu thương.
          + Sinh mới ngẫu nhiên danh sách cuộc gọi ban đầu tại t=0 (và chuỗi cuộc gọi Poisson trong suốt episode).
          => Đảm bảo mỗi episode có kịch bản giao thông, vị trí xe và cuộc gọi hoàn toàn đa dạng, khác nhau.
        """
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            self.city_map.reset_traffic(seed=seed)
        elif self.episode_count >= 1:
            # Sang các episode tiếp theo: sinh hạt giống con mới từ luồng rng đang tiến để làm mới giao thông bản đồ
            episode_map_seed = int(self.rng.integers(0, 2**31 - 1))
            self.city_map.reset_traffic(seed=episode_map_seed)
        else:
            # Episode đầu tiên ngay khi khởi tạo __init__: đảm bảo đồng bộ bộ đếm thời gian giao thông về 0
            self.city_map.time_step = 0
            self.city_map.traffic_time_step = 0

        self.episode_count += 1
        self.current_minute = 0
        self.traffic_time_step = 0
        self.edge_keys = sorted(self.city_map.edges.keys())
        self.next_call_id = 1
        self.calls.clear()
        self.calls_generated_this_step.clear()
        self.calls_failed_this_step.clear()
        self.calls_completed_this_step.clear()
        self.served_calls_count = 0
        self.failed_calls_count = 0
        self.perfect_calls_count = 0
        self.last_step_reward = 0.0
        self.total_reward = 0.0
        self.last_reward_breakdown = {
            "wait_penalty": 0.0,
            "failure_penalty": 0.0,
            "distance_penalty": 0.0,
            "completion_bonus": 0.0,
            "total": 0.0,
        }
        self.is_truncated = False

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

        # Tại t = 0: Cuộc gọi có thể sinh ra vào đầu phút
        if generate_initial_calls:
            self.calls_generated_this_step = self._generate_new_calls()
            # "Tại t=0 các cuộc gọi thất bại vẫn bị trừ điểm thưởng như thường"
            if self.calls_failed_this_step:
                initial_fail_penalty = sum(
                    REWARD_CONFIG[c.severity]["fail_penalty"]
                    for c in self.calls_failed_this_step
                )
                self.last_step_reward = -initial_fail_penalty
                self.total_reward = -initial_fail_penalty
                self.last_reward_breakdown = {
                    "wait_penalty": 0.0,
                    "failure_penalty": initial_fail_penalty,
                    "distance_penalty": 0.0,
                    "completion_bonus": 0.0,
                    "total": -initial_fail_penalty,
                }

        self.last_action_id = 0
        self.last_decoded_action = []
        self.last_valid_action_count = int(np.sum(self.get_action_mask()))
        self.last_slot_summary = self.format_call_slots()

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

    # =========================================================================
    # QUẢN LÝ SLOT CUỘC GỌI, TẬP HÀNH ĐỘNG TIỀM NĂNG, ACTION MASK & GIẢI MÃ
    # =========================================================================
    def get_call_slots(self) -> List[Optional[EmergencyCall]]:
        """
        Sắp xếp các cuộc gọi đang PENDING vào hệ thống Slot cố định (Slot 0, Slot 1, ..., Slot M-1).
        Quy tắc sắp xếp Slot thống nhất (Canonical FIFO Slot Rule):
        - Chỉ xét các cuộc gọi đang ở trạng thái PENDING ('Yêu cầu xử lý').
        - Sắp xếp tăng dần theo (created_time, call_id) (cuộc gọi xuất hiện trước xếp vào slot nhỏ hơn;
          nếu cùng thời điểm phát sinh thì mã cuộc gọi nhỏ hơn xếp trước).
        - Đưa lần lượt vào Slot 0, Slot 1, ..., Slot (max_pending_calls - 1).
        - Nếu số cuộc gọi PENDING ít hơn số slot tối đa, các slot còn lại mang giá trị None (Trống).
        """
        sorted_pending = sorted(
            self.pending_calls,
            key=lambda c: (c.created_time, c.call_id),
        )
        slots: List[Optional[EmergencyCall]] = [None] * self.max_pending_calls
        for idx, call in enumerate(sorted_pending[: self.max_pending_calls]):
            slots[idx] = call
        return slots

    def format_call_slots(self, slots: Optional[List[Optional[EmergencyCall]]] = None) -> str:
        """Định dạng chuỗi hiển thị trạng thái các Slot cuộc gọi hiện tại."""
        if slots is None:
            slots = self.get_call_slots()
        parts: List[str] = []
        for idx, c in enumerate(slots):
            if c is not None:
                parts.append(f"Slot {idx}: #{c.call_id} (Mức {c.severity}, chờ {c.wait_time:.1f}p)")
            else:
                parts.append(f"Slot {idx}: Trống")
        return " | ".join(parts)

    def _build_potential_actions(self) -> List[Tuple[Tuple[int, int], ...]]:
        """
        Tạo tất cả các cách ghép hành động tiềm năng (joint slot assignments) giữa
        N xe cứu thương (1..N) và M slot cuộc gọi (0..M-1), bao gồm cả không phân công ().
        - Mỗi xe nhận tối đa 1 slot, mỗi slot giao tối đa cho 1 xe.
        - Các cặp (amb_id, slot_idx) trong mỗi hành động được sắp xếp tăng dần theo amb_id.
        - Với 4 xe và 3 slot:
          + k = 0 (không phân công): 1 cách -> ()
          + k = 1 (phân công 1 xe): C(4,1) * P(3,1) = 12 cách
          + k = 2 (phân công đồng thời 2 xe): C(4,2) * P(3,2) = 36 cách
          + k = 3 (phân công đồng thời 3 xe): C(4,3) * P(3,3) = 24 cách
          => Tổng cộng = 73 cách ghép hành động tiềm năng.
        """
        amb_ids = list(range(1, self.num_ambulances + 1))
        slot_indices = list(range(self.max_pending_calls))
        max_k = min(self.num_ambulances, self.max_pending_calls)

        potential: List[Tuple[Tuple[int, int], ...]] = [()]  # Action 0: Không phân công ()
        for k in range(1, max_k + 1):
            for chosen_ambs in itertools.combinations(amb_ids, k):
                for chosen_slots in itertools.permutations(slot_indices, k):
                    joint_action = tuple(zip(chosen_ambs, chosen_slots))
                    potential.append(joint_action)
        return potential

    def get_action_mask(self) -> np.ndarray:
        """
        Xây dựng Action Mask (mảng boolean kích thước 73) để kiểm tra tính hợp lệ
        của từng cách ghép hành động tiềm năng trong trạng thái hiện tại S_t:
        - Action 0 (không phân công) luôn hợp lệ (mask[0] = True).
        - Một hành động ghép ((amb_1, slot_1), (amb_2, slot_2), ...) hợp lệ (True) khi và chỉ khi:
          1. Mọi xe amb_i trong tập đều đang RẢNH (status == IDLE và current_call_id is None).
          2. Mọi slot_i trong tập đều đang chứa cuộc gọi PENDING (slots[slot_i] is not None).
        """
        slots = self.get_call_slots()
        idle_amb_set: Set[int] = {
            a.ambulance_id
            for a in self.ambulances.values()
            if a.status == AmbulanceStatus.IDLE and a.current_call_id is None
        }
        valid_slot_set: Set[int] = {
            idx
            for idx, call in enumerate(slots)
            if call is not None and call.status == CallStatus.PENDING
        }

        mask = np.zeros(len(self.potential_actions), dtype=bool)
        for action_idx, joint_slot_action in enumerate(self.potential_actions):
            is_valid = True
            for amb_id, slot_idx in joint_slot_action:
                if amb_id not in idle_amb_set or slot_idx not in valid_slot_set:
                    is_valid = False
                    break
            mask[action_idx] = is_valid
        return mask

    def is_action_valid(
        self,
        action: Union[int, Tuple[int, int], Sequence[Tuple[int, int]]],
        from_slots: bool = False,
    ) -> bool:
        """
        Kiểm tra tính hợp lệ của một hành động cụ thể ở trạng thái hiện tại:
        - Nếu action là int (action_id trong [0, 72]): kiểm tra trực tiếp qua get_action_mask()[action].
        - Nếu action là tập các cặp (amb_id, slot_idx) với from_slots=True: kiểm tra qua slot.
        - Nếu action là tập các cặp (amb_id, call_id) (như [(1, 10), (3, 12)]):
          kiểm tra không trùng xe, không trùng cuộc gọi, mọi xe đều IDLE và mọi cuộc gọi đều PENDING.
        """
        if isinstance(action, (int, np.integer)):
            act_id = int(action)
            if act_id < 0 or act_id >= len(self.potential_actions):
                return False
            return bool(self.get_action_mask()[act_id])

        # Chuẩn hóa về danh sách các cặp (int, int)
        if (
            isinstance(action, tuple)
            and len(action) == 2
            and isinstance(action[0], (int, np.integer))
            and isinstance(action[1], (int, np.integer))
        ):
            pairs: List[Tuple[int, int]] = [(int(action[0]), int(action[1]))]
        else:
            pairs = [(int(p[0]), int(p[1])) for p in action]  # type: ignore[arg-type]

        seen_ambs: Set[int] = set()
        seen_targets: Set[int] = set()
        slots = self.get_call_slots() if from_slots else []

        for amb_id, target in pairs:
            if amb_id in seen_ambs or target in seen_targets:
                return False
            amb = self.ambulances.get(amb_id)
            if amb is None or amb.status != AmbulanceStatus.IDLE or amb.current_call_id is not None:
                return False

            if from_slots:
                if target < 0 or target >= len(slots) or slots[target] is None:
                    return False
                if slots[target].status != CallStatus.PENDING:
                    return False
            else:
                call = self.calls.get(target)
                if call is None or call.status != CallStatus.PENDING:
                    return False

            seen_ambs.add(amb_id)
            seen_targets.add(target)

        return True

    def decode_action(
        self,
        action: Optional[Union[int, Tuple[int, int], Sequence[Tuple[int, int]]]],
        from_slots: bool = False,
    ) -> List[Tuple[int, int]]:
        """
        Giải mã hành động (Action Decoder) về tập các cặp phân công đồng thời hợp lệ
        dưới dạng [(amb_id, call_id), ...] (ví dụ [(1, 10), (3, 12)] nghĩa là Xe 1 nhận cuộc gọi 10,
        Xe 3 nhận cuộc gọi 12):
        - Nếu action là None: trả về [] (không phân công).
        - Nếu action là int (chỉ số action_id trong [0, 72] của self.potential_actions):
          Lấy cách ghép theo slot self.potential_actions[action_id] = ((amb_id, slot_idx), ...),
          ánh xạ từng slot_idx sang call_id thực tế trong get_call_slots() và lọc các cặp hợp lệ.
        - Nếu action là tập các cặp (amb_id, slot_idx) với from_slots=True:
          Ánh xạ slot_idx -> call_id và lọc các cặp hợp lệ 1-1.
        - Nếu action là tập các cặp (amb_id, call_id) (ví dụ [(1, 10), (3, 12)] hoặc (1, 10)):
          Kiểm tra và lọc các cặp hợp lệ 1-1 (xe IDLE, cuộc gọi PENDING).
        """
        if action is None:
            return []

        slots = self.get_call_slots()

        # Trường hợp 1: action là chỉ số nguyên action_id trong [0, num_potential_actions - 1]
        if isinstance(action, (int, np.integer)):
            act_id = int(action)
            if act_id < 0 or act_id >= len(self.potential_actions):
                return []
            slot_pairs = self.potential_actions[act_id]
            from_slots = True
        # Trường hợp 2: action là 1 cặp đơn lẻ (amb_id, target_id)
        elif (
            isinstance(action, tuple)
            and len(action) == 2
            and isinstance(action[0], (int, np.integer))
            and isinstance(action[1], (int, np.integer))
        ):
            slot_pairs = ((int(action[0]), int(action[1])),)
        # Trường hợp 3: action là danh sách/tập các cặp [(amb_id, target_id), ...]
        else:
            slot_pairs = tuple((int(p[0]), int(p[1])) for p in action)  # type: ignore[arg-type]

        decoded_pairs: List[Tuple[int, int]] = []
        assigned_ambs: Set[int] = set()
        assigned_calls: Set[int] = set()

        for amb_id, target_val in slot_pairs:
            if amb_id in assigned_ambs:
                continue
            amb = self.ambulances.get(amb_id)
            if amb is None or amb.status != AmbulanceStatus.IDLE or amb.current_call_id is not None:
                continue

            if from_slots:
                if target_val < 0 or target_val >= len(slots):
                    continue
                call = slots[target_val]
            else:
                call = self.calls.get(target_val)

            if call is None or call.status != CallStatus.PENDING:
                continue
            if call.call_id in assigned_calls:
                continue

            decoded_pairs.append((amb_id, call.call_id))
            assigned_ambs.add(amb_id)
            assigned_calls.add(call.call_id)

        return decoded_pairs

    def encode_action(
        self,
        dispatch_pairs: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
    ) -> int:
        """
        Mã hóa một tập các cặp phân công đồng thời [(amb_id, call_id), ...]
        sang chỉ số hành động tiềm năng action_id trong [0, 72] dựa trên vị trí Slot hiện tại.
        Nếu không phân công hoặc không khớp slot hợp lệ, trả về 0 (hành động không phân công).
        """
        if not dispatch_pairs:
            return 0

        if (
            isinstance(dispatch_pairs, tuple)
            and len(dispatch_pairs) == 2
            and isinstance(dispatch_pairs[0], (int, np.integer))
            and isinstance(dispatch_pairs[1], (int, np.integer))
        ):
            pairs = [(int(dispatch_pairs[0]), int(dispatch_pairs[1]))]
        else:
            pairs = [(int(p[0]), int(p[1])) for p in dispatch_pairs]  # type: ignore[arg-type]

        slots = self.get_call_slots()
        call_id_to_slot: Dict[int, int] = {
            c.call_id: idx for idx, c in enumerate(slots) if c is not None
        }

        slot_pairs: List[Tuple[int, int]] = []
        used_ambs: Set[int] = set()
        used_slots: Set[int] = set()

        for amb_id, call_id in sorted(pairs, key=lambda x: x[0]):
            if amb_id in used_ambs or call_id not in call_id_to_slot:
                continue
            slot_idx = call_id_to_slot[call_id]
            if slot_idx in used_slots:
                continue
            slot_pairs.append((amb_id, slot_idx))
            used_ambs.add(amb_id)
            used_slots.add(slot_idx)

        canonical_key = tuple(slot_pairs)
        return self.slot_action_to_id.get(canonical_key, 0)

    def get_valid_action_ids(self) -> List[int]:
        """Trả về danh sách các chỉ số action_id hợp lệ (mask[action_id] == True) ở trạng thái hiện tại."""
        mask = self.get_action_mask()
        return [int(idx) for idx, is_valid in enumerate(mask) if is_valid]

    def get_valid_actions(self) -> List[List[Tuple[int, int]]]:
        """
        Trả về tất cả các tập hành động phân công đồng thời hợp lệ ở trạng thái hiện tại
        (đã được giải mã từ các cách ghép slot hợp lệ sang dạng [(amb_id, call_id), ...]),
        bao gồm cả hành động không phân công [].
        """
        return [self.decode_action(act_id) for act_id in self.get_valid_action_ids()]

    def get_observation(self) -> Dict[str, Any]:
        """Quan sát trạng thái hiện tại S_t của môi trường (kèm thông tin Slot, Edge ID và Action Mask)."""
        amb_obs = {
            aid: amb.to_dict(self.city_map)
            for aid, amb in self.ambulances.items()
        }

        slots = self.get_call_slots()
        slot_obs: List[Optional[Dict[str, Any]]] = []
        for slot_idx, c in enumerate(slots):
            if c is not None:
                c_edge_id = c.edge_id if c.edge_id != 0 else self.city_map.edge_to_id.get(c.edge, 0)
                c_dict = c.to_dict(edge_id=c_edge_id)
                c_dict["slot"] = slot_idx
                slot_obs.append(c_dict)
            else:
                slot_obs.append(None)

        call_obs = [
            c.to_dict(edge_id=(c.edge_id if c.edge_id != 0 else self.city_map.edge_to_id.get(c.edge, 0)))
            for c in self.active_calls
        ]

        # Tình trạng tắc đường trên từng cung đường có hướng (u -> v) và theo edge_id (1..176)
        edge_congestion: Dict[Tuple[int, int], int] = {}
        edge_congestion_by_id: Dict[int, int] = {}
        edge_obs: Dict[Tuple[int, int], Dict[str, Any]] = {}
        edge_obs_by_id: Dict[int, Dict[str, Any]] = {}

        for ek in self.edge_keys:
            edge = self.city_map.edges[ek]
            eid = edge.edge_id
            edge_congestion[ek] = edge.congestion
            edge_congestion_by_id[eid] = edge.congestion

            e_info = {
                "edge_id": eid,
                "id": eid,
                "edge": ek,
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "congestion": edge.congestion,
                "congestion_name": edge.congestion_name,
                "speed": edge.speed,
                "length": edge.length,
                "travel_time_minutes": round(edge.travel_time_minutes, 3),
                "is_center": edge.is_center,
            }
            edge_obs[ek] = e_info
            edge_obs_by_id[eid] = e_info

        edge_congestion_vector = np.array(
            [edge_congestion[ek] for ek in self.edge_keys],
            dtype=np.int32,
        )

        return {
            "episode": self.episode_count,
            "current_minute": self.current_minute,
            "traffic_time_step": self.traffic_time_step,
            "edge_congestion_vector": edge_congestion_vector,
            "call_slots": slot_obs,
            "action_mask": self.get_action_mask(),
            "ambulances": amb_obs,
            "edges": edge_obs,
            "edges_by_id": edge_obs_by_id,
            "edge_congestion": edge_congestion,
            "edge_congestion_by_id": edge_congestion_by_id,
            "edge_id_to_nodes": dict(self.city_map.id_to_edge),
            "nodes_to_edge_id": dict(self.city_map.edge_to_id),
            "pending_call_count": len(self.pending_calls),
            "active_calls": call_obs,
            "served_calls": self.served_calls_count,
            "failed_calls": self.failed_calls_count,
            "perfect_calls": self.perfect_calls_count,
            "total_reward": self.total_reward,
            "last_step_reward": self.last_step_reward,
        }

    def penalize_call_failure(self, call: EmergencyCall) -> float:
        """
        Trừ điểm thưởng cho một cuộc gọi bị thất bại (nếu chưa bị trừ).
        Đảm bảo tại bất kỳ thời điểm nào (kể cả tại t=0), cuộc gọi thất bại
        đều bị trừ điểm thưởng chuẩn xác theo bảng cấu hình REWARD_CONFIG.
        """
        if call.failure_penalized:
            return 0.0
        call.failure_penalized = True
        if call.status != CallStatus.FAILED:
            call.status = CallStatus.FAILED
        self.failed_calls_count += 1
        if call not in self.calls_failed_this_step:
            self.calls_failed_this_step.append(call)
        penalty = REWARD_CONFIG[call.severity]["fail_penalty"]
        self.last_step_reward -= penalty
        self.total_reward -= penalty
        self.last_reward_breakdown["failure_penalty"] += penalty
        self.last_reward_breakdown["total"] -= penalty
        return penalty

    def _generate_new_calls(self) -> List[EmergencyCall]:
        """
        Sinh cuộc gọi mới theo phân phối Poisson (k ~ Poisson(poisson_lambda), mặc định lambda = 0.1).
        - Nếu thời gian >= max_time (360 phút): không sinh thêm cuộc gọi.
        - Giới hạn: Chỉ cho phép tối đa max_pending_calls (3) cuộc gọi trong trạng thái 'Yêu cầu xử lý' (PENDING).
        - Cuộc gọi vượt quá bị từ chối và tính là Thất bại.
        """
        if self.current_minute >= self.max_time:
            return []

        # Số cuộc gọi sinh ra theo Poisson(self.poisson_lambda)
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
                edge_id=edge.edge_id,
            )
            self.next_call_id += 1

            # Kiểm tra giới hạn 3 cuộc gọi PENDING
            current_pending = [c for c in self.calls.values() if c.status == CallStatus.PENDING]
            if len(current_pending) >= self.max_pending_calls:
                # Bị từ chối ngay lập tức -> Thất bại
                call.mark_rejected()
                call.failure_penalized = True
                self.failed_calls_count += 1
                self.calls_failed_this_step.append(call)
            else:
                call.status = CallStatus.PENDING

            self.calls[call.call_id] = call
            new_calls.append(call)

        return new_calls

    def step(
        self,
        action: Optional[Union[int, Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
        auto_dispatch: bool = False,
        from_slots: bool = False,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        r"""
        Thực hiện một bước mô phỏng 1 phút: [t, t+1).
        
        Quy trình chuẩn theo yêu cầu:
        1. Quan sát trạng thái S_t và sắp xếp các cuộc gọi PENDING vào Slot (0, 1, 2).
        2. Xây dựng Action Mask kiểm tra tính hợp lệ trên tập 73 cách ghép hành động tiềm năng.
        3. Agent chọn hành động A_t:
           - Có thể là chỉ số action_id \in [0, 72] (tương ứng cách ghép slot tiềm năng),
           - Hoặc tập các cặp phân công đồng thời [(1, 10), (3, 12)] (Xe 1 nhận cuộc gọi ID 10, Xe 3 nhận ID 12).
        4. Giải mã hành động (decode_action) và thực hiện phân công đồng thời cho các xe nhận nhiệm vụ.
        5. Trong khoảng [t, t+1):
           - Các xe di chuyển theo tuyến hiện tại (di chuyển liên tục sub-minute).
           - Xử lý sự kiện đến hiện trường, chăm sóc sơ cứu, đến BV và bàn giao.
           - Cập nhật thời gian chờ thực sự \Delta W_{c,t} và kiểm tra mốc thời gian thất bại.
           - Nếu cuộc gọi thất bại trong khi xe đang tiếp cận, xe lập tức về rảnh!
        6. Tại t+1:
           - Cập nhật trạng thái giao thông.
           - Tính toán lại tuyến đường (reroute) nếu có thay đổi.
           - Sinh cuộc gọi mới (Poisson).
           - Kiểm tra giới hạn 3 cuộc gọi PENDING.
        7. Tính phần thưởng r_t theo công thức:
           r_t = - \sum_c w_c \Delta W_{c,t} - \sum_{c \in F_t} p_c - 0.05 \sum_i \Delta d_{i,t} + \sum_{c \in C_t} b_c
        8. Trả về trạng thái mới S_{t+1}.
        """
        t0 = float(self.current_minute)
        self.calls_failed_this_step.clear()
        self.calls_completed_this_step.clear()

        # Ghi nhận trạng thái Slot và Action Mask ở đầu phút t0 trước khi phân công
        self.last_slot_summary = self.format_call_slots()
        mask_at_t0 = self.get_action_mask()
        self.last_valid_action_count = int(np.sum(mask_at_t0))

        # --------------------------------------------------------------------
        # 3 & 4. GIẢI MÃ VÀ XỬ LÝ TẬP HÀNH ĐỘNG PHÂN CÔNG ĐỒNG THỜI
        # --------------------------------------------------------------------
        valid_dispatch_pairs: List[Tuple[int, int]] = []
        if action is not None:
            # Giải mã hành động (hỗ trợ cả action_id nguyên [0..72], cặp slot, hoặc tập cặp [(amb_id, call_id), ...])
            valid_dispatch_pairs = self.decode_action(action, from_slots=from_slots)
        elif auto_dispatch:
            # Chọn tập các cặp phân công đồng thời tối ưu (ưu tiên mức độ nghiêm trọng cao & chờ lâu)
            idle_ambs = [
                a for a in self.ambulances.values()
                if a.status == AmbulanceStatus.IDLE and a.current_call_id is None
            ]
            sorted_pending = sorted(
                self.pending_calls,
                key=lambda c: (-c.severity, c.created_time, c.call_id),
            )
            candidate_pairs: List[Tuple[int, int]] = []
            for c in sorted_pending:
                if not idle_ambs:
                    break
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
                    candidate_pairs.append((best_amb.ambulance_id, c.call_id))
                    idle_ambs.remove(best_amb)

            # Mã hóa sang action_id trên Slot rồi giải mã ngược lại để đảm bảo thống nhất 100% qua cơ chế Slot & Mask
            act_id = self.encode_action(candidate_pairs)
            valid_dispatch_pairs = self.decode_action(act_id)

        # Lưu thông tin hành động đồng thời đã giải mã của bước này
        self.last_action_id = self.encode_action(valid_dispatch_pairs)
        self.last_decoded_action = sorted(valid_dispatch_pairs, key=lambda x: x[0])

        # Thực hiện đồng thời tất cả các cặp phân công hợp lệ (amb_id, call_id) trong tập hành động
        for aid, cid in self.last_decoded_action:
            amb = self.ambulances.get(aid)
            call = self.calls.get(cid)
            if (
                amb is not None
                and call is not None
                and amb.status == AmbulanceStatus.IDLE
                and amb.current_call_id is None
                and call.status == CallStatus.PENDING
            ):
                amb.dispatch_to_call(call, self.city_map)

        # --------------------------------------------------------------------
        # 5. TRONG KHOẢNG [t, t+1): CÁC XE DI CHUYỂN LIÊN TỤC & XỬ LÝ SỰ KIỆN
        # --------------------------------------------------------------------
        # Ghi nhận các cuộc gọi đang chờ ở đầu bước t0 (để tính \Delta W_{c,t} chính xác)
        waiting_calls_at_t0: List[EmergencyCall] = [
            c for c in self.calls.values()
            if c.status in (CallStatus.PENDING, CallStatus.WAITING_FOR_AMBULANCE)
        ]

        for amb in self.ambulances.values():
            amb.step_continuous(
                dt_minute=1.0,
                current_minute=t0,
                city_map=self.city_map,
                calls_dict=self.calls,
            )

        # Cập nhật thời gian thực sự chờ \Delta W_{c,t} trong bước và kiểm tra mốc thất bại
        wait_penalty = 0.0
        for call in waiting_calls_at_t0:
            w_c = REWARD_CONFIG[call.severity]["wait_penalty_per_min"]
            if (
                call.arrived_at_scene_time is not None
                and call.status in (CallStatus.AT_SCENE, CallStatus.COMPLETED)
            ):
                # Xe đã đến hiện trường giữa phút trong khoảng [t0, t0 + 1]:
                # Thời gian thực sự chờ trong bước là \Delta W_{c,t} = t_arrival - t0
                delta_w = max(0.0, min(1.0, call.arrived_at_scene_time - t0))
                call.wait_time += delta_w
                wait_penalty += w_c * delta_w
            else:
                # Xe chưa đến hiện trường trong khoảng [t0, t0 + 1): chờ trọn 1.0 phút
                delta_w = 1.0
                call.update_wait_time(delta_w)
                wait_penalty += w_c * delta_w

                if call.status == CallStatus.FAILED or call.check_failure(current_time=t0 + 1.0):
                    # Cuộc gọi bị thất bại do quá hạn
                    if not call.failure_penalized:
                        call.failure_penalized = True
                        self.failed_calls_count += 1
                        self.calls_failed_this_step.append(call)

                    # "Khi cuộc gọi bị xoá mà trong lúc đó có xe đang tiếp cận thì xe sẽ lập tức về trạng thái rảnh"
                    if call.assigned_ambulance_id is not None:
                        assigned_amb = self.ambulances.get(call.assigned_ambulance_id)
                        if assigned_amb and assigned_amb.status == AmbulanceStatus.EN_ROUTE_TO_CALL:
                            assigned_amb.abort_mission_to_idle()

        # Kiểm tra phần thưởng b_c cho các cuộc gọi vừa hoàn thành trong bước (C_t)
        completion_bonus = 0.0
        for call in self.calls.values():
            if call.status == CallStatus.COMPLETED and not call.completion_rewarded:
                call.completion_rewarded = True
                self.calls_completed_this_step.append(call)
                self.served_calls_count += 1
                if call.is_perfect:
                    self.perfect_calls_count += 1
                    b_c = REWARD_CONFIG[call.severity]["perfect_reward"]
                else:
                    b_c = REWARD_CONFIG[call.severity]["complete_reward"]
                completion_bonus += b_c

        # --------------------------------------------------------------------
        # 6. TẠI t+1: CẬP NHẬT GIAO THÔNG, REROUTE, SINH CUỘC GỌI MỚI
        # --------------------------------------------------------------------
        self.current_minute += 1
        self.traffic_time_step += 1
        self.city_map.time_step = self.traffic_time_step - 1
        self.city_map.traffic_time_step = self.traffic_time_step - 1

        # Cập nhật trạng thái giao thông (Markov transition) song song với bước thời gian
        self.city_map.step_traffic()
        self.traffic_time_step = self.city_map.traffic_time_step

        # Reroute xe nếu có thay đổi trên lộ trình
        for amb in self.ambulances.values():
            amb.reroute_if_beneficial(self.city_map)

        # Sinh cuộc gọi mới tại đầu phút t+1 (các cuộc gọi bị từ chối cũng thuộc F_t)
        new_calls = self._generate_new_calls()
        self.calls_generated_this_step = new_calls

        # --------------------------------------------------------------------
        # 7. TÍNH PHẦN THƯỞNG r_t THEO CÔNG THỨC CHUẨN:
        # r_t = - \sum_c w_c \Delta W_{c,t} - \sum_{c \in F_t} p_c - 0.05 \sum_i \Delta d_{i,t} + \sum_{c \in C_t} b_c
        # --------------------------------------------------------------------
        failure_penalty = sum(
            REWARD_CONFIG[c.severity]["fail_penalty"]
            for c in self.calls_failed_this_step
        )
        step_distance = sum(amb.last_step_distance for amb in self.ambulances.values())
        distance_penalty = DISTANCE_PENALTY_PER_KM * step_distance

        step_reward = -wait_penalty - failure_penalty - distance_penalty + completion_bonus
        self.last_step_reward = step_reward
        self.total_reward += step_reward
        self.last_reward_breakdown = {
            "wait_penalty": wait_penalty,
            "failure_penalty": failure_penalty,
            "distance_penalty": distance_penalty,
            "completion_bonus": completion_bonus,
            "total": step_reward,
        }

        # --------------------------------------------------------------------
        # 8. KIỂM TRA ĐIỀU KIỆN KẾT THÚC & TRẢ VỀ
        # --------------------------------------------------------------------
        # Mô phỏng kết thúc bình thường khi đã vượt qua max_time (360 phút)
        # VÀ không còn cuộc gọi nào đang hoạt động, tất cả xe đã trở về Rảnh (hoàn tất giao bệnh nhân)
        done = False
        if self.current_minute >= self.max_time:
            active_count = len(self.active_calls)
            all_ambs_idle = all(a.status == AmbulanceStatus.IDLE for a in self.ambulances.values())
            if active_count == 0 and all_ambs_idle:
                done = True

        # Kiểm tra ngưỡng vượt quá thời gian mô phỏng (300 phút quá giờ) -> Dừng và báo TRUNCATED
        if not done and self.current_minute >= (self.max_time + self.max_overtime):
            done = True
            self.is_truncated = True

        info = {
            "episode": self.episode_count,
            "minute": self.current_minute,
            "current_minute": self.current_minute,
            "traffic_time_step": self.traffic_time_step,
            "generated_calls": len(new_calls),
            "action_id": self.last_action_id,
            "decoded_action": list(self.last_decoded_action),
            "valid_action_count": self.last_valid_action_count,
            "served": self.served_calls_count,
            "failed": self.failed_calls_count,
            "perfect": self.perfect_calls_count,
            "step_reward": step_reward,
            "total_reward": self.total_reward,
            "reward_breakdown": dict(self.last_reward_breakdown),
            "truncated": self.is_truncated,
            "backlog": self.get_backlog() if self.is_truncated else None,
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
        lines.append(f"Bước thời gian giao thông (traffic_time_step): {self.traffic_time_step}")
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

        lines.append(f"Trạng thái Slot (đầu bước): {self.last_slot_summary}")
        lines.append(
            f"Hành động đồng thời (Action #{self.last_action_id}, "
            f"hợp lệ {self.last_valid_action_count}/{self.num_potential_actions}): "
            f"{self.last_decoded_action if self.last_decoded_action else '[] (Không phân công)'}"
        )
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
        rb = self.last_reward_breakdown
        lines.append(
            f"Reward bước này (r_t): {self.last_step_reward:.2f} "
            f"(Phạt chờ: -{rb['wait_penalty']:.2f}, Phạt thất bại: -{rb['failure_penalty']:.2f}, "
            f"Phạt quãng đường: -{rb['distance_penalty']:.2f}, Thưởng hoàn thành: +{rb['completion_bonus']:.2f})"
        )
        lines.append(f"Tổng Reward tích lũy: {self.total_reward:.2f}")
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

        # Nếu mô phỏng bị dừng (TRUNCATED), thông báo lý do và liệt kê chi tiết công việc tồn đọng
        if self.is_truncated:
            lines.append("")
            lines.append(self.format_backlog_report())

        return "\n".join(lines)

    def get_backlog(self) -> Dict[str, Any]:
        """
        Thống kê các công việc đang bị tồn đọng (backlog) khi chạm ngưỡng thời gian tối đa:
        - Các cuộc gọi đang PENDING (chưa có xe nhận).
        - Các cuộc gọi đang WAITING_FOR_AMBULANCE (đang chờ xe đến).
        - Các cuộc gọi đang AT_SCENE (đang sơ cứu tại hiện trường).
        - Các xe đang EN_ROUTE_TO_HOSPITAL (đang chở bệnh nhân đi cấp cứu).
        - Các xe đang AT_HOSPITAL (đang bàn giao bệnh nhân tại bệnh viện).
        """
        pending = [c for c in self.calls.values() if c.status == CallStatus.PENDING]
        waiting = [c for c in self.calls.values() if c.status == CallStatus.WAITING_FOR_AMBULANCE]
        at_scene = [c for c in self.calls.values() if c.status == CallStatus.AT_SCENE]
        en_route_hosp = [a for a in self.ambulances.values() if a.status == AmbulanceStatus.EN_ROUTE_TO_HOSPITAL]
        at_hosp = [a for a in self.ambulances.values() if a.status == AmbulanceStatus.AT_HOSPITAL]

        return {
            "pending_calls": pending,
            "waiting_calls": waiting,
            "at_scene_calls": at_scene,
            "en_route_hospital_ambulances": en_route_hosp,
            "at_hospital_ambulances": at_hosp,
            "total_backlog_count": len(pending) + len(waiting) + len(at_scene) + len(en_route_hosp) + len(at_hosp),
        }

    def format_backlog_report(self) -> str:
        """
        Định dạng thông báo báo cáo chi tiết các công việc bị tồn đọng khi chạm ngưỡng Truncated (vượt quá 300 phút).
        """
        backlog = self.get_backlog()
        lines: List[str] = []
        lines.append("==================================================")
        lines.append("THÔNG BÁO: MÔ PHỎNG BỊ DỪNG (TRUNCATED) DO CHẠM NGƯỠNG 300 PHÚT QUÁ GIỜ!")
        lines.append(f"Thời gian hiện tại: {self.current_minute} phút (Giới hạn: {self.max_time}p + {self.max_overtime}p quá giờ)")
        lines.append(f"Tổng số công việc còn tồn đọng: {backlog['total_backlog_count']}")
        lines.append("--------------------------------------------------")
        lines.append("DANH SÁCH CÁC CÔNG VIỆC TỒN ĐỌNG CHI TIẾT:")

        # 1. Cuộc gọi PENDING
        lines.append(f"1. Cuộc gọi yêu cầu xử lý chưa có xe nhận (PENDING): {len(backlog['pending_calls'])} cuộc gọi")
        for c in backlog["pending_calls"]:
            lines.append(
                f"   • Cuộc gọi #{c.call_id} (Mức {c.severity}) tại N{c.edge[0]}->N{c.edge[1]} (+{c.distance_from_start:.2f}km), "
                f"đã chờ: {c.wait_time:.1f}p / Hạn chót: {c.deadline:.1f}p"
            )

        # 2. Cuộc gọi WAITING_FOR_AMBULANCE
        lines.append(f"2. Cuộc gọi đang đợi xe cứu thương tới (WAITING_FOR_AMBULANCE): {len(backlog['waiting_calls'])} cuộc gọi")
        for c in backlog["waiting_calls"]:
            lines.append(
                f"   • Cuộc gọi #{c.call_id} (Mức {c.severity}) giao cho Xe {c.assigned_ambulance_id}, "
                f"đã chờ: {c.wait_time:.1f}p / Hạn chót: {c.deadline:.1f}p"
            )

        # 3. Cuộc gọi AT_SCENE
        lines.append(f"3. Cuộc gọi đang được sơ cứu tại hiện trường (AT_SCENE): {len(backlog['at_scene_calls'])} cuộc gọi")
        for c in backlog["at_scene_calls"]:
            lines.append(f"   • Cuộc gọi #{c.call_id} (Mức {c.severity}), Xe {c.assigned_ambulance_id} đang xử lý tại chỗ")

        # 4. Xe đang đưa bệnh nhân đến bệnh viện
        lines.append(f"4. Xe cứu thương đang đưa bệnh nhân tới bệnh viện (EN_ROUTE_TO_HOSPITAL): {len(backlog['en_route_hospital_ambulances'])} xe")
        for a in backlog["en_route_hospital_ambulances"]:
            lines.append(f"   • Xe {a.ambulance_id} đang chở bệnh nhân cuộc gọi #{a.current_call_id} tới {a.target_name}")

        # 5. Xe đang bàn giao tại bệnh viện
        lines.append(f"5. Xe cứu thương đang bàn giao tại bệnh viện (AT_HOSPITAL): {len(backlog['at_hospital_ambulances'])} xe")
        for a in backlog["at_hospital_ambulances"]:
            lines.append(f"   • Xe {a.ambulance_id} đang bàn giao bệnh nhân cuộc gọi #{a.current_call_id} tại {a.target_name}")

        lines.append("==================================================")
        return "\n".join(lines)
