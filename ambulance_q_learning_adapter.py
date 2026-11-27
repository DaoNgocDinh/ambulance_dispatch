"""Adapter chuyển môi trường cứu thương sang giao diện state/action Q-learning.

Tuân thủ RL Algorithm Development & Evaluation Standard v1.0:
- Đọc trạng thái động qua API chuẩn `env.get_observation()`, `env.get_action_mask()`, `env.decode_action()`.
- Ánh xạ tường minh chuỗi trạng thái tiếng Việt trong Observation thông qua Enum `AmbulanceStatus`.
- Tính ETA thực tế trực tiếp từ `obs["edges_by_id"]` mà không can thiệp trạng thái nội bộ `env`.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from dispatch_env import (
    AmbulanceDispatchEnv,
    AmbulanceStatus,
    DISTANCE_PENALTY_PER_KM,
    REWARD_CONFIG,
    SEVERITY_CONFIG,
)
from map_env.edge import SPEEDS_CENTER, SPEEDS_OUTER


class AmbulanceDispatchAdapter:
    """Biến Observation của môi trường thành state key gọn và hỗ trợ canonical action keys."""

    IDLE_STATUS_STR = AmbulanceStatus.IDLE.value

    def __init__(self, top_k_per_slot: int = 2) -> None:
        if top_k_per_slot < 1:
            raise ValueError("top_k_per_slot phải >= 1.")
        self.top_k_per_slot = top_k_per_slot
        self._signature_cache: Dict[Tuple[Any, ...], str] = {}
        self._cached_edges_sig: Optional[Tuple[Tuple[int, int, int, float], ...]] = None
        self._cached_adj: Dict[int, List[Tuple[int, float]]] = {}
        self._sp_cache: Dict[int, Dict[int, float]] = {}
        self._eta_cache: Dict[Tuple[int, float, int, float], float] = {}

    @staticmethod
    def _bucket(value: float, limits: Tuple[float, ...]) -> int:
        """Đổi số liên tục thành chỉ số khoảng để state có số giá trị hữu hạn."""
        for index, limit in enumerate(limits):
            if value <= limit:
                return index
        return len(limits)

    def _environment_signature(self, env: AmbulanceDispatchEnv) -> str:
        """Tạo fingerprint theo cấu trúc MDP để tách Q-table các cấu hình khác nhau mà vẫn tổng quát hóa qua seed."""
        city_map = env.city_map
        cache_key = (
            city_map.num_nodes,
            len(city_map.nodes),
            len(city_map.edges_by_id),
            len(city_map.hospitals),
            len(city_map.road_pairs),
            len(env.potential_actions),
            env.num_ambulances,
            env.max_pending_calls,
            env.max_time,
            env.max_overtime,
            env.poisson_lambda,
        )
        cached = self._signature_cache.get(cache_key)
        if cached is not None:
            return cached

        action_space = [tuple(action) for action in env.potential_actions]
        configuration = (
            city_map.num_nodes,
            len(city_map.nodes),
            len(city_map.edges_by_id),
            len(city_map.hospitals),
            len(city_map.road_pairs),
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
        signature = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        if len(self._signature_cache) > 64:
            self._signature_cache.clear()
        self._signature_cache[cache_key] = signature
        return signature

    def _ensure_snapshot_cache(self, obs: Mapping[str, Any]) -> None:
        """Đồng bộ cache đồ thị và ETA theo đúng ảnh chụp `obs["edges_by_id"]` hiện tại."""
        edges_by_id = obs["edges_by_id"]
        edges_sig = tuple(
            (
                int(edge_id),
                int(info["from_node"]),
                int(info["to_node"]),
                round(float(info["travel_time_minutes"]), 4),
            )
            for edge_id, info in sorted(edges_by_id.items())
        )
        if edges_sig == self._cached_edges_sig:
            return

        adj: Dict[int, List[Tuple[int, float]]] = {}
        for _, u, v, t_min in edges_sig:
            adj.setdefault(u, []).append((v, t_min))

        self._cached_edges_sig = edges_sig
        self._cached_adj = adj
        self._sp_cache.clear()
        self._eta_cache.clear()

    def _shortest_times_from(self, obs: Mapping[str, Any], source_node: int) -> Dict[int, float]:
        """Dijkstra từ ``source_node`` chỉ sử dụng ``obs["edges_by_id"]``."""
        self._ensure_snapshot_cache(obs)
        cached = self._sp_cache.get(source_node)
        if cached is not None:
            return cached

        adj = self._cached_adj
        dist: Dict[int, float] = {source_node: 0.0}
        pq: List[Tuple[float, int]] = [(0.0, source_node)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float("inf")):
                continue
            for v, w in adj.get(u, ()):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))

        self._sp_cache[source_node] = dist
        return dist

    def compute_eta_from_obs(
        self,
        obs: Mapping[str, Any],
        start_edge_id: int,
        start_dist: float,
        dest_edge_id: int,
        dest_dist: float,
    ) -> float:
        """Tính ETA (phút) giữa hai vị trí trên cung đường chỉ từ Observation."""
        self._ensure_snapshot_cache(obs)
        cache_key = (
            start_edge_id,
            round(start_dist, 4),
            dest_edge_id,
            round(dest_dist, 4),
        )
        cached = self._eta_cache.get(cache_key)
        if cached is not None:
            return cached

        edges_by_id = obs["edges_by_id"]
        e_start = edges_by_id[start_edge_id]
        e_dest = edges_by_id[dest_edge_id]

        if start_edge_id == dest_edge_id and start_dist <= dest_dist:
            eta = round(((dest_dist - start_dist) / float(e_start["speed"])) * 60.0, 3)
        else:
            rem_start_dist = max(0.0, float(e_start["length"]) - start_dist)
            rem_start_time = (rem_start_dist / float(e_start["speed"])) * 60.0
            v_start = int(e_start["to_node"])
            u_dest = int(e_dest["from_node"])
            sp_times = self._shortest_times_from(obs, v_start)
            inter_time = sp_times.get(u_dest, float("inf"))
            dest_time = (dest_dist / float(e_dest["speed"])) * 60.0
            eta = round(rem_start_time + inter_time + dest_time, 3)

        self._eta_cache[cache_key] = eta
        return eta

    @classmethod
    def get_idle_ambulances_from_obs(cls, obs: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Lấy các xe đang rảnh từ ``obs["ambulances"]`` theo thứ tự ID ổn định."""
        ambulances = obs["ambulances"]
        return [
            ambulances[aid]
            for aid in sorted(ambulances)
            if ambulances[aid]["status"] == cls.IDLE_STATUS_STR
            and ambulances[aid]["current_call_id"] is None
        ]

    @classmethod
    def is_decision_state_from_obs(cls, obs: Mapping[str, Any]) -> bool:
        """Kiểm tra có ít nhất 1 slot PENDING và ít nhất 1 xe rảnh trong Observation."""
        has_pending = any(slot is not None for slot in obs["call_slots"])
        if not has_pending:
            return False
        return bool(cls.get_idle_ambulances_from_obs(obs))

    def is_decision_state(self, env: AmbulanceDispatchEnv) -> bool:
        """Kiểm tra bước hiện tại có thực sự cần ra quyết định điều phối hay không."""
        return self.is_decision_state_from_obs(env.get_observation())

    def _rank_idle_ambulances_by_slot(
        self, obs: Mapping[str, Any]
    ) -> Tuple[List[Optional[Dict[str, Any]]], List[Dict[str, Any]], Dict[int, Dict[int, int]], Dict[int, List[Tuple[float, int]]]]:
        """Xếp hạng các xe rảnh theo ETA thực tế tới từng slot cuộc gọi trong Observation."""
        slots: List[Optional[Dict[str, Any]]] = obs["call_slots"]
        idle_ambs = self.get_idle_ambulances_from_obs(obs)
        slot_ranks: Dict[int, Dict[int, int]] = {}
        slot_etas: Dict[int, List[Tuple[float, int]]] = {}

        for slot_idx, call_dict in enumerate(slots):
            if call_dict is None or not idle_ambs:
                continue
            dest_edge_id = int(call_dict["edge_id"])
            dest_dist = float(call_dict["dist"])
            ranked = sorted(
                (
                    self.compute_eta_from_obs(
                        obs,
                        int(amb["current_edge_id"]),
                        float(amb["distance_from_start"]),
                        dest_edge_id,
                        dest_dist,
                    ),
                    int(amb["id"]),
                )
                for amb in idle_ambs
            )
            slot_etas[slot_idx] = ranked
            slot_ranks[slot_idx] = {
                amb_id: rank for rank, (_, amb_id) in enumerate(ranked)
            }

        return slots, idle_ambs, slot_ranks, slot_etas

    def encode_state(self, env: AmbulanceDispatchEnv, obs: Optional[Mapping[str, Any]] = None) -> str:
        """Mã hóa Observation thành state key gọn, bất biến theo hoán vị ID xe."""
        observation = env.get_observation() if obs is None else obs
        signature = self._environment_signature(env)
        slots, idle_ambs, _, slot_etas = self._rank_idle_ambulances_by_slot(observation)
        active_slots = [(idx, call) for idx, call in enumerate(slots) if call is not None]

        if not active_slots or not idle_ambs:
            return json.dumps((signature, "WAIT"), separators=(",", ":"))

        current_minute = float(observation["current_minute"])
        idle_bucket = min(len(idle_ambs), 3)  # 1, 2, 3+ xe rảnh
        call_features = []
        best_amb_ids = []

        for slot_idx in range(len(slots)):
            call_dict = slots[slot_idx]
            if call_dict is None:
                call_features.append(None)
                continue

            ranked = slot_etas[slot_idx]
            best_eta, best_amb_id = ranked[0]
            best_amb_ids.append(best_amb_id)

            remaining_time = max(0.0, float(call_dict["deadline"]) - current_minute)
            if best_eta <= 0.6 * remaining_time:
                eta_feasibility = 0
            elif best_eta <= remaining_time:
                eta_feasibility = 1
            else:
                eta_feasibility = 2

            call_features.append((int(call_dict["severity"]), eta_feasibility))

        has_top1_conflict = int(len(set(best_amb_ids)) < len(best_amb_ids))
        state = (
            signature,
            idle_bucket,
            tuple(call_features),
            has_top1_conflict,
        )
        return json.dumps(state, separators=(",", ":"))

    @staticmethod
    def valid_actions(env: AmbulanceDispatchEnv, obs: Optional[Mapping[str, Any]] = None) -> List[int]:
        """Lấy toàn bộ action ID hợp lệ từ ``obs['action_mask']`` / ``env.get_action_mask()``."""
        mask = env.get_action_mask() if obs is None else obs["action_mask"]
        return [int(idx) for idx in np.flatnonzero(mask)]

    def _decode_to_slot_pairs(
        self,
        env: AmbulanceDispatchEnv,
        action_id: int,
        call_id_to_slot: Mapping[int, int],
    ) -> List[Tuple[int, int]]:
        """Giải mã action_id qua API chuẩn `env.decode_action` thành cặp `(amb_id, slot_idx)`."""
        decoded = env.decode_action(action_id)
        return [
            (int(amb_id), call_id_to_slot[int(call_id)])
            for amb_id, call_id in decoded
            if int(call_id) in call_id_to_slot
        ]

    def candidate_actions(
        self, env: AmbulanceDispatchEnv, obs: Optional[Mapping[str, Any]] = None
    ) -> List[int]:
        """Lọc tập action ứng viên từ action mask hợp lệ:
        - Luôn tuân thủ tuyệt đối `action_mask`.
        - Giữ lại `action_id = 0` (WAIT) bất cứ khi nào `action_mask` cho phép.
        - Không ép số phân công tối thiểu hay buộc xe rảnh phải nhận ca khả thi.
        - Chỉ lọc các cặp phân công dùng top-K xe có ETA nhanh nhất cho từng slot.
        """
        observation = env.get_observation() if obs is None else obs
        all_valid = self.valid_actions(env, observation)
        if len(all_valid) <= 1 or not self.is_decision_state_from_obs(observation):
            return all_valid

        slots, idle_ambs, slot_ranks, _ = self._rank_idle_ambulances_by_slot(observation)
        call_id_to_slot = {
            int(call_dict["call_id"]): idx
            for idx, call_dict in enumerate(slots)
            if call_dict is not None
        }
        decoded_slot_pairs = {
            action_id: self._decode_to_slot_pairs(env, action_id, call_id_to_slot)
            for action_id in all_valid
        }

        def filter_with_k(k: int) -> List[int]:
            selected: List[int] = []
            for action_id in all_valid:
                if action_id == 0:
                    selected.append(action_id)
                    continue
                slot_pairs = decoded_slot_pairs[action_id]
                if all(
                    slot_idx in slot_ranks
                    and slot_ranks[slot_idx].get(amb_id, k) < k
                    for amb_id, slot_idx in slot_pairs
                ):
                    selected.append(action_id)
            return selected

        candidates = filter_with_k(self.top_k_per_slot)
        if not any(aid != 0 for aid in candidates) and self.top_k_per_slot < len(idle_ambs):
            candidates = filter_with_k(min(len(idle_ambs), self.top_k_per_slot + 1))

        return candidates if candidates else all_valid

    def get_action_keys(
        self,
        env: AmbulanceDispatchEnv,
        actions: Sequence[int],
        obs: Optional[Mapping[str, Any]] = None,
    ) -> Dict[int, int]:
        """Ánh xạ mỗi ``action_id`` hợp lệ sang canonical key bất biến theo thứ hạng ETA."""
        observation = env.get_observation() if obs is None else obs
        slots, idle_ambs, slot_ranks, _ = self._rank_idle_ambulances_by_slot(observation)
        if not idle_ambs or not any(slot is not None for slot in slots):
            return {action_id: 0 for action_id in actions}

        call_id_to_slot = {
            int(call_dict["call_id"]): idx
            for idx, call_dict in enumerate(slots)
            if call_dict is not None
        }
        base = 5  # 0 = unassigned, 1..4 = ETA rank buckets
        num_slots = len(slots)
        key_map: Dict[int, int] = {}

        for action_id in actions:
            slot_choice = [0] * num_slots
            for amb_id, slot_idx in self._decode_to_slot_pairs(env, action_id, call_id_to_slot):
                rank = slot_ranks.get(slot_idx, {}).get(amb_id, 3)
                slot_choice[slot_idx] = min(rank, 3) + 1

            canonical_key = 0
            for val in slot_choice:
                canonical_key = canonical_key * base + val
            key_map[action_id] = canonical_key

        return key_map

    @staticmethod
    def step(
        env: AmbulanceDispatchEnv, action_id: int
    ) -> Tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Gửi action ID hợp lệ tới API môi trường."""
        return env.step(action=action_id)
