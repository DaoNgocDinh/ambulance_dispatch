"""
Bộ quản lý và tích hợp chính sách điều phối tập trung (Policy Manager).
Tích hợp đồng nhất 4 thuật toán từ các thành viên trong nhóm:
1. Heuristic (Baseline) - Quy tắc tham lam ưu tiên mức độ khẩn cấp & xe gần nhất
2. Q-Learning - Học tăng cường Off-policy Q-Table với mã hóa slot & action mask
3. SARSA - Học tăng cường On-policy SARSA với 6 Meta-Actions & định hình phần thưởng
4. Monte Carlo - Học tăng cường On-policy First-Visit MC Control theo Episode
Cùng với các chế độ hỗ trợ: Ngẫu nhiên (Random) và Thủ công (Manual).
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from dispatch_env import AmbulanceDispatchEnv, AmbulanceStatus, CallStatus
from map_env import CityMap

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PolicyManager:
    """Quản lý các thuật toán điều phối xe cứu thương."""

    POLICIES = [
        {
            "id": "heuristic",
            "name": "Heuristic (Baseline)",
            "author": "Nhóm / Baseline",
            "type": "Rule-based",
            "description": "Ưu tiên theo mức độ nghiêm trọng và điều xe cứu thương rảnh có khoảng cách ngắn nhất.",
            "badge_color": "#3b82f6",
        },
        {
            "id": "q_learning",
            "name": "Q-Learning",
            "author": "Nhánh q-learning",
            "type": "RL (Off-policy)",
            "description": "Bảng Q-Table rút gọn 4 đặc trưng, lọc ứng viên qua Action Mask và đối xứng Canonical Keys.",
            "badge_color": "#10b981",
        },
        {
            "id": "sarsa",
            "name": "SARSA",
            "author": "Nhánh SARSA",
            "type": "RL (On-policy)",
            "description": "Tabular On-policy SARSA với 6 Meta-Actions: điều gần nhất, xe thứ 2, cân bằng tải, edge-first, tái định vị.",
            "badge_color": "#f59e0b",
        },
        {
            "id": "monte_carlo",
            "name": "Monte Carlo",
            "author": "Nhánh on-policy-MC",
            "type": "RL (First-Visit MC)",
            "description": "First-Visit MC Control học cập nhật Q(s,a) sau toàn bộ episode với bảng tra trạng thái rời rạc.",
            "badge_color": "#8b5cf6",
        },
        {
            "id": "random",
            "name": "Ngẫu Nhiên (Random)",
            "author": "Baseline",
            "type": "Random Valid",
            "description": "Chọn ngẫu nhiên đồng đều trong tập các hành động hợp lệ (thỏa mãn Action Mask) tại mỗi bước.",
            "badge_color": "#64748b",
        },
    ]

    def __init__(self, models_dir: Optional[Union[str, Path]] = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else PROJECT_ROOT
        self._q_agent = None
        self._q_adapter = None
        self._sarsa_agent = None
        self._mc_agent = None
        self._init_models()

    def _init_models(self) -> None:
        """Khởi tạo và tải trước các mô hình đã huấn luyện nếu có."""
        # 1. Q-Learning
        try:
            from ambulance_q_learning_adapter import AmbulanceDispatchAdapter
            from q_learning import QLearningAgent

            ql_path = self.models_dir / "results" / "q_learning_final_v2" / "q_table.json"
            self._q_adapter = AmbulanceDispatchAdapter()
            if ql_path.exists():
                self._q_agent = QLearningAgent.load_from_file(ql_path)
            else:
                self._q_agent = QLearningAgent()
        except Exception as e:
            print(f"[PolicyManager] Warning loading Q-Learning: {e}")

        # 2. SARSA
        try:
            from dispatch_env.sarsa import SarsaAgent

            sarsa_path = self.models_dir / "sarsa_policy_meta_5000_seed42.json"
            if sarsa_path.exists():
                self._sarsa_agent = SarsaAgent.load_checkpoint(sarsa_path)
            else:
                self._sarsa_agent = SarsaAgent()
        except Exception as e:
            print(f"[PolicyManager] Warning loading SARSA: {e}")

        # 3. Monte Carlo
        try:
            from group_adapter import GroupEnvAdapter
            from mc_agent import MonteCarloAgent

            mc_path = self.models_dir / "mc_model_seed42.pkl"
            if mc_path.exists():
                self._mc_agent = MonteCarloAgent.load(str(mc_path))
            else:
                # Khởi tạo MC agent với số action mặc định (6 xe x 3 slots + 1 = 19 actions)
                self._mc_agent = MonteCarloAgent(number_of_actions=19, seed=42)
        except Exception as e:
            print(f"[PolicyManager] Warning loading Monte Carlo: {e}")

    @classmethod
    def get_available_policies(cls) -> List[Dict[str, Any]]:
        """Trả về danh sách thông tin metadata của tất cả các thuật toán."""
        return cls.POLICIES

    def select_action(
        self,
        policy_id: str,
        env: AmbulanceDispatchEnv,
        obs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, str, str]:
        """
        Chọn hành động tiếp theo theo thuật toán được chỉ định.
        Trả về: (action_payload_cho_env_step, policy_name, mô_tả_hành_động)
        """
        obs = env.get_observation() if obs is None else obs
        pid = (policy_id or "heuristic").lower()

        # -------------------------------------------------------------
        # 1. HEURISTIC BASELINE
        # -------------------------------------------------------------
        if pid in ("heuristic", "auto"):
            # Gọi trực tiếp qua auto_dispatch=True trong env.step
            return "auto_dispatch", "Heuristic (Baseline)", "Điều phối tham lam: xe gần nhất -> ca khẩn cấp nhất"

        # -------------------------------------------------------------
        # 2. Q-LEARNING
        # -------------------------------------------------------------
        if pid == "q_learning":
            if self._q_agent is None or self._q_adapter is None:
                return "auto_dispatch", "Q-Learning (Fallback Heuristic)", "Chưa tải được mô hình Q-table"

            is_decision = self._q_adapter.is_decision_state_from_obs(obs)
            if not is_decision:
                return 0, "Q-Learning", "Không có ca chờ hoặc không có xe rảnh -> Chờ (Action 0)"

            try:
                state_key = self._q_adapter.encode_state(env, obs=obs)
                candidates = self._q_adapter.candidate_actions(env, obs=obs)
                action_keys = self._q_adapter.get_action_keys(env, candidates, obs=obs)
                action_id = self._q_agent.choose_action(
                    state_key, candidates, explore=False, action_keys=action_keys
                )
                decoded = env.decode_action(action_id)
                desc = f"Action {action_id}: Phân công {decoded}" if decoded else "Chờ (Action 0)"
                return action_id, "Q-Learning", desc
            except Exception as e:
                return "auto_dispatch", "Q-Learning (Fallback)", f"Lỗi tính toán ({e}), fallback sang Heuristic"

        # -------------------------------------------------------------
        # 3. SARSA
        # -------------------------------------------------------------
        if pid == "sarsa":
            if self._sarsa_agent is None:
                return "auto_dispatch", "SARSA (Fallback Heuristic)", "Chưa tải được mô hình SARSA"

            try:
                from dispatch_env.sarsa import resolve_meta_action, sarsa_action_mask
                from dispatch_env.sarsa_state import encode_sarsa_state

                s_meta = encode_sarsa_state(obs)
                s_mask = sarsa_action_mask(obs, env=env)
                meta_act = self._sarsa_agent.choose_action(s_meta, action_mask=s_mask, epsilon=0.0)
                action_id, pairs = resolve_meta_action(env, obs, meta_act)
                meta_names = {
                    0: "Gần nhất",
                    1: "Xe thứ hai",
                    2: "Cân bằng tải (ít chạy nhất)",
                    3: "Edge-First",
                    4: "Tái định vị về tâm",
                    5: "Không điều xe (No-op)",
                }
                meta_name = meta_names.get(meta_act, f"Meta-{meta_act}")
                desc = f"Meta-Action {meta_act} ({meta_name}) -> Action {action_id}: {pairs}"
                return action_id, "SARSA", desc
            except Exception as e:
                return "auto_dispatch", "SARSA (Fallback)", f"Lỗi ({e}), fallback sang Heuristic"

        # -------------------------------------------------------------
        # 4. MONTE CARLO
        # -------------------------------------------------------------
        if pid in ("monte_carlo", "mc"):
            if self._mc_agent is None:
                return "auto_dispatch", "Monte Carlo (Fallback Heuristic)", "Chưa tải được mô hình MC"

            try:
                from group_adapter import GroupEnvAdapter

                adapter = GroupEnvAdapter(env=env)
                state = adapter.discretize_state()
                valid = adapter.get_valid_actions()
                action_idx = self._mc_agent.select_action(state, valid, training=False)
                pair = adapter._decode(action_idx)
                desc = adapter.describe_action(action_idx)
                return pair, "Monte Carlo", desc
            except Exception as e:
                return "auto_dispatch", "Monte Carlo (Fallback)", f"Lỗi ({e}), fallback sang Heuristic"

        # -------------------------------------------------------------
        # 5. RANDOM VALID
        # -------------------------------------------------------------
        if pid == "random":
            mask = obs.get("action_mask")
            if mask is not None:
                valid_ids = [int(i) for i, m in enumerate(mask) if m]
                if valid_ids:
                    chosen_id = int(random.choice(valid_ids))
                    decoded = env.decode_action(chosen_id)
                    return chosen_id, "Ngẫu Nhiên", f"Chọn ngẫu nhiên Action {chosen_id}: {decoded}"
            return 0, "Ngẫu Nhiên", "Chờ (Action 0)"

        # Mặc định Heuristic
        return "auto_dispatch", "Heuristic (Baseline)", "Mặc định Heuristic"

    def execute_step(
        self,
        policy_id: str,
        env: AmbulanceDispatchEnv,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any], str]:
        """
        Thực hiện một bước trong môi trường với thuật toán được chọn.
        Trả về: (obs, reward, done, info, action_description)
        """
        obs = env.get_observation()
        action_payload, policy_name, desc = self.select_action(policy_id, env, obs)

        if action_payload == "auto_dispatch":
            next_obs, reward, done, info = env.step(auto_dispatch=True)
        else:
            next_obs, reward, done, info = env.step(action=action_payload)

        info["policy_name"] = policy_name
        info["policy_id"] = policy_id
        info["action_description"] = desc
        return next_obs, reward, done, info, desc

    def run_benchmark(
        self,
        policies: Optional[List[str]] = None,
        num_episodes: int = 3,
        seeds: Optional[List[int]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Chạy so sánh đối chuẩn hiệu năng giữa 4 thuật toán trên các seed giống hệt nhau.
        """
        if policies is None:
            policies = ["heuristic", "q_learning", "sarsa", "monte_carlo"]

        if seeds is None:
            seeds = [42, 100, 2024][:num_episodes]

        cfg = {
            "num_ambulances": 6,
            "max_time": 360,
            "max_overtime": 300,
            "poisson_lambda": 0.1,
            "max_pending_calls": 3,
        }
        if config:
            cfg.update(config)

        results = {}
        for pol_id in policies:
            pol_info = next((p for p in self.POLICIES if p["id"] == pol_id), {"name": pol_id})
            ep_records = []

            for seed in seeds:
                city_map = CityMap(seed=seed)
                env = AmbulanceDispatchEnv(
                    city_map=city_map,
                    num_ambulances=cfg["num_ambulances"],
                    max_time=cfg["max_time"],
                    max_overtime=cfg["max_overtime"],
                    poisson_lambda=cfg["poisson_lambda"],
                    max_pending_calls=cfg["max_pending_calls"],
                    seed=seed,
                )

                step_rewards = []
                steps = 0
                done = False

                while not done:
                    _, reward, done, _, _ = self.execute_step(pol_id, env)
                    step_rewards.append(reward)
                    steps += 1

                total_gen = len(env.calls)
                served = env.served_calls_count
                failed = env.failed_calls_count
                perfect = env.perfect_calls_count
                total_reward = env.total_reward

                # Tính tổng quãng đường di chuyển của đội xe
                total_dist = sum(a.total_distance_traveled for a in env.ambulances.values())

                # Tính thời gian phản hồi trung bình các ca hoàn thành
                completed_calls = [c for c in env.calls.values() if c.status == CallStatus.COMPLETED and c.arrived_at_scene_time is not None]
                avg_response = (
                    sum(c.arrived_at_scene_time - c.created_time for c in completed_calls) / len(completed_calls)
                    if completed_calls
                    else 0.0
                )

                ep_records.append({
                    "seed": seed,
                    "total_reward": float(total_reward),
                    "steps": steps,
                    "generated_calls": total_gen,
                    "served_calls": served,
                    "failed_calls": failed,
                    "perfect_calls": perfect,
                    "served_rate": round((served / total_gen * 100) if total_gen > 0 else 0.0, 2),
                    "failed_rate": round((failed / total_gen * 100) if total_gen > 0 else 0.0, 2),
                    "total_distance_km": round(float(total_dist), 2),
                    "avg_response_time": round(float(avg_response), 2),
                })

            # Tính trung bình cộng qua các seed
            n = len(ep_records)
            mean_reward = round(sum(r["total_reward"] for r in ep_records) / n, 2) if n > 0 else 0.0
            mean_served = round(sum(r["served_calls"] for r in ep_records) / n, 2) if n > 0 else 0.0
            mean_failed = round(sum(r["failed_calls"] for r in ep_records) / n, 2) if n > 0 else 0.0
            mean_perfect = round(sum(r["perfect_calls"] for r in ep_records) / n, 2) if n > 0 else 0.0
            mean_served_rate = round(sum(r["served_rate"] for r in ep_records) / n, 2) if n > 0 else 0.0
            mean_distance = round(sum(r["total_distance_km"] for r in ep_records) / n, 2) if n > 0 else 0.0
            mean_response = round(sum(r["avg_response_time"] for r in ep_records) / n, 2) if n > 0 else 0.0

            results[pol_id] = {
                "policy_id": pol_id,
                "name": pol_info.get("name", pol_id),
                "author": pol_info.get("author", ""),
                "type": pol_info.get("type", ""),
                "description": pol_info.get("description", ""),
                "badge_color": pol_info.get("badge_color", "#64748b"),
                "mean_reward": mean_reward,
                "mean_served_calls": mean_served,
                "mean_failed_calls": mean_failed,
                "mean_perfect_calls": mean_perfect,
                "mean_served_rate": mean_served_rate,
                "mean_distance_km": mean_distance,
                "mean_response_time": mean_response,
                "episodes": ep_records,
            }

        return {
            "status": "success",
            "seeds": seeds,
            "config": cfg,
            "results": results,
        }
