"""Lõi Q-learning dạng bảng, không phụ thuộc vào môi trường mô phỏng cụ thể.

Môi trường bên ngoài cung cấp state key, action ID hợp lệ và reward.
Lớp này chỉ thực hiện lựa chọn hành động, cập nhật Q-value và lưu bảng học.
"""

from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple


ReplayTransition = Tuple[str, int, float, str, Tuple[int, ...], bool, int]


class QLearningAgent:
    """Lưu và cập nhật Q(s, a) cho từng cặp trạng thái - hành động.

    Q-table dùng khóa (state, action_key). Nếu một cặp chưa từng xuất hiện,
    giá trị Q ban đầu được khởi tạo bằng ``initial_q`` (mặc định 0.0).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
        *,
        adaptive_alpha: bool = False,
        alpha_min: float = 0.02,
        alpha_decay: float = 0.05,
        initial_q: float = 0.0,
        replay_capacity: int = 0,
        replay_batch_size: int = 0,
    ) -> None:
        # Kiểm tra tham số trước khi tạo agent để tránh chạy với cấu hình sai.
        if not 0 < alpha <= 1:
            raise ValueError("alpha phải nằm trong (0, 1].")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma phải nằm trong [0, 1].")
        if not 0 <= epsilon <= 1 or not 0 <= epsilon_min <= 1:
            raise ValueError("epsilon và epsilon_min phải nằm trong [0, 1].")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay phải nằm trong (0, 1].")
        if not 0 < alpha_min <= alpha:
            raise ValueError("alpha_min phải nằm trong (0, alpha].")
        if alpha_decay < 0:
            raise ValueError("alpha_decay phải không âm.")
        if replay_capacity < 0 or replay_batch_size < 0:
            raise ValueError("replay_capacity và replay_batch_size phải không âm.")

        self.alpha = alpha  # Tốc độ học ban đầu.
        self.alpha_min = alpha_min  # Ngưỡng tốc độ học thấp nhất khi giảm theo số lần thăm.
        self.alpha_decay = alpha_decay  # Tốc độ giảm alpha theo số lần thăm (s, a).
        self.adaptive_alpha = adaptive_alpha  # Bật giảm alpha theo visit count (Robbins-Monro).
        self.gamma = gamma  # Hệ số chiết khấu: mức độ coi trọng phần thưởng tương lai.
        self.epsilon = epsilon  # Xác suất chọn ngẫu nhiên để tiếp tục khám phá.
        self.epsilon_min = epsilon_min  # Mức khám phá thấp nhất sau khi giảm dần.
        self.epsilon_decay = epsilon_decay  # Hệ số giảm epsilon sau mỗi episode.
        self.initial_q = initial_q  # Giá trị Q khởi tạo cho cặp (s, a) chưa từng gặp.
        self.replay_capacity = replay_capacity
        self.replay_batch_size = replay_batch_size

        self.rng = random.Random(seed)  # Bộ sinh số riêng giúp kết quả có thể lặp lại.
        # Khóa là (state key, action key); giá trị là mức tốt mà agent đã học được.
        self.q_table: Dict[Tuple[str, int], float] = {}
        self.visit_counts: Dict[Tuple[str, int], int] = {}
        self.replay_buffer: Deque[ReplayTransition] = deque(maxlen=replay_capacity)

    @staticmethod
    def _resolve_key(action: int, action_keys: Optional[Mapping[int, int]]) -> int:
        if action_keys is None:
            return action
        return action_keys.get(action, action)

    def choose_action(
        self,
        state: str,
        valid_actions: Sequence[int],
        explore: bool = True,
        action_keys: Optional[Mapping[int, int]] = None,
    ) -> int:
        """Chọn hành động theo epsilon-greedy, chỉ trong danh sách hợp lệ.

        Khi ``action_keys`` được cung cấp, nhiều hành động thô đối xứng có thể
        dùng chung một canonical action key trong Q-table để khái quát hóa tốt hơn.
        """
        if not valid_actions:
            raise ValueError("Phải cung cấp ít nhất một action hợp lệ.")

        # Nhánh khám phá: chọn đều theo nhóm canonical key nếu có ánh xạ đối xứng.
        if explore and self.rng.random() < self.epsilon:
            if action_keys is None:
                return self.rng.choice(list(valid_actions))
            grouped: Dict[int, List[int]] = {}
            for action in valid_actions:
                key = self._resolve_key(action, action_keys)
                grouped.setdefault(key, []).append(action)
            chosen_key = self.rng.choice(list(grouped.keys()))
            return self.rng.choice(grouped[chosen_key])

        # Nhánh khai thác: chọn giá trị Q lớn nhất trong các hành động được phép.
        q_values = [
            (action, self.q_table.get((state, self._resolve_key(action, action_keys)), self.initial_q))
            for action in valid_actions
        ]
        best_value = max(value for _, value in q_values)
        best_actions = [action for action, value in q_values if value == best_value]
        if not explore:
            dispatch_bests = [a for a in best_actions if a != 0]
            if dispatch_bests:
                return dispatch_bests[0]
            return best_actions[0]
        return self.rng.choice(best_actions)

    def _effective_alpha(self, state: str, key: int) -> float:
        if not self.adaptive_alpha:
            return self.alpha
        visits = self.visit_counts.get((state, key), 1)
        decayed = self.alpha / (1.0 + self.alpha_decay * max(0, visits - 1))
        return max(self.alpha_min, decayed)

    def _apply_bellman_update(
        self,
        state: str,
        key: int,
        reward: float,
        next_state: str,
        next_keys: Sequence[int],
        done: bool,
        steps: int,
        learning_rate: float,
    ) -> None:
        old_value = self.q_table.get((state, key), self.initial_q)
        if done or not next_keys:
            future_value = 0.0
        else:
            future_value = max(
                self.q_table.get((next_state, next_key), self.initial_q)
                for next_key in next_keys
            )

        discount = self.gamma ** max(1, steps)
        target = reward + discount * future_value
        self.q_table[(state, key)] = old_value + learning_rate * (target - old_value)

    def update(
        self,
        state: str,
        action: int,
        reward: float,
        next_state: str,
        next_valid_actions: Sequence[int],
        done: bool,
        *,
        steps: int = 1,
        action_key: Optional[int] = None,
        next_action_keys: Optional[Mapping[int, int]] = None,
        record_experience: bool = True,
    ) -> None:
        """Cập nhật Q(s, a) theo công thức Q-learning (hỗ trợ Semi-MDP nhiều phút ``steps``).

        Q mới = Q cũ + alpha * (reward + gamma^steps * max Q(next_state, a') - Q cũ)
        """
        key = action if action_key is None else action_key
        table_key = (state, key)
        self.visit_counts[table_key] = self.visit_counts.get(table_key, 0) + 1
        learning_rate = self._effective_alpha(state, key)

        resolved_next_keys = tuple(
            dict.fromkeys(
                self._resolve_key(next_action, next_action_keys)
                for next_action in next_valid_actions
            )
        )
        self._apply_bellman_update(
            state=state,
            key=key,
            reward=reward,
            next_state=next_state,
            next_keys=resolved_next_keys,
            done=done,
            steps=steps,
            learning_rate=learning_rate,
        )

        if record_experience and self.replay_capacity > 0:
            self.replay_buffer.append(
                (state, key, float(reward), next_state, resolved_next_keys, bool(done), max(1, int(steps)))
            )

    def replay_experiences(self, batch_size: Optional[int] = None) -> int:
        """Ôn tập ngẫu nhiên các trải nghiệm đã lưu để lan truyền phần thưởng trễ nhanh hơn."""
        if not self.replay_buffer:
            return 0
        count = self.replay_batch_size if batch_size is None else batch_size
        if count <= 0:
            return 0

        sample_size = min(count, len(self.replay_buffer))
        transitions = self.rng.sample(list(self.replay_buffer), sample_size)
        for state, key, reward, next_state, next_keys, done, steps in transitions:
            learning_rate = max(self.alpha_min, 0.5 * self._effective_alpha(state, key))
            self._apply_bellman_update(
                state=state,
                key=key,
                reward=reward,
                next_state=next_state,
                next_keys=next_keys,
                done=done,
                steps=steps,
                learning_rate=learning_rate,
            )
        return sample_size

    def decay_exploration(self) -> None:
        """Giảm xác suất khám phá sau mỗi episode, giữ epsilon không dưới giới hạn."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Path) -> None:
        """Lưu Q-table thành JSON để có thể tiếp tục dùng hoặc kiểm tra."""
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"state": state, "action_id": action, "q_value": value}
            for (state, action), value in self.q_table.items()
        ]
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path) -> None:
        """Đọc Q-table JSON và báo lỗi rõ nếu cấu trúc dữ liệu không đúng."""
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("Q-table phải là danh sách các dòng JSON.")

        loaded: Dict[Tuple[str, int], float] = {}
        for row in rows:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("state"), str)
                or not isinstance(row.get("action_id"), int)
                or not isinstance(row.get("q_value"), (int, float))
            ):
                raise ValueError("Dòng dữ liệu không hợp lệ trong Q-table.")
            loaded[(row["state"], row["action_id"])] = float(row["q_value"])
        self.q_table = loaded
