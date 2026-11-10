"""Lõi Q-learning dạng bảng, không phụ thuộc vào môi trường mô phỏng cụ thể.

Môi trường bên ngoài cung cấp state key, action ID hợp lệ và reward.
Lớp này chỉ thực hiện lựa chọn hành động, cập nhật Q-value và lưu bảng học.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


class QLearningAgent:
    """Lưu và cập nhật Q(s, a) cho từng cặp trạng thái - hành động.

    Q-table dùng khóa (state, action_id). Nếu một cặp chưa từng xuất hiện,
    giá trị Q ban đầu được xem là 0.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
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

        self.alpha = alpha  # Tốc độ học: mức độ Q-value mới ảnh hưởng lên giá trị cũ.
        self.gamma = gamma  # Hệ số chiết khấu: mức độ coi trọng phần thưởng tương lai.
        self.epsilon = epsilon  # Xác suất chọn ngẫu nhiên để tiếp tục khám phá.
        self.epsilon_min = epsilon_min  # Mức khám phá thấp nhất sau khi giảm dần.
        self.epsilon_decay = epsilon_decay  # Hệ số giảm epsilon sau mỗi episode.
        self.rng = random.Random(seed)  # Bộ sinh số riêng giúp kết quả có thể lặp lại.
        # Khóa là (state key, action ID); giá trị là mức tốt mà agent đã học được.
        self.q_table: Dict[Tuple[str, int], float] = {}

    def choose_action(
        self,
        state: str,
        valid_actions: Sequence[int],
        explore: bool = True,
    ) -> int:
        """Chọn hành động theo epsilon-greedy, chỉ trong danh sách hợp lệ.

        Khi explore=True, agent chọn ngẫu nhiên với xác suất epsilon để khám phá.
        Các lần còn lại chọn hành động có Q-value cao nhất (khai thác kiến thức).
        Khi hòa điểm, chọn ngẫu nhiên trong nhóm tốt nhất để tránh thiên lệch.
        """
        if not valid_actions:
            raise ValueError("Phải cung cấp ít nhất một action hợp lệ.")

        # Nhánh khám phá: thử một hành động hợp lệ bất kỳ.
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(list(valid_actions))

        # Nhánh khai thác: chọn giá trị Q lớn nhất trong các hành động được phép.
        best_value = max(self.q_table.get((state, action), 0.0) for action in valid_actions)
        best_actions = [
            action
            for action in valid_actions
            if self.q_table.get((state, action), 0.0) == best_value
        ]
        return self.rng.choice(best_actions)

    def update(
        self,
        state: str,
        action: int,
        reward: float,
        next_state: str,
        next_valid_actions: Sequence[int],
        done: bool,
    ) -> None:
        """Cập nhật Q(s,a) theo công thức Q-learning:

        Q mới = Q cũ + alpha * (mục tiêu - Q cũ)
        Mục tiêu = reward + gamma * max(Q(next_state, hành động kế tiếp))

        Chỉ xét next_valid_actions để không học từ hành động môi trường cấm.
        Nếu episode đã kết thúc hoặc không còn hành động tiếp theo, giá trị tương
        lai bằng 0; khi đó mục tiêu chỉ còn reward của bước hiện tại.
        """
        # Nếu chưa gặp cặp này, bắt đầu học từ Q-value bằng 0.
        old_value = self.q_table.get((state, action), 0.0)
        if done or not next_valid_actions:
            future_value = 0.0
        else:
            # Q-learning dùng hành động tương lai tốt nhất, không phụ thuộc
            # hành động mà policy epsilon-greedy sẽ thực sự chọn ở bước sau.
            future_value = max(
                self.q_table.get((next_state, next_action), 0.0)
                for next_action in next_valid_actions
            )

        # Tính mục tiêu Bellman rồi dịch Q cũ một phần alpha về phía mục tiêu.
        target = reward + self.gamma * future_value
        self.q_table[(state, action)] = old_value + self.alpha * (target - old_value)

    def decay_exploration(self) -> None:
        """Giảm xác suất khám phá sau mỗi episode, giữ epsilon không dưới giới hạn."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Path) -> None:
        """Lưu Q-table thành JSON để có thể tiếp tục dùng hoặc kiểm tra."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # Chuyển dictionary có khóa tuple thành danh sách object dễ lưu dưới dạng JSON.
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
            # Chỉ thay Q-table hiện tại sau khi từng dòng đã được xác nhận hợp lệ.
            loaded[(row["state"], row["action_id"])] = float(row["q_value"])
        self.q_table = loaded
