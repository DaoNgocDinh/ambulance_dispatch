"""
Định nghĩa thực thể Cung đường nối có hướng (DirectedEdge) giữa hai nút giao.
- Mỗi đường nối 2 chiều gồm 2 cung có hướng riêng biệt: (u -> v) và (v -> u).
- Độ dài (length) được tự động tính từ khoảng cách hình học giữa 2 nút.
- Biến độ tắc đường (congestion) nhận giá trị rời rạc in {1, 2, 3}:
    + 1: Thông thoáng (Clear / Tốc độ chuẩn ~50 km/h)
    + 2: Đông đúc (Moderate / Tốc độ trung bình ~30 km/h)
    + 3: Ùn tắc (Congested / Tốc độ chậm ~15 km/h)
- 2 chiều (u -> v) và (v -> u) có độ tắc HOÀN TOÀN ĐỘC LẬP và biến thiên theo thời gian thực.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


CONGESTION_CLEAR: int = 1       # 1: Thông thoáng
CONGESTION_MODERATE: int = 2    # 2: Bình thường
CONGESTION_HEAVY: int = 3       # 3: Tắc

CONGESTION_NAMES: Dict[int, str] = {
    CONGESTION_CLEAR: "Thông thoáng",
    CONGESTION_MODERATE: "Bình thường",
    CONGESTION_HEAVY: "Tắc",
}

# Vận tốc quy định theo từng vùng (km/h)
# "Vận tốc cho phép trong trung tâm luôn thấp hơn xa trung tâm"
SPEEDS_CENTER: Dict[int, float] = {
    CONGESTION_CLEAR: 40.0,
    CONGESTION_MODERATE: 25.0,
    CONGESTION_HEAVY: 12.0,
}

SPEEDS_OUTER: Dict[int, float] = {
    CONGESTION_CLEAR: 60.0,
    CONGESTION_MODERATE: 40.0,
    CONGESTION_HEAVY: 20.0,
}

# Tương thích ngược mặc định
CONGESTION_SPEEDS = SPEEDS_OUTER


@dataclass
class DirectedEdge:
    """
    Biểu diễn cung đường có hướng từ from_node -> to_node.
    """
    from_node: int
    to_node: int
    length: float                      # Chiều dài cung đường (km)
    congestion: int = CONGESTION_CLEAR # Độ tắc in {1, 2, 3}
    name: str = ""                     # Tên cung đường
    is_center: bool = False            # Tuyến đường thuộc vùng lõi trung tâm

    @property
    def edge_key(self) -> tuple[int, int]:
        """Khóa định danh cung đường có hướng (from_node, to_node)."""
        return (self.from_node, self.to_node)

    @property
    def speed(self) -> float:
        """Vận tốc lưu thông hiện tại (km/h) dựa theo mức độ tắc và phân vùng."""
        speeds = SPEEDS_CENTER if self.is_center else SPEEDS_OUTER
        return speeds.get(self.congestion, 30.0)

    @property
    def travel_time_hours(self) -> float:
        """Thời gian di chuyển dự kiến (giờ) = độ dài / tốc độ."""
        return self.length / self.speed

    @property
    def travel_time_minutes(self) -> float:
        """Thời gian di chuyển dự kiến (phút)."""
        return self.travel_time_hours * 60.0

    @property
    def congestion_name(self) -> str:
        """Mô tả mức độ tắc đường."""
        return CONGESTION_NAMES.get(self.congestion, "Không xác định")

    def set_congestion(self, level: int) -> None:
        """Gán thủ công mức độ tắc đường (1, 2, hoặc 3)."""
        if level not in (CONGESTION_CLEAR, CONGESTION_MODERATE, CONGESTION_HEAVY):
            raise ValueError(f"Độ tắc đường phải thuộc {{1, 2, 3}}, nhận được {level}")
        self.congestion = level

    def update_congestion_markov(
        self,
        rng: np.random.Generator,
        rush_hour: bool = False,
    ) -> int:
        """
        Cập nhật độ tắc đường ngẫu nhiên theo chuỗi Markov thời gian thực.
        - Tỉ lệ chuyển đổi trạng thái tắc tăng khi về gần trung tâm và giảm khi càng xa trung tâm.
        """
        curr = self.congestion - 1  # 0, 1, 2 cho indexing

        if self.is_center:
            # Gần trung tâm: Tỉ lệ tắc tăng cao
            if rush_hour:
                trans = [
                    [0.30, 0.40, 0.30],  # Từ 1 (Thông thoáng)
                    [0.10, 0.45, 0.45],  # Từ 2 (Bình thường)
                    [0.02, 0.28, 0.70],  # Từ 3 (Tắc)
                ]
            else:
                trans = [
                    [0.45, 0.35, 0.20],  # Từ 1 (Thông thoáng)
                    [0.20, 0.50, 0.30],  # Từ 2 (Bình thường)
                    [0.05, 0.35, 0.60],  # Từ 3 (Tắc)
                ]
        else:
            # Xa trung tâm: Tỉ lệ tắc giảm, thông thoáng cao hơn
            if rush_hour:
                trans = [
                    [0.55, 0.30, 0.15],  # Từ 1 (Thông thoáng)
                    [0.25, 0.45, 0.30],  # Từ 2 (Bình thường)
                    [0.10, 0.45, 0.45],  # Từ 3 (Tắc)
                ]
            else:
                trans = [
                    [0.70, 0.22, 0.08],  # Từ 1 (Thông thoáng)
                    [0.35, 0.45, 0.20],  # Từ 2 (Bình thường)
                    [0.20, 0.50, 0.30],  # Từ 3 (Tắc)
                ]

        next_idx = int(rng.choice([0, 1, 2], p=trans[curr]))
        self.congestion = next_idx + 1
        return self.congestion

    def __repr__(self) -> str:
        return (
            f"Edge({self.from_node}->{self.to_node}, dài={self.length:.2f}km, "
            f"tắc={self.congestion}[{self.congestion_name}], "
            f"t={self.travel_time_minutes:.2f}phút)"
        )
