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
CONGESTION_MODERATE: int = 2    # 2: Đông đúc
CONGESTION_HEAVY: int = 3       # 3: Ùn tắc

CONGESTION_NAMES: Dict[int, str] = {
    CONGESTION_CLEAR: "Thông thoáng (1)",
    CONGESTION_MODERATE: "Đông đúc (2)",
    CONGESTION_HEAVY: "Ùn tắc (3)",
}

# Tốc độ di chuyển ước tính theo độ tắc đường (km/h)
CONGESTION_SPEEDS: Dict[int, float] = {
    CONGESTION_CLEAR: 50.0,
    CONGESTION_MODERATE: 30.0,
    CONGESTION_HEAVY: 15.0,
}


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

    @property
    def edge_key(self) -> tuple[int, int]:
        """Khóa định danh cung đường có hướng (from_node, to_node)."""
        return (self.from_node, self.to_node)

    @property
    def speed(self) -> float:
        """Vận tốc lưu thông hiện tại (km/h) dựa theo mức độ tắc."""
        return CONGESTION_SPEEDS.get(self.congestion, 40.0)

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
        Phân biệt xác suất trong giờ bình thường và giờ cao điểm.
        """
        curr = self.congestion - 1  # 0, 1, 2 cho indexing

        if rush_hour:
            # Giờ cao điểm: xác suất tăng tắc đường cao hơn
            # Ma trận chuyển trạng thái P(c_{t+1} | c_t)
            trans = [
                [0.40, 0.45, 0.15],  # Từ 1 (Thông thoáng)
                [0.15, 0.50, 0.35],  # Từ 2 (Đông)
                [0.05, 0.35, 0.60],  # Từ 3 (Ùn tắc)
            ]
        else:
            # Giờ bình thường: xác suất thông thoáng cao hơn
            trans = [
                [0.75, 0.20, 0.05],  # Từ 1 (Thông thoáng)
                [0.35, 0.50, 0.15],  # Từ 2 (Đông)
                [0.20, 0.50, 0.30],  # Từ 3 (Ùn tắc)
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
