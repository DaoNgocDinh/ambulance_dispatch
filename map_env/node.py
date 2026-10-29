"""
Định nghĩa thực thể Nút giao thông (Node) trong mạng lưới đô thị.
Mỗi nút tượng trưng cho một điểm giao của các cung đường:
- Đường cụt (Dead-end): bậc 1 (kết nối với 1 nút khác)
- Đường rẽ / Nối tiếp (Turn / Pass-through): bậc 2 (kết nối với 2 nút khác)
- Ngã 3 (T-junction / 3-way intersection): bậc 3 (kết nối với 3 nút khác)
- Ngã 4 (Crossroads / 4-way intersection): bậc 4 (kết nối với 4 nút khác)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class IntersectionType(str, Enum):
    """Phân loại kiểu điểm giao cắt giao thông."""
    DEAD_END = "Đường cụt"         # Bậc 1
    TURN = "Đường rẽ / Nối tiếp"   # Bậc 2
    THREE_WAY = "Ngã 3"            # Bậc 3
    FOUR_WAY = "Ngã 4"             # Bậc 4
    COMPLEX = "Nút giao phức hợp"  # Bậc >= 5


@dataclass
class MapNode:
    """
    Biểu diễn một nút giao thông trong mạng lưới.
    - node_id: Đánh số từ 1 đến n (1..20).
    - x, y: Tọa độ địa lý trên mặt phẳng đô thị (đơn vị: km).
    - name: Tên mô tả khu vực / nút giao.
    """
    node_id: int
    x: float
    y: float
    name: str = ""
    neighbors: List[int] = field(default_factory=list)

    @property
    def degree(self) -> int:
        """Bậc của nút (số lượng nút kề kết nối)."""
        return len(self.neighbors)

    @property
    def intersection_type(self) -> IntersectionType:
        """Xác định kiểu điểm giao cắt dựa trên số lượng đường nối kề."""
        deg = self.degree
        if deg == 1:
            return IntersectionType.DEAD_END
        elif deg == 2:
            return IntersectionType.TURN
        elif deg == 3:
            return IntersectionType.THREE_WAY
        elif deg == 4:
            return IntersectionType.FOUR_WAY
        else:
            return IntersectionType.COMPLEX

    @property
    def coord(self) -> Tuple[float, float]:
        """Tọa độ (x, y) của nút."""
        return (self.x, self.y)

    def add_neighbor(self, neighbor_id: int) -> None:
        """Thêm một nút kề nếu chưa tồn tại."""
        if neighbor_id not in self.neighbors:
            self.neighbors.append(neighbor_id)
            self.neighbors.sort()

    def __repr__(self) -> str:
        return (
            f"Node({self.node_id}, name='{self.name}', type='{self.intersection_type.value}', "
            f"bậc={self.degree}, kề={self.neighbors})"
        )
