"""
Định nghĩa thực thể Bệnh viện (Hospital) nằm trên DUY NHẤT 1 cung đường có hướng.
Yêu cầu cập nhật:
- "Bệnh viện chỉ ở 1 đường không ở 2 đường 1 lúc" -> Chỉ thuộc 1 cung có hướng (from_node -> to_node).
- "Điểm trên đường tính bằng độ dài từ điểm đi" -> Vị trí được đo bằng khoảng cách (km) tính từ điểm xuất phát (from_node).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class HospitalOnRoad:
    """
    Bệnh viện nằm trên DUY NHẤT 1 cung đường có hướng: from_node -> to_node.
    - hospital_id: Mã định danh (1 hoặc 2).
    - name: Tên bệnh viện.
    - from_node: Điểm xuất phát của cung đường (điểm đi).
    - to_node: Điểm đến của cung đường (điểm đến).
    - road_length: Chiều dài toàn bộ cung đường (km).
    - distance_from_start: Khoảng cách từ điểm đi (from_node) đến bệnh viện (km).
    - coord: Tọa độ (x, y) trên bản đồ (km), tính bằng nội suy từ điểm đi.
    """
    hospital_id: int
    name: str
    from_node: int
    to_node: int
    road_length: float
    distance_from_start: float
    coord: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not (0.0 < self.distance_from_start < self.road_length):
            raise ValueError(
                f"Khoảng cách từ điểm đi ({self.distance_from_start} km) phải nằm trong khoảng (0, {self.road_length} km)."
            )

    @property
    def distance_to_end(self) -> float:
        """Khoảng cách còn lại từ bệnh viện đến điểm đến (to_node) (km)."""
        return round(self.road_length - self.distance_from_start, 2)

    @property
    def offset_ratio(self) -> float:
        """Tỉ lệ khoảng cách từ điểm đi [0..1]."""
        return round(self.distance_from_start / self.road_length, 3)

    @property
    def directed_edge_key(self) -> Tuple[int, int]:
        """Cung đường có hướng duy nhất chứa bệnh viện."""
        return (self.from_node, self.to_node)

    def __repr__(self) -> str:
        return (
            f"Hospital(id={self.hospital_id}, name='{self.name}', "
            f"cung_đường={self.from_node}->{self.to_node}, "
            f"cách_điểm_đi({self.from_node})={self.distance_from_start:.2f}km, "
            f"cách_điểm_đến({self.to_node})={self.distance_to_end:.2f}km, "
            f"tọa_độ=({self.coord[0]:.2f}, {self.coord[1]:.2f}))"
        )
