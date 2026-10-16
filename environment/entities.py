"""
Các thực thể dữ liệu cốt lõi của hệ thống điều phối xe cứu thương:
- Ambulance: đại diện cho từng xe cứu thương k in {0, ..., K-1} với [p_k, st_k]
- EmergencyRequest: đại diện cho từng cuộc gọi cấp cứu m với [loc_m, prio_m, wait_time_m]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import (
    AMB_STATE_IDLE,
    AMB_STATE_NAMES,
    GOLDEN_TIME_STEPS,
    MAX_WAIT_TIME_STEPS,
    PRIO_NAMES,
)


@dataclass
class EmergencyRequest:
    """
    Biểu diễn một cuộc gọi cấp cứu trong hệ thống.
    Trong vector trạng thái X_req, mỗi cuộc gọi đang chờ gồm bộ 3:
        [loc_m, prio_m, wait_time_m]
    """

    req_id: int
    loc: int                  # Vị trí nút xảy ra sự cố: loc_m in {1, ..., 15}
    prio: int                 # Mức độ ưu tiên: 1 (Đỏ - Nguy kịch), 2 (Vàng - Khẩn cấp), 3 (Xanh - Tiêu chuẩn)
    wait_time: int = 0        # Thời gian bệnh nhân đã chờ (số bước thời gian)
    spawn_step: int = 0       # Bước thời gian cuộc gọi xuất hiện
    assigned_amb_id: Optional[int] = None
    response_time: Optional[int] = None

    @property
    def golden_time(self) -> int:
        """Thời gian vàng T_golden tương ứng với mức độ ưu tiên."""
        return GOLDEN_TIME_STEPS[self.prio]

    @property
    def max_wait_time(self) -> int:
        """Thời gian chờ tối đa T_max trước khi có nguy cơ tử vong."""
        return MAX_WAIT_TIME_STEPS[self.prio]

    @property
    def prio_name(self) -> str:
        return PRIO_NAMES.get(self.prio, "Không xác định")

    def to_state_triplet(self) -> list[int]:
        """Trả về [loc_m, prio_m, wait_time_m] theo đúng mục 4.2.1."""
        return [int(self.loc), int(self.prio), int(self.wait_time)]


@dataclass
class Ambulance:
    """
    Biểu diễn một xe cứu thương k trong đội xe K = 3 chiếc.
    Trong vector trạng thái X_amb, mỗi xe gồm bộ đôi:
        [p_k, st_k]
    với p_k in {1, ..., 15} và st_k in {0, 1, 2, 3, 4}.
    """

    amb_id: int               # Chỉ số xe k in {0, 1, 2}
    home_station: int         # Nút trạm trực mặc định (1, 8, hoặc 15)
    node: int                 # Vị trí nút hiện tại p_k in {1, ..., 15}
    status: int = AMB_STATE_IDLE  # Trạng thái hoạt động st_k in {0, 1, 2, 3, 4}
    target_node: Optional[int] = None
    moving_to_node: Optional[int] = None
    edge_progress: float = 0.0
    forced_next_node: Optional[int] = None
    active_request: Optional[EmergencyRequest] = None

    @property
    def is_idle(self) -> bool:
        """Xe đang rảnh (st_k == 0) hoặc đang tái định vị (có thể điều hướng nhận ca cấp cứu ngay)."""
        return self.status == AMB_STATE_IDLE

    @property
    def can_be_dispatched(self) -> bool:
        """Xe rảnh tại chỗ (st_k == 0) có thể được phân bổ ca cấp cứu."""
        return self.status == AMB_STATE_IDLE

    @property
    def status_name(self) -> str:
        return AMB_STATE_NAMES.get(self.status, "Không xác định")

    def reset(self, initial_node: Optional[int] = None) -> None:
        """Đặt lại trạng thái xe về trạm trực ban đầu."""
        self.node = initial_node if initial_node is not None else self.home_station
        self.status = AMB_STATE_IDLE
        self.target_node = None
        self.moving_to_node = None
        self.edge_progress = 0.0
        self.forced_next_node = None
        self.active_request = None

    def to_state_pair(self) -> list[int]:
        """Trả về [p_k, st_k] theo đúng mục 4.2.1."""
        return [int(self.node), int(self.status)]
