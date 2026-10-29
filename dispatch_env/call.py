"""
Định nghĩa thực thể Cuộc gọi cấp cứu (Emergency Call).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
from dispatch_env.types import CallStatus, SEVERITY_CONFIG


@dataclass
class EmergencyCall:
    """
    Thực thể Cuộc gọi cấp cứu.
    - call_id: Mã định danh cuộc gọi.
    - edge: Cung đường có hướng chứa vị trí tai nạn (chỉ ở 1 hướng đường).
    - distance_from_start: Vị trí trên cung đường tính từ điểm đi (km).
    - coord: Tọa độ địa lý (x, y) trên bản đồ (km).
    - severity: Mức độ nghiêm trọng (1, 2, 3).
    - created_time: Thời điểm phát sinh (phút).
    - failure_time: Khoảng thời gian cho phép trước khi thất bại (30, 40, 60 phút).
    - processing_time: Thời gian cần để nhân viên y tế sơ cứu/xử lý tại hiện trường (4, 6, 8 phút).
    - wait_time: Thời gian đã chờ xe cứu thương tới hiện trường (phút).
    - status: Trạng thái cuộc gọi.
    - assigned_ambulance_id: Mã xe đang nhận xử lý cuộc gọi (nếu có).
    - arrived_at_scene_time: Thời điểm xe cứu thương tới hiện trường (phút).
    - completed_time: Thời điểm cuộc gọi hoàn thành sơ cứu và xuất phát đi bệnh viện (phút).
    - is_perfect: Đạt hoàn thành hoàn hảo (xong trước thời điểm phát sinh + 50% thời gian thất bại).
    - failure_reason: Lý do thất bại ('timeout' hoặc 'queue_overflow').
    """
    call_id: int
    edge: Tuple[int, int]
    distance_from_start: float
    coord: Tuple[float, float]
    severity: int
    created_time: float
    failure_time: float = 30.0
    processing_time: float = 4.0
    wait_time: float = 0.0
    status: CallStatus = CallStatus.PENDING
    assigned_ambulance_id: Optional[int] = None
    arrived_at_scene_time: Optional[float] = None
    completed_time: Optional[float] = None
    is_perfect: bool = False
    failure_type: Optional[str] = None
    failure_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.severity in SEVERITY_CONFIG:
            fail_t, proc_t = SEVERITY_CONFIG[self.severity]
            self.failure_time = fail_t
            self.processing_time = proc_t

    @property
    def deadline(self) -> float:
        """Mốc thời gian thất bại tuyệt đối = thời điểm phát sinh + thời gian thất bại."""
        return self.created_time + self.failure_time

    @property
    def perfect_threshold(self) -> float:
        """Mốc thời gian để được tính là hoàn hảo = thời điểm phát sinh + 50% * thời gian thất bại."""
        return self.created_time + 0.5 * self.failure_time

    def update_wait_time(self, dt: float) -> None:
        """Cập nhật thời gian chờ đợi xe cứu thương nếu chưa được chăm sóc y tế."""
        if self.status in (CallStatus.PENDING, CallStatus.WAITING_FOR_AMBULANCE):
            self.wait_time += dt

    def check_failure(self, current_time: float) -> bool:
        """
        Kiểm tra nếu cuộc gọi bị quá hạn (chạm mốc thất bại khi đang chờ).
        Lưu ý: Khi đã ở trạng thái 'Đang xử lý' (AT_SCENE), mốc thời gian thất bại bị loại bỏ.
        """
        if self.status in (CallStatus.PENDING, CallStatus.WAITING_FOR_AMBULANCE):
            if current_time >= self.deadline:
                self.status = CallStatus.FAILED
                self.failure_type = "timeout"
                if self.assigned_ambulance_id is not None:
                    self.failure_reason = (
                        f"Quá hạn thời gian chờ cấp cứu (đã chờ {self.wait_time:.1f}/{self.failure_time:.0f} phút) "
                        f"trước khi Xe {self.assigned_ambulance_id} kịp tiếp cận hiện trường"
                    )
                else:
                    self.failure_reason = (
                        f"Quá hạn thời gian chờ cấp cứu (đã chờ {self.wait_time:.1f}/{self.failure_time:.0f} phút) "
                        f"mà chưa có xe cứu thương nào tiếp nhận"
                    )
                return True
        return False

    def mark_at_scene(self, arrival_time: float) -> None:
        """Đánh dấu xe cứu thương đã tới hiện trường."""
        self.status = CallStatus.AT_SCENE
        self.arrived_at_scene_time = arrival_time

    def mark_completed(self, complete_time: float) -> None:
        """
        Đánh dấu hoàn thành khi xe bắt đầu xuất phát đến bệnh viện.
        Kiểm tra nếu hoàn thành hoàn hảo.
        """
        self.status = CallStatus.COMPLETED
        self.completed_time = complete_time
        if complete_time <= self.perfect_threshold:
            self.is_perfect = True

    def mark_rejected(self) -> None:
        """Đánh dấu cuộc gọi bị từ chối do hàng chờ PENDING đã đầy 3 cuộc gọi."""
        self.status = CallStatus.FAILED
        self.failure_type = "queue_overflow"
        self.failure_reason = "Bị từ chối tiếp nhận do hàng chờ yêu cầu xử lý đã đạt tối đa 3 cuộc gọi (quá tải hệ thống)"

    def __repr__(self) -> str:
        loc_str = f"N{self.edge[0]}->N{self.edge[1]} (+{self.distance_from_start:.2f}km)"
        return (
            f"Call(ID={self.call_id}, mức={self.severity}, vị_trí={loc_str}, "
            f"trạng_thái='{self.status.value}', đã_chờ={self.wait_time:.1f}p/{self.failure_time:.1f}p, "
            f"xe={self.assigned_ambulance_id})"
        )
