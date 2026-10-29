"""
Định nghĩa các Enum và Hằng số cho Môi trường Điều phối Xe Cứu Thương.
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, Tuple


class AmbulanceStatus(str, Enum):
    IDLE = "Rảnh"
    EN_ROUTE_TO_CALL = "Đang đến cuộc gọi"
    AT_SCENE = "Xử lý tại cuộc gọi"
    EN_ROUTE_TO_HOSPITAL = "Đang tới bệnh viện"
    AT_HOSPITAL = "Xử lý tại bệnh viện"


class CallStatus(str, Enum):
    PENDING = "Yêu cầu xử lý"
    WAITING_FOR_AMBULANCE = "Đợi xe cứu thương"
    AT_SCENE = "Đang xử lý"
    COMPLETED = "Hoàn thành"
    FAILED = "Thất bại"


# Cấu hình Mức độ nghiêm trọng (1, 2, 3)
# - Thời gian thất bại (30, 40, 60) phút
# - Thời gian cần để xử lý tại hiện trường (4, 6, 8) phút
#   (độ nghiêm trọng càng cao thời gian cần xử lý càng tăng)
SEVERITY_CONFIG: Dict[int, Tuple[float, float]] = {
    1: (30.0, 4.0),  # Mức 1: thời gian thất bại = 30 phút, xử lý = 4 phút
    2: (40.0, 6.0),  # Mức 2: thời gian thất bại = 40 phút, xử lý = 6 phút
    3: (60.0, 8.0),  # Mức 3: thời gian thất bại = 60 phút, xử lý = 8 phút
}
