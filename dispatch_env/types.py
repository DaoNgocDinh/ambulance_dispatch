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

SEVERITY_CONFIG: Dict[int, Tuple[float, float]] = {
    1: (60.0, 4.0),  # Mức 1: thời gian thất bại = 60 phút, xử lý = 4 phút
    2: (40.0, 6.0),  # Mức 2: thời gian thất bại = 40 phút, xử lý = 6 phút
    3: (30.0, 8.0),  # Mức 3: thời gian thất bại = 30 phút, xử lý = 8 phút
}

# Cấu hình phần thưởng / hình phạt theo mức độ nghiêm trọng (1, 2, 3):
# | Mức độ | Đợi (điểm trừ/phút) w_c | Thất bại/Quá hạn p_c | Mỗi km đi | Hoàn thành | Hoàn hảo |
# |   1    |            1            |          30          |   0.05    |     50     |    70    |
# |   2    |            2            |          60          |   0.05    |     65     |    80    |
# |   3    |            4            |         120          |   0.05    |     70     |   100    |
REWARD_CONFIG: Dict[int, Dict[str, float]] = {
    1: {
        "wait_penalty_per_min": 1.0,
        "fail_penalty": 30.0,
        "complete_reward": 50.0,
        "perfect_reward": 70.0,
    },
    2: {
        "wait_penalty_per_min": 2.0,
        "fail_penalty": 60.0,
        "complete_reward": 65.0,
        "perfect_reward": 80.0,
    },
    3: {
        "wait_penalty_per_min": 4.0,
        "fail_penalty": 120.0,
        "complete_reward": 70.0,
        "perfect_reward": 100.0,
    },
}

DISTANCE_PENALTY_PER_KM: float = 0.05
