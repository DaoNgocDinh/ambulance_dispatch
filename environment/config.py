"""
Cấu hình tham số cho Môi trường Điều phối Xe cứu thương Thành phố Thông minh (MDP).
Bám sát nội dung Thẻ 1 của báo cáo với cấu hình:
- 15 nút giao thông (|V| = 15)
- 3 xe cứu thương (K = 3)
- Tối đa 3 cuộc gọi cấp cứu chờ đồng thời (M = 3)
- 5 tuyến đường huyết mạch (|E_arterial| = 5)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ============================================================================
# 1. CẤU HÌNH QUY MÔ MÔ HÌNH (THEO YÊU CẦU)
# ============================================================================
NUM_NODES: int = 15                # |V| = 15 nút (đánh số 1..15 trong báo cáo)
NUM_AMBULANCES: int = 3            # K = 3 xe cứu thương
MAX_PENDING_REQUESTS: int = 3      # M = 3 cuộc gọi cấp cứu đồng thời tối đa
NUM_ARTERIAL_ROADS: int = 5        # 5 tuyến đường huyết mạch biến thiên giao thông

# Các nút đặc biệt (1-indexed từ 1 đến 15)
STATION_NODES: List[int] = [1, 8, 15]   # 3 trạm trực chiến lược (Tây Bắc, Trung tâm, Đông Nam)
HOSPITAL_NODES: List[int] = [3, 13]     # 2 bệnh viện đa khoa (Bệnh viện Bắc, Bệnh viện Nam)


# ============================================================================
# 2. TRẠNG THÁI XE CỨU THƯƠNG & MỨC ĐỘ KHẨN CẤP & GIAO THÔNG (MỤC 4.2.1)
# ============================================================================
# st_k in {0, 1, 2, 3, 4}
AMB_STATE_IDLE: int = 0            # 0: Rảnh / Chờ lệnh tại trạm hoặc nút hiện tại
AMB_STATE_TO_SCENE: int = 1        # 1: Đang di chuyển đi đón bệnh nhân
AMB_STATE_ON_SCENE: int = 2        # 2: Đang sơ cứu / tiếp nhận tại hiện trường
AMB_STATE_TO_HOSPITAL: int = 3     # 3: Đang chuyển bệnh nhân tới bệnh viện
AMB_STATE_RELOCATING: int = 4      # 4: Đang di chuyển tái định vị về trạm trực

AMB_STATE_NAMES: Dict[int, str] = {
    AMB_STATE_IDLE: "Rảnh (Idle)",
    AMB_STATE_TO_SCENE: "Đi đón (En Route)",
    AMB_STATE_ON_SCENE: "Tại hiện trường (On Scene)",
    AMB_STATE_TO_HOSPITAL: "Chuyển viện (To Hospital)",
    AMB_STATE_RELOCATING: "Tái định vị (Relocating)",
}

# Mức độ ưu tiên y tế prio_m in {1: Đỏ, 2: Vàng, 3: Xanh}
PRIO_NONE: int = 0
PRIO_RED: int = 1                  # Mức 1: Đỏ - Nguy kịch (Critical)
PRIO_YELLOW: int = 2               # Mức 2: Vàng - Khẩn cấp (Urgent)
PRIO_GREEN: int = 3                # Mức 3: Xanh - Tiêu chuẩn (Standard)

PRIO_NAMES: Dict[int, str] = {
    PRIO_NONE: "Trống",
    PRIO_RED: "Đỏ (Nguy kịch)",
    PRIO_YELLOW: "Vàng (Khẩn cấp)",
    PRIO_GREEN: "Xanh (Tiêu chuẩn)",
}

# Xác suất xuất hiện các mức độ ưu tiên khi có cuộc gọi mới
PRIO_PROBABILITIES: Dict[int, float] = {
    PRIO_RED: 0.20,                # 20% ca nguy kịch (Đỏ)
    PRIO_YELLOW: 0.35,             # 35% ca khẩn cấp (Vàng)
    PRIO_GREEN: 0.45,              # 45% ca tiêu chuẩn (Xanh)
}

# Thời gian vàng (T_golden) và Thời gian chờ tối đa (T_max) theo từng mức độ (đơn vị: bước thời gian)
GOLDEN_TIME_STEPS: Dict[int, int] = {
    PRIO_RED: 4,                   # Ca Đỏ cần tiếp cận rất nhanh (<= 4 bước)
    PRIO_YELLOW: 6,                # Ca Vàng (<= 6 bước)
    PRIO_GREEN: 8,                 # Ca Xanh (<= 8 bước)
}

MAX_WAIT_TIME_STEPS: Dict[int, int] = {
    PRIO_RED: 7,                   # Quá 7 bước -> nguy cơ tử vong cao (-100)
    PRIO_YELLOW: 10,               # Quá 10 bước -> quá hạn tối đa (-100)
    PRIO_GREEN: 14,                # Quá 14 bước -> quá hạn tối đa (-100)
}

# Mức độ lưu thông c_e in {0: Thông thoáng, 1: Đông, 2: Ùn tắc}
TRAFFIC_CLEAR: int = 0             # 0: Thông thoáng
TRAFFIC_BUSY: int = 1              # 1: Đông
TRAFFIC_CONGESTED: int = 2         # 2: Ùn tắc

TRAFFIC_NAMES: Dict[int, str] = {
    TRAFFIC_CLEAR: "Thông thoáng",
    TRAFFIC_BUSY: "Đông",
    TRAFFIC_CONGESTED: "Ùn tắc",
}


# ============================================================================
# 3. HÀM PHẦN THƯỞNG ĐA MỤC TIÊU R(s, a) (MỤC 4.2.3)
# ============================================================================
REWARD_GOLDEN_TIME: float = 50.0           # +50: Tiếp cận trong Thời gian vàng (T_resp <= T_golden)
REWARD_HOSPITAL_DELIVERY: float = 20.0     # +20: Chuyển viện an toàn thành công
REWARD_OPTIMAL_DIRECTION: float = 1.0      # +1: Di chuyển đúng hướng tối ưu (tiến gần hiện trường/đích)

# Phạt bệnh nhân phải chờ đợi trên mỗi bước thời gian
PENALTY_WAIT_PER_STEP: Dict[int, float] = {
    PRIO_RED: -5.0,                        # Mức 1 (Đỏ - Nguy kịch): -5 / bước
    PRIO_YELLOW: -3.0,                     # Mức 2 (Vàng - Khẩn cấp): -3 / bước
    PRIO_GREEN: -1.0,                      # Mức 3 (Xanh - Tiêu chuẩn): -1 / bước
}

PENALTY_MAX_WAIT_EXCEEDED: float = -100.0  # -100: Chờ quá thời gian tối đa (T_wait > T_max)
PENALTY_CONGESTED_ROAD: float = -2.0       # -2 / bước: Xe đi vào cung đường đang ùn tắc
PENALTY_EMPTY_STATIONS: float = -5.0       # -5: Làm trống hoàn toàn trạm trực (mất độ phủ an toàn)
PENALTY_INVALID_ACTION: float = -10.0      # Phạt khi chọn hành động không hợp lệ (nếu không dùng mask)


# ============================================================================
# 4. ĐỘNG HỌC GIAO THÔNG, QUÁ TRÌNH POISSON & CHIẾT KHẤU (MỤC 4.2.4 & 4.2.5)
# ============================================================================
DEFAULT_GAMMA_TABULAR: float = 0.95        # Hệ số chiết khấu cho DP, Monte Carlo, Q-Learning
DEFAULT_GAMMA_PPO: float = 0.99            # Hệ số chiết khấu cho PPO

V_MAX: float = 1.0                         # Vận tốc chuẩn tối đa (1 cạnh chuẩn / bước khi thông thoáng)
ALPHA_CONGESTION: float = 0.35             # Hệ số suy giảm vận tốc do tắc nghẽn: v = v_max * (1 - alpha * cong)
NOISE_STD: float = 0.05                    # Độ lệch chuẩn nhiễu Gaussian/Log-Normal của vận tốc


@dataclass
class EnvConfig:
    """Lớp lưu trữ toàn bộ cấu hình của môi trường để dễ dàng tùy chỉnh khi thực nghiệm."""

    num_nodes: int = NUM_NODES
    num_ambulances: int = NUM_AMBULANCES
    max_pending_requests: int = MAX_PENDING_REQUESTS
    num_arterial_roads: int = NUM_ARTERIAL_ROADS

    station_nodes: List[int] = field(default_factory=lambda: list(STATION_NODES))
    hospital_nodes: List[int] = field(default_factory=lambda: list(HOSPITAL_NODES))

    max_steps: int = 100                   # Số bước thời gian tối đa trong 1 episode (ví dụ 1 ca trực)
    delta_t: float = 1.0                   # Bước thời gian rời rạc Delta_t
    base_arrival_rate: float = 0.35        # Cường độ sinh cuộc gọi trung bình toàn thành phố mỗi bước
    noise_distribution: str = "gaussian"   # "gaussian" hoặc "lognormal" hoặc "none"
    gamma: float = DEFAULT_GAMMA_TABULAR
