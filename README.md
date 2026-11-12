# Môi Trường Bản Đồ Đô Thị 80 Nút & Mô Phỏng Điều Phối Xe Cứu Thương (Ambulance Dispatch Simulation)

Hệ thống mô phỏng điều phối xe cứu thương thông minh trên nền tảng mạng lưới giao thông **80 NÚT GIAO THÔNG** (86 tuyến đường 2 chiều = **172 cung đường có hướng**, `edge_id` từ $1$ đến $172$), **6 XE CỨU THƯƠNG** (mặc định), **4 BỆNH VIỆN**, và cơ chế **sinh cuộc gọi cấp cứu Poisson** ($\lambda = 0.1$) với quy trình ra quyết định theo tiến trình thời gian Markov (MDP).

---

## 1. Bảng Tổng Hợp Các Biến & Tham Số Chuẩn (Đồng Bộ Với `run_simulation.py`)

Toàn bộ các file trong dự án ([`run_simulation.py`](run_simulation.py), [`dispatch_env/types.py`](dispatch_env/types.py), [`dispatch_env/environment.py`](dispatch_env/environment.py), [`map_env/city_map.py`](map_env/city_map.py)) sử dụng chung bộ tham số mặc định:

| Tên biến / Hằng số | Giá trị mặc định | Ý nghĩa |
| :--- | :---: | :--- |
| `num_ambulances` (`DEFAULT_NUM_AMBULANCES`) | `6` | Số lượng xe cứu thương hoạt động trong môi trường |
| `max_time` (`DEFAULT_MAX_TIME`) | `360` phút | Thời gian mô phỏng sinh cuộc gọi tối đa ($6.0$ giờ) |
| `max_overtime` (`DEFAULT_MAX_OVERTIME`) | `300` phút | Thời gian quá giờ tối đa sau `max_time` trước khi dừng và báo `Truncated` (tổng tối đa $660$ phút) |
| `poisson_lambda` (`DEFAULT_POISSON_LAMBDA`) | `0.1` | Tốc độ sinh cuộc gọi trung bình mỗi phút theo phân phối Poisson |
| `max_pending_calls` (`DEFAULT_MAX_PENDING_CALLS`) | `3` | Số cuộc gọi tối đa được phép nằm trong hàng chờ `PENDING` (tương ứng $3$ Slot cố định `Slot 0..2`) |
| `seed` (`DEFAULT_SEED`) | `42` | Hạt giống sinh số ngẫu nhiên khởi tạo môi trường |
| `num_nodes` | `80` | Tổng số nút giao thông trên bản đồ (`1..80`, gồm nút trung tâm `1..20` và ngoại ô `21..80`) |
| `num_directed_edges` | `172` | Tổng số cung đường có hướng $(u \to v)$ với ID số nguyên `edge_id` từ $1$ đến $172$ và độ tắc độc lập $\{1, 2, 3\}$ |
| `num_hospitals` | `4` | Số bệnh viện ($2$ gần trung tâm: BV 1, BV 2; $2$ xa trung tâm: BV 3, BV 4) |
| `num_potential_actions` | `229` (với $6$ xe, $3$ slot)<br>`73` (với $4$ xe, $3$ slot) | Tổng số cách ghép hành động đồng thời tiềm năng (bao gồm cả hành động không phân công `()`) |

---

## 2. Cấu Trúc Tổng Quan Thực Thể Hệ Thống

### 2.1. Cung Đường Định Danh Số Nguyên (`DirectedEdge`, `edge_id = 1..172`)
- **Định danh số nguyên:** Mỗi cung đường có hướng $(u \to v)$ được gán duy nhất một `edge_id` số nguyên từ $1$ đến $172$.
- **Ánh xạ 2 chiều:**
  - `city_map.edge_to_id[(u, v)] -> int`: Lấy ID từ cặp nút.
  - `city_map.id_to_edge[edge_id] -> (u, v)`: Lấy cặp nút từ ID.
  - `city_map.edges_by_id[edge_id] -> DirectedEdge`: Truy xuất đối tượng cung đường trực tiếp qua ID.
- **Thiết lập độ tắc đường:** Hỗ trợ gọi linh hoạt qua cả `(u, v)` hoặc `edge_id`:
  ```python
  city_map.set_edge_congestion(edge_id, level=3)
  city_map.set_edge_congestion(u, v, level=2)
  ```

### 2.2. Xe Cứu Thương (`Ambulance` - Mặc định 6 xe)
- **Mã định danh:** `Xe 1 .. Xe 6` (tùy biến qua `num_ambulances`).
- **Vị trí hiện tại:** `current_edge = (u, v)`, `current_edge_id` ($1..172$), `distance_from_start` (km), `coord = (x, y)`.
- **Trạng thái xe (`AmbulanceStatus`):**
  - `Rảnh (IDLE)`: Hiện không xử lý cuộc gọi nào, sẵn sàng nhận nhiệm vụ mới. *(Trạng thái ban đầu các xe luôn Rảnh)*.
  - `Đang đến cuộc gọi (EN_ROUTE_TO_CALL)`: Xe đang di chuyển tới vị trí nạn nhân.
  - `Xử lý tại cuộc gọi (AT_SCENE)`: Xe đã tới hiện trường và nhân viên y tế đang sơ cứu.
  - `Đang tới bệnh viện (EN_ROUTE_TO_HOSPITAL)`: Xe chở bệnh nhân di chuyển tới bệnh viện tối ưu.
  - `Xử lý tại bệnh viện (AT_HOSPITAL)`: Bàn giao bệnh nhân tại cổng cấp cứu bệnh viện.
- **Biến mục tiêu & thời gian chi tiết trong Ambulance:**
  - `target_type` (và alias `targer_type`): Xác định đích đến hiện tại là `"call"` (cuộc gọi), `"hospital"` (bệnh viện) hoặc `None` (khi rảnh).
  - `target_edge` & `target_edge_id`: Cung đường có hướng chứa đích đến.
  - `target_dist`: Vị trí đích đến trên cung đường (km tính từ nút bắt đầu).
  - `assigned_hospital_id`: ID bệnh viện tiếp nhận ($1..4$) hoặc `None`.
  - `hospital_handover_time`: Thời gian bàn giao cần thiết tại bệnh viện ($3.0 - 5.0$ phút).
  - `scene_time_remaining`: Thời gian sơ cứu tại hiện trường còn lại, quyết định khi nào xe sơ cứu xong và chuyển sang di chuyển đến bệnh viện.
  - `hospital_time_remaining`: Thời gian bàn giao tại bệnh viện còn lại, quyết định khi nào xe rảnh trở lại (`IDLE`).
  - `route_edge_ids`: Danh sách các `edge_id` trong lộ trình đang di chuyển.
- **Cơ chế thu hồi nhiệm vụ:** Khi cuộc gọi bị xoá (thất bại/quá hạn) trong lúc xe đang trên đường tiếp cận, xe sẽ **lập tức trở về trạng thái Rảnh (`IDLE`)** và sẵn sàng nhận cuộc gọi mới.

### 2.3. Cuộc Gọi Cấp Cứu (`EmergencyCall`)
- **Thời điểm xuất hiện:** Xuất hiện vào **đầu mỗi phút**. Tại $t = 0$ có thể sinh cuộc gọi ngay từ đầu mô phỏng.
- **Vị trí:** Nằm trên **DUY NHẤT 1 hướng đường** có hướng $u \to v$ (`edge_id` từ $1..172$) tại khoảng cách $d$ tính từ nút $u$.
- **Mức độ nghiêm trọng & Cấu hình thời gian (`SEVERITY_CONFIG`):**
  - **Mức 1:** Thời gian thất bại = $60$ phút, Thời gian cần xử lý tại hiện trường = $4$ phút.
  - **Mức 2:** Thời gian thất bại = $40$ phút, Thời gian cần xử lý tại hiện trường = $6$ phút.
  - **Mức 3:** Thời gian thất bại = $30$ phút, Thời gian cần xử lý tại hiện trường = $8$ phút.
- **Các biến theo dõi tiến trình cuộc gọi:**
  - `created_time`: Thời điểm phát sinh cuộc gọi.
  - `failure_time`: Thời gian cho phép trước khi cuộc gọi thất bại ($60, 40, 30$ phút).
  - `processing_time`: Thời gian sơ cứu khi xe đến hiện trường ($4, 6, 8$ phút).
  - `assigned_ambulance_id`: Liên kết giữa cuộc gọi và xe cứu thương được phân công (hoặc `None` nếu chưa nhận).
  - `arrived_at_scene_time`: Thời điểm xe cứu thương tiếp cận hiện trường (được ghi nhận ngay khi đến).
- **Quy tắc tính Deadline & Hoàn hảo:**
  - Cuộc gọi thất bại khi thời gian vượt quá $T_{\text{deadline}} = T_{\text{created}} + T_{\text{failure}}$. **Xe đến đúng deadline ($t_{\text{arrival}} \le T_{\text{deadline}}$) được tính là THÀNH CÔNG**.
  - Nếu hoàn thành sơ cứu trước $T_{\text{created}} + 50\% \times T_{\text{failure}}$ được tính là **hoàn thành hoàn hảo (`is_perfect = True`)**.
  - Khi xe đến hiện trường (`AT_SCENE`), mốc thời gian thất bại bị loại bỏ.
- **Trạng thái cuộc gọi (`CallStatus`):**
  - `Yêu cầu xử lý (PENDING)`: Chưa có xe nào nhận. **Chỉ cuộc gọi đang `PENDING` mới được chọn để phân công**.
  - `Đợi xe cứu thương (WAITING_FOR_AMBULANCE)`: Đã có xe nhận và đang di chuyển tới.
  - `Đang xử lý (AT_SCENE)`: Xe đang ở hiện trường để sơ cứu.
  - `Hoàn thành (COMPLETED)`: Khi sơ cứu xong và xe bắt đầu xuất phát đến bệnh viện.
  - `Thất bại (FAILED)`: Quá hạn thời gian chờ hoặc bị từ chối do vượt quá giới hạn `max_pending_calls = 3` (ghi rõ lý do thất bại 1 lần duy nhất).

### 2.4. Bệnh Viện (`HospitalOnRoad` - 4 Bệnh Viện)
- Mỗi bệnh viện nằm trên **DUY NHẤT 1 cung đường có hướng**:
  - **BV 1 (Đa khoa Trung tâm Thành phố - Gần trung tâm):** Cung $19 \to 3$ (`edge_id`), vị trí +0.80 km. Thời gian bàn giao = $3.0$ phút.
  - **BV 2 (Cấp cứu Tim mạch Nội đô - Gần trung tâm):** Cung $4 \to 9$ (`edge_id`), vị trí +1.00 km. Thời gian bàn giao = $4.0$ phút.
  - **BV 3 (Đa khoa Khu vực Bắc Đô thị - Xa trung tâm):** Cung $21 \to 24$ (`edge_id`), vị trí +1.80 km. Thời gian bàn giao = $5.0$ phút.
  - **BV 4 (Cấp cứu Ngoại ô Tây Nam - Xa trung tâm):** Cung $62 \to 65$ (`edge_id`), vị trí +1.80 km. Thời gian bàn giao = $4.0$ phút.

---

## 3. Không Gian Quan Sát (`Observation`), Slot, Hành Động Đồng Thời & Action Mask

### 3.1. Cấu Trúc Đầy Đủ Của `Observation` (`env.get_observation()`)
Mỗi bước gọi `reset()` hoặc `step()` trả về dictionary `obs` bao gồm:

```python
obs = {
    # 1. Thông tin thời gian song song và episode
    "episode": 1,                      # Chỉ số episode (1, 2, 3...)
    "current_minute": 15,              # Phút mô phỏng hiện tại (0, 1, 2...)
    "traffic_time_step": 15,           # Bước thời gian giao thông chạy song song (0, 1, 2...)

    # 2. Tình trạng tắc đường trên mạng lưới 172 cung đường
    "edge_congestion_vector": ndarray, # Vector np.int32 shape (172,) chứa độ tắc {1, 2, 3} theo edge_id = 1..172
    "edge_congestion_by_id": {1: 1, 2: 2, ...}, # Dict ánh xạ edge_id -> mức tắc
    "edge_congestion": {(u, v): 1, ...},        # Dict ánh xạ (u, v) -> mức tắc
    "edges_by_id": {                   # Chi tiết từng cung đường theo edge_id (1..172)
        1: {
            "edge_id": 1, "id": 1, "edge": (u, v), "from_node": u, "to_node": v,
            "congestion": 1, "congestion_name": "Thông thoáng",
            "speed": 40.0, "length": 2.5, "travel_time_minutes": 3.75, "is_center": True
        }, ...
    },
    "edge_id_to_nodes": {1: (u, v), ...},
    "nodes_to_edge_id": {(u, v): 1, ...},

    # 3. Quản lý Slot cuộc gọi PENDING & Action Mask
    "call_slots": [slot_0, slot_1, slot_2], # 3 Slot chuẩn FIFO chứa dict cuộc gọi PENDING hoặc None
    "action_mask": ndarray,                 # Mảng boolean shape (229,) kiểm tra tính hợp lệ của action

    # 4. Trạng thái chi tiết từng Xe cứu thương (Ambulance)
    "ambulances": {
        1: {
            "id": 1, "ambulance_id": 1,
            "status": "Đang đến cuộc gọi",
            "current_edge": (u, v),
            "current_edge_id": 12,
            "distance_from_start": 1.2,
            "coord": (15.2, 14.8),
            "current_speed": 40.0,
            "current_call_id": 10,
            "target_type": "call",              # "call" | "hospital" | None
            "targer_type": "call",              # Alias tương thích
            "target_id": 10,
            "target_edge": (x, y),
            "target_edge_id": 45,
            "target_dist": 0.8,
            "target_name": "Cuộc gọi #10",
            "assigned_hospital_id": None,       # ID bệnh viện bàn giao (1..4) hoặc None
            "hospital_handover_time": 0.0,      # Thời gian bàn giao quy định tại BV (phút)
            "scene_time_remaining": 0.0,        # Thời gian sơ cứu tại hiện trường còn lại (phút)
            "hospital_time_remaining": 0.0,     # Thời gian bàn giao tại BV còn lại (phút)
            "route": [(u, v), ...],
            "route_edge_ids": [12, 18, 45],
            "total_distance_traveled": 15.6,
        }, ...
    },

    # 5. Danh sách các cuộc gọi theo trạng thái
    "active_calls": [                  # Danh sách dict các cuộc gọi đang xử lý / đang chờ
        {
            "id": 10, "call_id": 10,
            "severity": 2,
            "edge": (x, y),
            "edge_id": 45,
            "dist": 0.8,
            "coord": (18.0, 20.5),
            "status": "Đợi xe cứu thương",
            "created_time": 10.0,
            "failure_time": 40.0,
            "processing_time": 6.0,
            "assigned_ambulance_id": 1,
            "arrived_at_scene_time": None,      # Ghi nhận thời điểm đến hiện trường
            "wait_time": 5.0,
            "deadline": 50.0,
            "perfect_threshold": 30.0,
            "is_perfect": False,
        }, ...
    ],
    "pending_call_count": 1,
    "served_calls": [...],
    "failed_calls": [...],
    "perfect_calls": [...],
}
```

### 3.2. Quy Tắc Sắp Xếp Slot Thống Nhất (`get_call_slots()`)
- Chỉ xét các cuộc gọi đang ở trạng thái `PENDING`.
- Sắp xếp tăng dần theo `(created_time, call_id)` (FIFO) rồi đưa lần lượt vào `Slot 0`, `Slot 1`, `Slot 2`. Các slot chưa có cuộc gọi mang giá trị `None`.

### 3.3. Tập Hành Động Tiềm Năng, Action Mask & Giải Mã Hành Động
- Tại mỗi phút, Agent chọn một **tập các cặp phân công đồng thời** 1-đối-1 dưới dạng `[(1, 10), (3, 12)]` (Xe 1 nhận cuộc gọi ID 10, Xe 3 nhận cuộc gọi ID 12) hoặc truyền trực tiếp chỉ số `action_id`:
  $$\sum_{k=0}^{M} \binom{N}{k} \binom{M}{k} k! = \begin{cases} 1 + 18 + 90 + 120 = 229 & \text{khi } N = 6, M = 3 \\ 1 + 12 + 36 + 24 = 73 & \text{khi } N = 4, M = 3 \end{cases}$$
- **`env.get_action_mask()`**: Trả về mảng boolean đánh dấu các hành động hợp lệ (`mask[0] = True` luôn hợp lệ cho hành động không phân công `()`).
- **`env.decode_action(action)`**: Giải mã `action_id` (hoặc cặp slot) sang danh sách các cặp phân công đồng thời thực tế `[(amb_id, call_id), ...]`.
- **`env.encode_action([(amb_id, call_id), ...])`**: Mã hóa ngược từ tập cặp phân công sang `action_id`.

### 3.4. Đa Dạng Giữa Các Episode Khi Gọi `env.reset()`
- Khi huấn luyện qua nhiều episode liên tiếp bằng cách gọi `obs = env.reset()` (không truyền `seed`), bộ sinh số ngẫu nhiên của môi trường tự động tiến lên để tạo mới hoàn toàn:
  1. Phân bố độ tắc đường ban đầu trên toàn bộ $172$ cung đường (`city_map.reset_traffic()`),
  2. Vị trí xuất phát ngẫu nhiên (cung đường và khoảng cách) của tất cả các xe cứu thương,
  3. Chuỗi cuộc gọi cấp cứu phát sinh tại $t = 0$ và trong suốt episode.
- Nếu cần tái lập chính xác một kịch bản cố định để kiểm thử, truyền `env.reset(seed=42)`.

---

## 4. Công Thức Phần Thưởng (`REWARD_CONFIG` trong `dispatch_env/types.py`)

| Mức độ | Đợi (điểm trừ/ phút) ($w_c$) | Thất bại/ Quá hạn ($p_c$) | Mỗi km đi (điểm trừ/ km) | Hoàn thành ($b_c$) | Hoàn hảo ($b_c$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $1.0$ | $30$ | $0.05$ | $50$ | $70$ |
| **2** | $2.0$ | $60$ | $0.05$ | $65$ | $80$ |
| **3** | $4.0$ | $120$ | $0.05$ | $70$ | $100$ |

Tại mỗi bước thời gian $t$:

$$r_t = -\sum_c w_c \Delta W_{c,t} - \sum_{c \in F_t} p_c - 0.05 \sum_i \Delta d_{i,t} + \sum_{c \in C_t} b_c$$

Trong đó:
- $\Delta W_{c,t}$: Thời gian cuộc gọi $c$ thực sự chờ trong bước (phút).
- $w_c$: Trọng số phạt chờ theo mức độ ($1.0, 2.0, 4.0$).
- $F_t$: Tập cuộc gọi vừa thất bại hoặc bị từ chối trong bước. **Lưu ý: Tại $t = 0$, nếu cuộc gọi sinh ra vượt quá giới hạn 3 cuộc gọi PENDING và bị từ chối ngay lập tức, các cuộc gọi thất bại này vẫn bị trừ điểm phạt $p_c$ ngay tại $t = 0$ như thường** (ghi nhận trong `total_reward`, `last_step_reward` và file log Phút thứ 0).
- $p_c$: Mức phạt thất bại theo mức độ ($30, 60, 120$).
- $\Delta d_{i,t}$: Quãng đường xe $i$ di chuyển trong bước (km).
- $C_t$: Tập cuộc gọi vừa hoàn thành trong bước.
- $b_c$: Điểm thưởng hoàn thành thường hoặc hoàn hảo theo mức độ.

---

## 5. Hướng Dẫn Chạy Thử Nghiệm & Kiểm Thử

### 5.1. Chạy Mô Phỏng Toàn Diện (`run_simulation.py`)
Chạy với bộ tham số mặc định (`num_ambulances = 6`, `poisson_lambda = 0.1`, `max_time = 360`, `max_overtime = 300`, `max_pending_calls = 3`, `seed = 42`):
```powershell
& "C:\Users\ADMIN\AppData\Local\Python\bin\python.exe" run_simulation.py
```

Tùy biến các tham số trực tiếp từ dòng lệnh:
```powershell
# Chạy với 4 xe cứu thương và lambda = 0.2:
& "C:\Users\ADMIN\AppData\Local\Python\bin\python.exe" run_simulation.py --ambulances 4 --poisson 0.2

# Tùy chỉnh đầy đủ các tham số:
& "C:\Users\ADMIN\AppData\Local\Python\bin\python.exe" run_simulation.py --ambulances 6 --poisson 0.1 --max_time 360 --max_overtime 300 --max_pending 3 --seed 42 --log simulation.log
```

### 5.2. Chạy Toàn Bộ 34 Unit Tests Kiểm Thử
```powershell
& "C:\Users\ADMIN\AppData\Local\Python\bin\python.exe" -m unittest discover -s tests -v
```

### 5.3. Vẽ Bản Đồ Mạng Lưới 80 Nút & 4 Bệnh Viện
```powershell
& "C:\Users\ADMIN\AppData\Local\Python\bin\python.exe" visualize_map.py
```
