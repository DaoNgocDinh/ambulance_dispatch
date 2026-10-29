# Môi Trường Bản Đồ Đô Thị 80 Nút & Mô Phỏng Điều Phối Xe Cứu Thương (Ambulance Dispatch Simulation)

Hệ thống mô phỏng điều phối xe cứu thương thông minh trên nền tảng mạng lưới giao thông **80 NÚT GIAO THÔNG** (88 tuyến đường 2 chiều = **176 cung đường có hướng**), **6 XE CỨU THƯƠNG** (mặc định), **4 BỆNH VIỆN**, và cơ chế **sinh cuộc gọi cấp cứu Poisson** ($\lambda = 0.1$) với quy trình ra quyết định theo tiến trình thời gian Markov (MDP).

---

## 1. Bảng Tổng Hợp Các Biến & Tham Số Chuẩn (Đồng Bộ Với `run_simulation.py`)

Toàn bộ các file trong dự án ([`run_simulation.py`](run_simulation.py), [`dispatch_env/types.py`](dispatch_env/types.py), [`dispatch_env/environment.py`](dispatch_env/environment.py)) sử dụng chung bộ tham số mặc định:

| Tên biến / Hằng số | Giá trị mặc định | Ý nghĩa |
| :--- | :---: | :--- |
| `num_ambulances` (`DEFAULT_NUM_AMBULANCES`) | `6` | Số lượng xe cứu thương hoạt động trong môi trường |
| `max_time` (`DEFAULT_MAX_TIME`) | `360` phút | Thời gian mô phỏng sinh cuộc gọi tối đa ($6.0$ giờ) |
| `max_overtime` (`DEFAULT_MAX_OVERTIME`) | `300` phút | Thời gian quá giờ tối đa sau `max_time` trước khi dừng và báo `Truncated` (tổng tối đa $660$ phút) |
| `poisson_lambda` (`DEFAULT_POISSON_LAMBDA`) | `0.1` | Tốc độ sinh cuộc gọi trung bình mỗi phút theo phân phối Poisson |
| `max_pending_calls` (`DEFAULT_MAX_PENDING_CALLS`) | `3` | Số cuộc gọi tối đa được phép nằm trong hàng chờ `PENDING` (tương ứng $3$ Slot cố định `Slot 0..2`) |
| `seed` (`DEFAULT_SEED`) | `42` | Hạt giống sinh số ngẫu nhiên khởi tạo môi trường |
| `num_nodes` | `80` | Tổng số nút giao thông trên bản đồ (`1..80`, gồm nút trung tâm `1..20` và ngoại ô `21..80`) |
| `num_directed_edges` | `176` | Tổng số cung đường có hướng $(u \to v)$ với độ tắc độc lập $\{1, 2, 3\}$ |
| `num_hospitals` | `4` | Số bệnh viện ($2$ gần trung tâm: BV 1, BV 2; $2$ xa trung tâm: BV 3, BV 4) |
| `num_potential_actions` | `229` (với $6$ xe, $3$ slot)<br>`73` (với $4$ xe, $3$ slot) | Tổng số cách ghép hành động đồng thời tiềm năng (bao gồm cả hành động không phân công `()`) |

---

## 2. Cấu Trúc Tổng Quan Thực Thể Hệ Thống

### 2.1. Xe cứu thương (`Ambulance` - Mặc định 6 xe)
- **Mã định danh:** `Xe 1 .. Xe 6` (tùy biến qua `num_ambulances`).
- **Vị trí hiện tại:** Vị trí ban đầu luôn được khởi tạo **ngẫu nhiên** trên các cung đường có hướng riêng biệt, với khoảng cách $d \in [0, \text{length}]$.
- **Tuyến đường hiện tại (`route_edges`):** Danh sách các cung đường $(u \to v)$ mà xe sẽ lần lượt đi qua để đến đích.
- **Trạng thái xe (`AmbulanceStatus`):**
  - `Rảnh (IDLE)`: Hiện không xử lý cuộc gọi nào, sẵn sàng nhận nhiệm vụ mới. *(Trạng thái ban đầu các xe luôn Rảnh)*.
  - `Đang đến cuộc gọi (EN_ROUTE_TO_CALL)`: Xe đang di chuyển tới vị trí nạn nhân.
  - `Xử lý tại cuộc gọi (AT_SCENE)`: Xe đã tới hiện trường và nhân viên y tế đang sơ cứu.
  - `Đang tới bệnh viện (EN_ROUTE_TO_HOSPITAL)`: Xe chở bệnh nhân di chuyển tới bệnh viện tối ưu.
  - `Xử lý tại bệnh viện (AT_HOSPITAL)`: Bàn giao bệnh nhân tại cổng cấp cứu bệnh viện.
- **Cơ chế thu hồi nhiệm vụ:** Khi cuộc gọi bị xoá (thất bại/quá hạn) trong lúc xe đang trên đường tiếp cận, xe sẽ **lập tức trở về trạng thái Rảnh (`IDLE`)** và sẵn sàng nhận cuộc gọi mới.

### 2.2. Cuộc gọi cấp cứu (`EmergencyCall`)
- **Thời điểm xuất hiện:** Cuộc gọi xuất hiện vào **đầu mỗi phút**. Tại $t = 0$ có thể sinh cuộc gọi ngay từ đầu mô phỏng.
- **Vị trí:** Nằm trên **DUY NHẤT 1 hướng đường** có hướng $u \to v$ tại khoảng cách $d$ tính từ điểm đi $u$.
- **Mức độ nghiêm trọng & Cấu hình thời gian (`SEVERITY_CONFIG`):**
  - **Mức 1:** Thời gian thất bại = $60$ phút, Thời gian cần xử lý tại hiện trường = $4$ phút.
  - **Mức 2:** Thời gian thất bại = $40$ phút, Thời gian cần xử lý tại hiện trường = $6$ phút.
  - **Mức 3:** Thời gian thất bại = $30$ phút, Thời gian cần xử lý tại hiện trường = $8$ phút.
- **Quy tắc tính Deadline & Hoàn hảo:**
  - Cuộc gọi thất bại khi thời gian vượt quá $T_{\text{deadline}} = T_{\text{created}} + T_{\text{failure}}$. **Xe đến đúng deadline ($t_{\text{arrival}} \le T_{\text{deadline}}$) được tính là THÀNH CÔNG**.
  - Nếu hoàn thành sơ cứu trước $T_{\text{created}} + 50\% \times T_{\text{failure}}$ được tính là **hoàn thành hoàn hảo (`is_perfect = True`)**.
- **Trạng thái cuộc gọi (`CallStatus`):**
  - `Yêu cầu xử lý (PENDING)`: Chưa có xe nào nhận. **Chỉ cuộc gọi đang `PENDING` mới được chọn để phân công**.
  - `Đợi xe cứu thương (WAITING_FOR_AMBULANCE)`: Đã có xe nhận và đang di chuyển tới.
  - `Đang xử lý (AT_SCENE)`: Xe đang ở hiện trường để sơ cứu. **Tại đây loại bỏ hoàn toàn mốc thời gian thất bại**.
  - `Hoàn thành (COMPLETED)`: Khi sơ cứu xong và xe bắt đầu xuất phát đến bệnh viện.
  - `Thất bại (FAILED)`: Quá hạn thời gian chờ hoặc bị từ chối do vượt quá giới hạn `max_pending_calls = 3` (ghi rõ lý do thất bại 1 lần duy nhất).

### 2.3. Bệnh viện (`HospitalOnRoad` - 4 Bệnh viện)
- Mỗi bệnh viện nằm trên **DUY NHẤT 1 cung đường có hướng**:
  - **BV 1 (Đa khoa Trung tâm Thành phố - Gần trung tâm):** Cung $19 \to 3$ (+0.80 km). Thời gian bàn giao = $3.0$ phút.
  - **BV 2 (Cấp cứu Tim mạch Nội đô - Gần trung tâm):** Cung $4 \to 9$ (+1.00 km). Thời gian bàn giao = $4.0$ phút.
  - **BV 3 (Đa khoa Khu vực Bắc Đô thị - Xa trung tâm):** Cung $21 \to 24$ (+1.80 km). Thời gian bàn giao = $5.0$ phút.
  - **BV 4 (Cấp cứu Ngoại ô Tây Nam - Xa trung tâm):** Cung $62 \to 65$ (+1.80 km). Thời gian bàn giao = $4.0$ phút.

### 2.4. Tuyến đường, Vận tốc & Biến thiên Giao thông (`DirectedEdge`)
- **3 trạng thái độ tắc:** `Thông thoáng (1)`, `Bình thường (2)`, `Tắc (3)`.
- **Vùng Trung tâm (`N1..N20`):** Thông thoáng = $40\text{ km/h}$, Bình thường = $25\text{ km/h}$, Tắc = $12\text{ km/h}$ (xác suất chuyển sang tắc cao).
- **Vùng Ngoại ô (`N21..N80`):** Thông thoáng = $60\text{ km/h}$, Bình thường = $40\text{ km/h}$, Tắc = $20\text{ km/h}$ (xác suất thông thoáng cao).

---

## 3. Không Gian Quan Sát (`Observation`), Hành Động Đồng Thời (`Joint Action`), Slot & Action Mask

### 3.1. Cấu Trúc `Observation` (`env.get_observation()`)
Hàm `reset()` và `step()` trả về dictionary `obs` với đầy đủ thông tin thời gian song song và tình trạng tắc đường trên từng cung đường:
- `obs["episode"]`: Chỉ số episode hiện tại ($1, 2, 3, \dots$).
- `obs["current_minute"]`: Phút mô phỏng hiện tại ($0, 1, 2, \dots$).
- `obs["traffic_time_step"]`: Bước thời gian giao thông chạy **song song** với `current_minute` ($0, 1, 2, \dots$).
- `obs["edge_congestion"]`: `Dict[(u, v), int]` ánh xạ từng cung đường có hướng sang mức độ tắc $\{1, 2, 3\}$.
- `obs["edge_congestion_vector"]`: `np.ndarray` kích thước `(176,)` (`dtype=int32`) chứa độ tắc của toàn bộ $176$ cung đường theo thứ tự chuẩn `env.edge_keys`.
- `obs["edges"]`: `Dict[(u, v), Dict]` chứa chi tiết từng cung đường (`congestion`, `congestion_name`, `speed`, `length`, `travel_time_minutes`, `is_center`).
- `obs["ambulances"]`: Trạng thái chi tiết của từng xe cứu thương.
- `obs["call_slots"]`: Danh sách $3$ phần tử tương ứng `Slot 0`, `Slot 1`, `Slot 2` (chứa dict thông tin cuộc gọi `PENDING` hoặc `None` nếu trống).
- `obs["action_mask"]`: `np.ndarray` boolean kiểm tra tính hợp lệ của từng cách ghép hành động tiềm năng.

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
  1. Phân bố độ tắc đường ban đầu trên toàn bộ $176$ cung đường (`city_map.reset_traffic()`),
  2. Vị trí xuất phát ngẫu nhiên (cung đường và khoảng cách) của tất cả các xe cứu thương,
  3. Chuỗi cuộc gọi cấp cứu phát sinh tại $t = 0$ và trong suốt episode.
- Nếu cần tái lập chính xác một kịch bản cố định để kiểm thử, truyền `env.reset(seed=42)`.

---

## 4. Công Thức Phần Thưởng (`REWARD_CONFIG` trong `dispatch_env/types.py`)

| Mức độ | Đợi (điểm trừ/ phút) ($w_c$) | Thất bại/ Quá hạn ($p_c$) | Mỗi km đi (điểm trừ/ km) | Hoàn thành ($b_c$) | Hoàn hảo ($b_c$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $0.5$ | $30$ | $0.05$ | $50$ | $70$ |
| **2** | $1.0$ | $60$ | $0.05$ | $65$ | $80$ |
| **3** | $1.5$ | $120$ | $0.05$ | $70$ | $100$ |

Tại mỗi bước thời gian $t$:

$$r_t = -\sum_c w_c \Delta W_{c,t} - \sum_{c \in F_t} p_c - 0.05 \sum_i \Delta d_{i,t} + \sum_{c \in C_t} b_c$$

---

## 5. Lệnh Chạy Thử Nghiệm & Kiểm Thử

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

### 5.2. Chạy Toàn Bộ 31 Unit Tests Kiểm Thử
```powershell
& "C:\Users\ADMIN\AppData\Local\Python\bin\python.exe" -m unittest discover -s tests -v
```

### 5.3. Vẽ Bản Đồ Mạng Lưới 80 Nút & 4 Bệnh Viện
```powershell
& "C:\Users\ADMIN\AppData\Local\Python\bin\python.exe" visualize_map.py
```
