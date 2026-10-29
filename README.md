# Môi Trường Bản Đồ Đô Thị 80 Nút & Mô Phỏng Điều Phối Xe Cứu Thương (Ambulance Dispatch Simulation)

Hệ thống mô phỏng điều phối xe cứu thương thông minh trên nền tảng mạng lưới giao thông **80 NÚT GIAO THÔNG** (dạng cây phân nhánh kết hợp vùng lõi trung tâm dày đặc), **4 XE CỨU THƯƠNG**, **4 BỆNH VIỆN**, và cơ chế **sinh cuộc gọi cấp cứu Poisson** với quy trình ra quyết định theo tiến trình thời gian Markov (MDP).

---

## 1. Cấu Trúc Tổng Quan Thực Thể Hệ Thống

### 1.1. Xe cứu thương (4 xe)
- **Mã định danh:** Xe 1, Xe 2, Xe 3, Xe 4.
- **Vị trí hiện tại:** Vị trí ban đầu luôn được khởi tạo **ngẫu nhiên** trên các cung đường có hướng, với khoảng cách $d \in [0, \text{length}]$.
- **Tuyến đường hiện tại (current_route):** Danh sách các cung đường $(u \to v)$ mà xe sẽ lần lượt đi qua để đến đích.
- **Trạng thái xe:**
  - `Rảnh (IDLE)`: Hiện không xử lý cuộc gọi nào, sẵn sàng nhận nhiệm vụ mới. *(Lưu ý: Trạng thái ban đầu các xe luôn Rảnh)*.
  - `Đang đến cuộc gọi (EN_ROUTE_TO_CALL)`: Xe đang di chuyển tới vị trí nạn nhân.
  - `Xử lý tại cuộc gọi (AT_SCENE)`: Xe đã tới hiện trường và nhân viên y tế đang sơ cứu.
  - `Đang tới bệnh viện (EN_ROUTE_TO_HOSPITAL)`: Xe chở bệnh nhân di chuyển tới bệnh viện tối ưu.
  - `Xử lý tại bệnh viện (AT_HOSPITAL)`: Bàn giao bệnh nhân tại cổng cấp cứu bệnh viện.
- **Cuộc gọi đang xử lý:** Mã `call_id` (hoặc `None/NULL` tương ứng với xe đang rảnh).
- **Vị trí đích tới:** Vị trí hiện trường cuộc gọi hoặc bệnh viện tiếp nhận.
- **Cơ chế thu hồi nhiệm vụ:** Khi cuộc gọi bị xoá (thất bại/hết hạn) mà xe đang trên đường tiếp cận, xe sẽ **lập tức trở về trạng thái Rảnh** và sẵn sàng tìm cuộc gọi xử lý mới.

---

### 1.2. Cuộc gọi cấp cứu (Emergency Calls)
- **Mã cuộc gọi:** Số thứ tự tăng dần ($1, 2, 3, \dots$).
- **Vị trí:** Nằm trên **DUY NHẤT 1 hướng đường** có hướng $u \to v$ tại khoảng cách $d$ tính từ điểm đi $u$.
- **Mức độ nghiêm trọng:** $1, 2, 3$.
- **Thời gian thất bại & Thời gian xử lý:**
  - Mức độ nghiêm trọng càng cao thì thời gian cần xử lý càng tăng:
    - **Mức 1:** Thời gian thất bại = $30$ phút, Thời gian cần xử lý = $4$ phút.
    - **Mức 2:** Thời gian thất bại = $40$ phút, Thời gian cần xử lý = $6$ phút.
    - **Mức 3:** Thời gian thất bại = $60$ phút, Thời gian cần xử lý = $8$ phút.
  - Cuộc gọi thất bại khi tổng thời gian chờ vượt quá $T_{\text{failure}} + T_{\text{created}}$.
  - **Hoàn thành hoàn hảo:** Nếu hoàn thành trước $T_{\text{created}} + 50\% \times T_{\text{failure}}$ được tính là hoàn thành hoàn hảo.
- **Trạng thái cuộc gọi:**
  - `Yêu cầu xử lý (PENDING)`: Chưa có xe nào nhận cuộc gọi.
  - `Đợi xe cứu thương (WAITING_FOR_AMBULANCE)`: Đã có xe nhận và đang di chuyển tới.
  - `Đang xử lý (AT_SCENE)`: Xe đợi ở hiện trường để xử lý. **Tại đây loại bỏ hoàn toàn mốc thời gian thất bại**.
  - `Hoàn thành (COMPLETED)`: Khi sơ cứu xong và xe bắt đầu xuất phát đến bệnh viện.
  - `Thất bại (FAILED)`: Bị quá hạn thời gian chờ hoặc bị từ chối do vượt quá giới hạn hàng chờ.

---

### 1.3. Bệnh viện (4 Bệnh viện)
- Mỗi bệnh viện nằm trên **DUY NHẤT 1 cung đường có hướng** (không nằm trên chiều ngược lại):
  - **BV 1 (Đa khoa Trung tâm Thành phố - Gần trung tâm):** Nằm trên cung $19 \to 3$ (+0.80 km). Thời gian gửi bệnh nhân = $3.0$ phút.
  - **BV 2 (Cấp cứu Tim mạch Nội đô - Gần trung tâm):** Nằm trên cung $4 \to 9$ (+1.00 km). Thời gian gửi bệnh nhân = $4.0$ phút.
  - **BV 3 (Đa khoa Khu vực Bắc Đô thị - Xa trung tâm):** Nằm trên cung $21 \to 24$ (+1.80 km). Thời gian gửi bệnh nhân = $5.0$ phút.
  - **BV 4 (Cấp cứu Ngoại ô Tây Nam - Xa trung tâm):** Nằm trên cung $62 \to 65$ (+1.80 km). Thời gian gửi bệnh nhân = $4.0$ phút.
- **Thời gian gửi bệnh nhân:** Khác nhau giữa các bệnh viện với biên độ dao động hợp lý ($3.0 \to 5.0$ phút).

---

### 1.4. Tuyến đường, Vận tốc & Biến thiên Giao thông
- **3 trạng thái độ tắc:** `Thông thoáng (1)`, `Bình thường (2)`, `Tắc (3)`.
- **Phân vùng tốc độ & tỉ lệ chuyển đổi:**
  - **Vùng Trung tâm (Lõi nội đô dày đặc N1..N20):**
    - Vận tốc cho phép thấp: Thông thoáng = $40\text{ km/h}$, Bình thường = $25\text{ km/h}$, Tắc = $12\text{ km/h}$.
    - Tỉ lệ chuyển đổi sang trạng thái tắc tăng cao (xác suất tắc lên tới $60\% - 70\%$).
  - **Vùng Ngoại ô (Vành đai & Hành lang phân nhánh):**
    - Vận tốc cho phép cao: Thông thoáng = $60\text{ km/h}$, Bình thường = $40\text{ km/h}$, Tắc = $20\text{ km/h}$.
    - Tỉ lệ chuyển đổi trạng thái tắc thấp hơn, khả năng thông thoáng cao hơn.

---

## 2. Quy Trình Di Chuyển Liên Tục Trong 1 Phút $[t, t+1)$

Nếu xe chuyển đổi trạng thái, đến đích hoặc hoàn thành công việc khi chưa hết 1 phút:
1. **Tính thời gian đi đến cuối đoạn:** $t_{\text{leg}} = d_{\text{rem}} / v_{\text{minute}}$.
2. **Chuyển sang đoạn tiếp theo** trong tuyến đường đã chọn bằng Dijkstra: rẽ sang đường mới hoặc tới cuối đường quay đầu (U-turn).
3. **Dùng thời gian còn lại** $\Delta t = 1.0 - t_{\text{leg}}$ để tiếp tục di chuyển trên cung mới.
4. **Xử lý sự kiện tại hiện trường / bệnh viện giữa phút:**
   - Đến hiện trường: chuyển sang `AT_SCENE`, dùng thời gian còn lại để sơ cứu.
   - Sơ cứu xong: hoàn thành cuộc gọi, chuyển sang `EN_ROUTE_TO_HOSPITAL`, dùng thời gian còn lại đi tới bệnh viện.
   - Đến bệnh viện: chuyển sang `AT_HOSPITAL`, dùng thời gian còn lại để bàn giao.
   - Bàn giao xong: chuyển về `IDLE`.

---

## 3. Quy Trình Mô Phỏng Từng Phút (MDP Step)

Tại đầu phút $t_0$:
1. **Quan sát trạng thái hiện tại $S_t$:** Vị trí xe, cuộc gọi đang chờ, tình trạng giao thông.
2. **Xác định các hành động hợp lệ:** Các cặp $A = (\text{Xe rảnh}, \text{Cuộc gọi PENDING})$.
3. **Tác nhân chọn hành động $A_t$:** Phân công xe rảnh cho cuộc gọi.
4. **Thực hiện phân công & lập tuyến đường:** Dijkstra tìm lộ trình nhanh nhất tới hiện trường.
5. **Trong khoảng $[t, t+1)$:**
   - Các xe di chuyển liên tục theo quy tắc sub-minute.
   - Xử lý các sự kiện đến hiện trường, sơ cứu, đưa đi bệnh viện, bàn giao.
   - Cập nhật thời gian chờ và kiểm tra mốc thất bại.
6. **Tại $t+1$:**
   - Cập nhật ma trận Markov trạng thái giao thông.
   - Tính toán lại tuyến đường (reroute) nếu lộ trình bị ảnh hưởng.
   - Sinh cuộc gọi mới: Poisson ($\lambda = 0.5$). Nếu $t \ge 360$, ngừng sinh cuộc gọi.
   - Kiểm tra giới hạn: **Tối đa 3 cuộc gọi trong trạng thái PENDING**. Các cuộc gọi vượt quá bị từ chối ngay lập tức (FAILED).
7. **Tính phần thưởng $R_{t+1}$:**
   - Phạt thời gian chờ: $-1.0 \times \text{mức độ} / \text{phút}$.
   - Phạt cuộc gọi thất bại / từ chối: $-50.0 \times \text{mức độ}$.
   - Thưởng cuộc gọi hoàn thành: $+100.0 \times \text{mức độ}$ (+ thưởng thêm $+50.0 \times \text{mức độ}$ nếu hoàn thành hoàn hảo).
8. **Trả về trạng thái mới $S_{t+1}$.**

---

## 4. Cấu Trúc Thư Mục Dự Án

```text
ambulance_dispatch/
├── map_env/                    # Package bản đồ đô thị 80 nút
│   ├── __init__.py
│   ├── node.py                 # Nút giao thông & phân loại giao lộ (Ngã 4, 3, Rẽ, Cụt)
│   ├── edge.py                 # Cung đường có hướng, vận tốc theo vùng, Markov transition
│   ├── hospital.py             # Thực thể bệnh viện trên 1 cung đường có hướng
│   └── city_map.py             # Bản đồ 80 nút, 172 cung có hướng, định tuyến Dijkstra
├── dispatch_env/               # Package mô phỏng điều phối xe cứu thương
│   ├── __init__.py
│   ├── types.py                # Enum AmbulanceStatus, CallStatus, SEVERITY_CONFIG
│   ├── call.py                 # Thực thể EmergencyCall, tính hạn thất bại & hoàn thành hoàn hảo
│   ├── ambulance.py            # Thực thể Ambulance, di chuyển sub-minute, reroute Dijkstra
│   └── environment.py          # AmbulanceDispatchEnv (MDP step, Poisson, limit 3 pending, log)
├── tests/
│   ├── test_map.py             # 8 unit tests kiểm tra bản đồ 80 nút
│   └── test_dispatch_env.py    # 10 unit tests kiểm tra toàn diện môi trường điều phối
├── run_simulation.py           # Script chạy thử nghiệm toàn diện 360 phút và ghi file log
├── simulation.log              # File log chi tiết từng phút theo chuẩn format người dùng
├── visualize_map.py            # Script vẽ sơ đồ trực quan mạng lưới
├── city_map_80nodes.png        # Ảnh sơ đồ mạng lưới đô thị 80 nút & 4 bệnh viện
├── requirements.txt            # Thư viện phụ thuộc (numpy, matplotlib)
└── README.md                   # Tài liệu hướng dẫn chi tiết
```

---

## 5. Định Dạng File Log Chuẩn Đầu Ra (`simulation.log`)

Mỗi bước thời gian phút $t$ được ghi theo mẫu chuẩn:

```text
Phút thứ t:
Số cuộc gọi sinh ra: …
Thông tin cuộc gọi …: 
—--------------------------------------------
Xe 1 đang di chuyển trên tuyến đường :... Hiện đang ở vị trí …. Vận tốc: … đang xử lý cuộc gọi: ….. Trạng thái: ………….
Xe N đang di chuyển trên tuyến đường :... Hiện đang ở vị trí …. Vận tốc: … đang xử lý cuộc gọi: ….. Trạng thái: ………….
—--------------------------------------------
Cách tuyến đường chuyển trạng thái:
Từ tắc -> bình thường: ………..
Từ thông thoáng -> bình thường: ………..
Từ bình thường -> tắc: ………..
Từ bình thường -> thông thoáng: ………..
Từ tắc -> thông thoáng: ………..
Từ thông thoáng -> tắc: ………..
—--------------------------------------------
Số cuộc gọi thành công: 
Số cuộc gọi thất bại:
Lý do cuộc gọi thất bại trong phút này:
• Cuộc gọi #X (Mức ...): [Lý do chi tiết khi thất bại - chỉ ghi 1 lần duy nhất]
—---------------------------------------------
Xe … đã tính toán lại đoạn đường
```

---

## 6. Lệnh Chạy Thử Nghiệm

### 6.1. Chạy Mô Phỏng Toàn Diện & Tùy Biến Tham Số Poisson
Mặc định ($\lambda = 0.5$):
```powershell
& "C:\Users\ADMIN\AppData\Local\Python\bin\python.exe" run_simulation.py
```

Tùy biến tham số Poisson trực tiếp từ dòng lệnh:
```powershell
# Chạy với ít cuộc gọi hơn (lambda = 0.2 cuộc gọi/phút):
& "C:\Users\ADMIN\AppData\Local\Python\bin\python.exe" run_simulation.py --poisson 0.2

# Chạy với mật độ cuộc gọi cao (lambda = 1.0 cuộc gọi/phút):
& "C:\Users\ADMIN\AppData\Local\Python\bin\python.exe" run_simulation.py --poisson 1.0

# Chạy với thời gian và giới hạn hàng chờ tùy ý:
& "C:\Users\ADMIN\AppData\Local\Python\bin\python.exe" run_simulation.py --poisson 0.8 --max_pending 5 --max_time 180 --log test.log
```
*Mô phỏng với tiến trình Poisson, 4 xe cứu thương, 4 bệnh viện, xử lý các cuộc gọi đến khi hoàn tất và ghi ra [`simulation.log`](simulation.log).*

### 6.2. Chạy Toàn Bộ 18 Unit Tests Kiểm Thử
```powershell
& "C:\Users\ADMIN\AppData\Local\Python\bin\python.exe" -m unittest discover -s tests -v
```
*Tất cả 18 unit tests (8 tests bản đồ + 10 tests điều phối và quy tắc nghiệp vụ) đều vượt qua thành công.*

### 6.3. Vẽ Bản Đồ Mạng Lưới
```powershell
& "C:\Users\ADMIN\AppData\Local\Python\bin\python.exe" visualize_map.py
```
