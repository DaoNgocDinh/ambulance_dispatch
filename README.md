# Môi Trường Mô Phỏng Điều Phối Xe Cứu Thương Khẩn Cấp Trong Thành Phố Thông Minh (MDP)

Môi trường thử nghiệm Học tăng cường (Reinforcement Learning) cho đề tài **"Ứng dụng Học tăng cường trong Điều phối Xe cứu thương Khẩn cấp trong Thành phố Thông minh"**, được thiết kế bám sát mô hình Quy trình Quyết định Markov (**MDP**) tại **Thẻ 1** của báo cáo với bộ tham số:

- **Số nút giao thông ($|V|$):** `15` nút (đánh số từ `1` đến `15`)
- **Số xe cứu thương ($K$):** `3` xe cứu thương
- **Số cuộc gọi cấp cứu chờ đồng thời tối đa ($M$):** `3` cuộc gọi đồng thời
- **Số tuyến đường huyết mạch ($|E_{\text{arterial}}|$):** `5` tuyến đường huyết mạch biến thiên giao thông theo thời gian thực

---

## 1. Cấu Trúc Thư Mục

```text
ambulance_dispatch/
├── environment/                # Folder chứa toàn bộ mã nguồn môi trường MDP
│   ├── __init__.py             # Export các lớp và hằng số của môi trường
│   ├── config.py               # Cấu hình tham số mạng lưới, hàm thưởng R(s,a), Poisson, gamma
│   ├── city_graph.py           # Đồ thị 15 nút, 3 trạm trực, 2 bệnh viện, 5 tuyến đường huyết mạch
│   ├── entities.py             # Thực thể Xe cứu thương (Ambulance) và Cuộc gọi cấp cứu (EmergencyRequest)
│   └── ambulance_env.py        # Lớp môi trường chính SmartCityAmbulanceEnv (reset, step, render)
├── tests/
│   └── test_env.py             # Bộ kiểm thử tự động (Unit Tests) cho các thành phần S, A, P, R
├── demo_env.py                 # Script chạy minh họa từng bước và so sánh Random vs. Baseline
├── requirements.txt            # Các thư viện Python cần thiết
└── README.md                   # Tài liệu mô tả chi tiết về môi trường thử nghiệm
```

---

## 2. Thiết Kế Mạng Lưới Giao Thông Đô Thị (15 Nút & 5 Tuyến Huyết Mạch)

Mạng lưới đô thị được mô hình hóa dưới dạng đồ thị vô hướng có trọng số $G = (V, E)$ gồm $|V| = 15$ nút bố trí theo cấu trúc 3 phân khu (Bắc – Trung tâm – Nam) kết nối bởi các đường nội đô và **5 tuyến đường huyết mạch** chính:

```text
[1:Trạm TB] ---- [2:KDC Bắc] =====(HM1)===== [3:BV ĐK BẮC] ---- [4:KĐT Bắc] ---- [5:KCNC ĐB]
     |                | \                        || (HM2)        /   |                |
[6:KCN Tây] ---- [7:TTTM Tây] ====(HM3)==== [8:TRẠM TRUNG TÂM] =(HM4)= [9:TTTM Đông] - [10:KDC Đông]
     |                |   \                      || (HM5)      /     |                |
[11:KDC TN] ---- [12:Ngã Tư Nam] ---------- [13:BV ĐK NAM] ---- [14:KDC ĐN] ---- [15:Trạm ĐN]
```

### 2.1. Phân loại 15 nút giao thông ($V = \{1, \dots, 15\}$)
| Nhóm nút | Các nút | Vai trò trong mô phỏng |
| :--- | :--- | :--- |
| **Trạm trực ($3$ trạm)** | **Nút 1**, **Nút 8**, **Nút 15** | Nơi xuất phát ban đầu của 3 xe cứu thương và là đích đến của hành động tái định vị $a_{\text{relocate}}(k, j)$ nhằm duy trì độ phủ an toàn toàn thành phố. |
| **Bệnh viện ($2$ bệnh viện)** | **Nút 3**, **Nút 13** | Bệnh viện Đa khoa Bắc (Nút 3) và Bệnh viện Đa khoa Nam (Nút 13) – nơi xe cứu thương đưa bệnh nhân đến để hoàn tất ca cấp cứu. |
| **Khu thương mại & dân cư** | Các nút còn lại (`2, 4, 5, 6, 7, 9, 10, 11, 12, 14`) | Có trọng số mật độ dân cư $w_v$ khác nhau (cao nhất tại các trung tâm thương mại Nút 7, 8, 9), quyết định cường độ phát sinh cuộc gọi cấp cứu $\lambda(v, t)$. |

### 2.2. Danh sách 5 tuyến đường huyết mạch ($E_{\text{arterial}}$)
Trạng thái giao thông $c_e \in \{0, 1, 2\}$ biến thiên động trên 5 trục đường chính dẫn vào trung tâm và 2 bệnh viện:
1. **Tuyến huyết mạch 1 (`c_1`):** Cạnh `(2 <-> 3)` – Trục Khu Dân Cư Bắc $\leftrightarrow$ Bệnh Viện Đa Khoa Bắc.
2. **Tuyến huyết mạch 2 (`c_2`):** Cạnh `(3 <-> 8)` – Đại lộ Bệnh Viện Đa Khoa Bắc $\leftrightarrow$ Trạm Trung Tâm Thành Phố.
3. **Tuyến huyết mạch 3 (`c_3`):** Cạnh `(7 <-> 8)` – Trục Trung Tâm Thương Mại Tây $\leftrightarrow$ Trạm Trung Tâm Thành Phố.
4. **Tuyến huyết mạch 4 (`c_4`):** Cạnh `(8 <-> 9)` – Trục Trạm Trung Tâm Thành Phố $\leftrightarrow$ Trung Tâm Thương Mại Đông.
5. **Tuyến huyết mạch 5 (`c_5`):** Cạnh `(8 <-> 13)` – Đại lộ Trạm Trung Tâm Thành Phố $\leftrightarrow$ Bệnh Viện Đa Khoa Nam.

*(Ngoài ra đồ thị có các đường nhánh vành đai `(2-8)`, `(4-8)`, `(7-13)`, `(9-13)` để Agent có thể thực hiện hành động định tuyến tránh kẹt xe $a_{\text{route}}(k, u)$ khi các tuyến huyết mạch bị ùn tắc).*

---

## 3. Mô Hình Hóa Quy Trình Quyết Định Markov $\mathcal{M} = \langle S, A, P, R, \gamma \rangle$

### 3.1. Không gian Trạng thái ($S$)
Trạng thái tại bước thời gian $t$ là vector 20 chiều thu thập toàn cảnh thông tin đô thị:
$$S_t = [X_{\text{amb}}, X_{\text{req}}, X_{\text{traffic}}] \in \mathbb{Z}^{20}$$

1. **Trạng thái đội xe cứu thương $X_{\text{amb}} = [p_1, st_1, p_2, st_2, p_3, st_3]$ ($6$ chiều):**
   - $p_k \in \{1, \dots, 15\}$: Vị trí nút hiện tại của xe cứu thương $k \in \{1, 2, 3\}$.
   - $st_k \in \{0, 1, 2, 3, 4\}$: Trạng thái hoạt động của xe $k$:
     - `0` (`IDLE`): Đang rảnh / chờ lệnh tại trạm hoặc nút hiện tại.
     - `1` (`TO_SCENE`): Đang di chuyển đi đón bệnh nhân tại hiện trường.
     - `2` (`ON_SCENE`): Đang sơ cứu và tiếp nhận bệnh nhân tại hiện trường.
     - `3` (`TO_HOSPITAL`): Đang vận chuyển bệnh nhân tới bệnh viện gần nhất.
     - `4` (`RELOCATING`): Đang di chuyển tái định vị về trạm trực dự phòng.

2. **Danh sách cuộc gọi cấp cứu đang chờ $X_{\text{req}} = [\text{req}_1, \text{req}_2, \text{req}_3]$ ($9$ chiều):**
   - Mỗi slot $m \in \{1, 2, 3\}$ gồm bộ 3 giá trị $[\text{loc}_m, \text{prio}_m, \text{wait\_time}_m]$ (nếu slot trống thì bằng `[0, 0, 0]`):
     - $\text{loc}_m \in \{1, \dots, 15\}$: Vị trí nút phát sinh cuộc gọi cấp cứu.
     - $\text{prio}_m \in \{1, 2, 3\}$: Mức độ ưu tiên y tế:
       - `1` (**Đỏ – Nguy kịch**): $T_{\text{golden}} = 4$ bước, $T_{\max} = 7$ bước.
       - `2` (**Vàng – Khẩn cấp**): $T_{\text{golden}} = 6$ bước, $T_{\max} = 10$ bước.
       - `3` (**Xanh – Tiêu chuẩn**): $T_{\text{golden}} = 8$ bước, $T_{\max} = 14$ bước.
     - $\text{wait\_time}_m \ge 0$: Số bước thời gian bệnh nhân đã phải chờ.

3. **Trạng thái giao thông $X_{\text{traffic}} = [c_1, c_2, c_3, c_4, c_5]$ ($5$ chiều):**
   - Mức độ lưu thông của 5 tuyến đường huyết mạch:
     - `0`: Thông thoáng (`TRAFFIC_CLEAR`)
     - `1`: Đông (`TRAFFIC_BUSY`)
     - `2`: Ùn tắc (`TRAFFIC_CONGESTED`)

> **Phân tích số lượng trạng thái lý thuyết:**
> Với mô hình **15 nút, 3 xe cứu thương, tối đa 3 cuộc gọi đồng thời và 5 tuyến đường huyết mạch**:
> - Số tổ hợp của $X_{\text{amb}}$: $(15 \times 5)^3 = 75^3 = 421,875$
> - Số tổ hợp của $X_{\text{req}}$: $(1 + 15 \times 3 \times 15)^3 = 676^3 = 308,915,776$
> - Số tổ hợp của $X_{\text{traffic}}$: $3^5 = 243$
> - **Tổng số trạng thái lý thuyết:** $421,875 \times 308,915,776 \times 243 \approx \mathbf{3.17 \times 10^{16}}$ trạng thái.
>
> Để hỗ trợ đồng thời cả 4 thuật toán trong báo cáo (DP, Monte Carlo, Q-Learning và PPO), môi trường cung cấp 3 phương thức trích xuất trạng thái:
> - `env.get_state_vector()`: Vector nguyên 20 chiều $S_t = [X_{\text{amb}}, X_{\text{req}}, X_{\text{traffic}}]$.
> - `env.get_normalized_vector()`: Vector thực chuẩn hóa $[0, 1]$ (20 chiều) làm đầu vào cho mạng thần kinh của **PPO**.
> - `env.get_tabular_state()`: Tuple trạng thái rời rạc rút gọn ($1,536$ trạng thái) giúp các thuật toán bảng (**Dynamic Programming**, **Monte Carlo**, **Q-Learning**) hội tụ nhanh chóng.

---

### 3.2. Không gian Hành động ($A$)
Tập hành động rời rạc gồm **25 hành động** (`0..24`), chia thành 4 nhóm đúng theo mục 4.2.2:

| Mã hành động | Ký hiệu MDP | Ý nghĩa điều phối |
| :--- | :--- | :--- |
| `0 .. 8` ($3 \times 3$) | $a_{\text{dispatch}}(k, m)$ | Phân bổ xe rảnh $k \in \{0, 1, 2\}$ xuất phát phục vụ cuộc gọi cấp cứu ở slot $m \in \{0, 1, 2\}$. |
| `9 .. 17` ($3 \times 3$) | $a_{\text{relocate}}(k, j)$ | Tái định vị xe rảnh $k \in \{0, 1, 2\}$ chuyển sang trực dự phòng tại trạm $j \in \{0, 1, 2\}$ (Nút 1, 8, 15). |
| `18 .. 23` ($3 \times 2$) | $a_{\text{route}}(k, u)$ | Chỉ định xe đang di chuyển $k \in \{0, 1, 2\}$ chọn hướng rẽ tránh kẹt xe tại nút giao ($u=0$: nhánh có thời gian thực nhanh nhất; $u=1$: nhánh thay thế thứ 2). |
| `24` ($1$) | $a_{\text{standby}}$ | Giữ nguyên trạng thái (chờ tại trạm hoặc để các xe tiếp tục lộ trình hiện tại). |

Môi trường cung cấp hàm `env.get_action_mask()` và `env.get_valid_actions()` trả về mặt nạ các hành động hợp lệ tại mỗi bước thời gian.

---

### 3.3. Hàm Phần thưởng Đa mục tiêu ($R(s, a)$)
Hàm phần thưởng phản hồi trong `step(action)` tuân thủ chính xác các trọng số tại mục 4.2.3:

| Sự kiện / Điều kiện | Giá trị thưởng/phạt | Ý nghĩa |
| :--- | :---: | :--- |
| **Tiếp cận trong Thời gian vàng ($T_{\text{resp}} \le T_{\text{golden}}$)** | **`+50`** | Thưởng cao nhất khi xe cứu thương tiếp cận bệnh nhân kịp thời trong thời gian vàng. |
| **Chuyển viện an toàn thành công** | **`+20`** | Thưởng khi xe đưa bệnh nhân về tới bệnh viện (Nút 3 hoặc Nút 13) và hoàn thành ca cấp cứu. |
| **Di chuyển đúng hướng tối ưu** | **`+1`** | Thưởng nhẹ mỗi bước khi xe tiến gần hơn đến hiện trường hoặc bệnh viện. |
| **Bệnh nhân Mức 1 (Đỏ – Nguy kịch) phải chờ** | **`-5` / bước** | Phạt nặng trên mỗi bước thời gian ca nguy kịch chưa được xe tiếp cận. |
| **Bệnh nhân Mức 2 (Vàng – Khẩn cấp) phải chờ** | **`-3` / bước** | Phạt vừa trên mỗi bước thời gian ca khẩn cấp phải chờ. |
| **Bệnh nhân Mức 3 (Xanh – Tiêu chuẩn) phải chờ** | **`-1` / bước** | Phạt nhẹ trên mỗi bước thời gian ca tiêu chuẩn phải chờ. |
| **Chờ quá thời gian tối đa ($T_{\text{wait}} > T_{\max}$)** | **`-100`** | Phạt cực lớn khi để bệnh nhân chờ vượt ngưỡng tối đa (nguy cơ tử vong). |
| **Xe đi vào cung đường đang ùn tắc ($c_e = 2$)** | **`-2` / bước** | Phạt lãng phí thời gian di chuyển khi đi vào tuyến huyết mạch đang kẹt cứng. |
| **Làm trống hoàn toàn trạm trực** | **`-5`** | Phạt khi cả 3 trạm trực đều bị bỏ trống hoàn toàn (mất độ phủ an toàn). |

---

### 3.4. Động Học Chuyển Trạng Thái ($P(s' \mid s, a)$) & Hệ Số Chiết Khấu ($\gamma$)
1. **Động học giao thông:**
   - Trạng thái lưu thông của 5 tuyến đường huyết mạch chuyển đổi theo chuỗi Markov có xét chu kỳ **Giờ bình thường** và **Giờ cao điểm (Rush Hour)**.
   - Vận tốc di chuyển thực tế trên cung đường $e$ tại bước $t$:
     $$v(e, t) = v_{\max} \times \bigl(1 - \alpha \times \text{cong}(e, t)\bigr) + \epsilon$$
     với $v_{\max} = 1.0$, $\alpha = 0.35$, $\text{cong}(e, t) \in \{0, 1, 2\}$ và nhiễu ngẫu nhiên $\epsilon$ tuân theo phân phối **Gaussian** hoặc **Log-Normal**.
2. **Quy luật sinh cuộc gọi (Quá trình Poisson không đồng nhất):**
   - Tại mỗi nút $v \in \{1, \dots, 15\}$, số cuộc gọi mới phát sinh trong khoảng thời gian $\Delta t$ tuân theo phân phối Poisson với cường độ $\lambda(v, t)$ phụ thuộc vào mật độ dân cư và giờ cao điểm:
     $$P(k \text{ cuộc gọi mới}) = \frac{e^{-\lambda(v,t)\Delta t}\bigl(\lambda(v,t)\Delta t\bigr)^k}{k!}$$
3. **Hệ số chiết khấu ($\gamma$):**
   - Mặc định $\gamma = 0.95$ cho các thuật toán bảng (DP, Monte Carlo, Q-Learning) và $\gamma = 0.99$ cho thuật toán PPO (`DEFAULT_GAMMA_TABULAR`, `DEFAULT_GAMMA_PPO`).

---

## 4. Hướng Dẫn Sử Dụng & Chạy Thử Nghiệm

### 4.1. Chạy script minh họa và đánh giá Baseline
```bash
python demo_env.py
```

### 4.2. Chạy bộ kiểm thử tự động (Unit Tests)
```bash
python -m unittest discover -s tests
```

### 4.3. Ví dụ tích hợp môi trường vào thuật toán RL (Q-Learning / Monte Carlo / DP / PPO)
```python
from environment import SmartCityAmbulanceEnv, EnvConfig

# Khởi tạo môi trường (15 nút, 3 xe, tối đa 3 cuộc gọi, 5 tuyến huyết mạch)
env = SmartCityAmbulanceEnv(config=EnvConfig(max_steps=100, gamma=0.95), seed=42)
obs = env.reset(seed=42)

done = False
while not done:
    # 1. Lấy trạng thái:
    state_vec = obs["state_vector"]          # Vector MDP 20 chiều: [X_amb(6), X_req(9), X_traffic(5)]
    norm_vec = obs["normalized_vector"]      # Vector chuẩn hóa [0,1] (20 chiều) cho PPO
    tab_state = obs["tabular_state"]         # Tuple rời rạc thu gọn cho DP / MC / Q-Learning

    # 2. Lấy danh sách hành động hợp lệ (Action Masking):
    valid_actions = obs["valid_actions"]

    # 3. Chọn hành động (ví dụ dùng Baseline hoặc Agent):
    action = env.recommend_baseline_action()

    # 4. Bước chuyển trạng thái MDP:
    obs, reward, done, info = env.step(action)

print("Thống kê kết thúc Episode:", env.get_metrics())
```

---

## 5. Kết Quả Kiểm Chứng Môi Trường (Random vs. Baseline)

Kết quả chạy thử nghiệm `demo_env.py` trên **30 episodes** (mỗi episode 80 bước thời gian) cho thấy môi trường phản ánh chính xác chất lượng điều phối thông qua hàm phần thưởng và các chỉ số y tế:

| Chính sách điều phối | Tổng Reward TB | Thời gian phản hồi TB | Tỷ lệ đạt Thời gian vàng | Số ca chuyển viện TB | Số ca quá hạn ($> T_{\max}$) TB |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Ngẫu nhiên (Random)** | `-810.57` | `6.44` bước | `60.59%` | `12.10` ca | `8.63` ca |
| **Baseline (Xe gần nhất + Ưu tiên)** | **`+1210.23`** | **`4.10` bước** | **`83.98%`** | **`26.87` ca** | **`0.33` ca** |

