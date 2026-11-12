# BÁO CÁO MÔ HÌNH MDP CHO BÀI TOÁN ĐIỀU PHỐI XE CỨU THƯƠNG

**Phiên bản:** 1.0 — đối chiếu với mã nguồn trong workspace ngày 30/10/2026  
**Phạm vi:** mô phỏng điều phối tập trung trên bản đồ đô thị tổng hợp.  
**Tình trạng:** đặc tả nghiên cứu dựa trên code hiện có; các khác biệt và lỗi đã nhận diện được ghi tại Mục 10. Báo cáo này không khẳng định mô phỏng đã được hiệu chuẩn bằng dữ liệu thực.

## 1. Phát biểu bài toán

Một trung tâm điều phối quản lý một đội xe cứu thương và tiếp nhận các cuộc gọi khẩn cấp phát sinh ngẫu nhiên trên mạng lưới giao thông. Tại đầu mỗi phút, trung tâm quan sát trạng thái hệ thống và quyết định phân công đồng thời một số xe đang rảnh cho các cuộc gọi đang chờ. Mỗi xe và mỗi cuộc gọi chỉ tham gia tối đa một phân công tại thời điểm đó.

Xe được phân công di chuyển tới hiện trường, sơ cứu, vận chuyển bệnh nhân đến bệnh viện và bàn giao. Trong lúc đó, giao thông thay đổi và các cuộc gọi mới phát sinh. Hệ thống ngừng nhận cuộc gọi sau một khoảng thời gian, sau đó tiếp tục xử lý công việc còn lại.

**Mục tiêu nghiên cứu:** tìm chính sách điều phối làm tăng tổng phần thưởng tích lũy. Hàm thưởng hiện tại khuyến khích phục vụ cuộc gọi và phạt thời gian chờ, thất bại và quãng đường di chuyển. Các chỉ số thực tế như thời gian đáp ứng, số ca quá hạn và mức độ ưu tiên vẫn cần được báo cáo riêng; reward không tự chứng minh chất lượng y tế.

### 1.1. Cấu hình mặc định

| Tham số | Giá trị | Đơn vị / ý nghĩa |
|---|---:|---|
| Số nút giao | 80 | nút |
| Số cung đường có hướng | 172 | cung; tương ứng 86 tuyến hai chiều trong bản đồ mặc định, kiểm tra từ object `CityMap` |
| Số xe | 6 | xe cứu thương, có thể cấu hình |
| Số bệnh viện | 4 | bệnh viện |
| Số cuộc gọi `PENDING` tối đa | 3 | cuộc gọi; các cuộc gọi đã giao xe không thuộc giới hạn này |
| Cường độ sinh cuộc gọi | 0,1 | cuộc gọi/phút; tham số Poisson, không phải xác suất có cuộc gọi |
| Thời gian nhận cuộc gọi | 360 | phút |
| Giới hạn quá giờ | 300 | phút sau thời gian nhận cuộc gọi |
| Seed mặc định | 42 | dùng để tái lập kịch bản, không phải dữ liệu thực |
| Khoảng thời gian quyết định | 1 | phút |

Nguồn: `dispatch_env/types.py`, `map_env/city_map.py`, `run_simulation.py`. Các thông số là cấu hình mô phỏng và có thể thay đổi khi khởi tạo môi trường.

## 2. Ký hiệu và cấu trúc MDP

Mô hình nghiên cứu được viết:

\[
\mathcal M = (\mathcal S,\mathcal A,P,R,\gamma,\rho_0).
\]

| Ký hiệu | Ý nghĩa trong bài toán |
|---|---|
| \(\mathcal S\) | Không gian trạng thái đầy đủ của hệ thống tại thời điểm ra quyết định |
| \(\mathcal A(s)\) | Tập hành động phân công đồng thời hợp lệ tại trạng thái \(s\) |
| \(P\) | Quy luật sinh trạng thái tiếp theo từ trạng thái và hành động hiện tại |
| \(R\) | Quy tắc tính phần thưởng trong một bước |
| \(\gamma\) | Hệ số chiết khấu dùng trong mục tiêu huấn luyện |
| \(\rho_0\) | Phân phối khởi tạo của một episode |

Mục tiêu thông thường là tìm chính sách \(\pi\) tối đa hóa:

\[
J(\pi)=\mathbb E_{\rho_0,\pi,P}\left[\sum_{t=0}^{T-1}\gamma^t r_t\right].
\]

Trong đó \(T\) là thời điểm kết thúc một episode. **Code hiện tại chưa chọn thuật toán huấn luyện hoặc giá trị \(\gamma\)**; giá trị này cần được công bố khi thiết lập thí nghiệm. Nếu chọn \(\gamma<1\), reward muộn có trọng số nhỏ hơn reward sớm, vì vậy cần báo cáo cả các KPI không chiết khấu.

### 2.1. Trạng thái đầy đủ và observation

Gọi trạng thái nội bộ đầy đủ là \(S_t\), và dữ liệu mà agent nhận được là \(O_t=h(S_t)\). Báo cáo định nghĩa trạng thái MDP mục tiêu tại Mục 3; **không đồng nhất nó với observation hiện tại**, vì observation vẫn thiếu một số biến ảnh hưởng tới chuyển trạng thái. Khi chưa bổ sung hoặc chứng minh được khả năng suy ra các biến đó, agent đang nhận quan sát một phần, chưa chứng minh được điều kiện Markov đầy đủ.

Các tham số cố định như đồ thị, độ dài cung và vị trí bệnh viện có thể coi là cấu hình \(\theta\) của một môi trường cố định. Nếu cấu hình thay đổi giữa các episode và agent cần thích nghi, cần đưa thông tin đủ về cấu hình vào observation hoặc công bố bài toán quan sát một phần.

## 3. Không gian trạng thái \(\mathcal S\)

Tại đầu phút \(t\), trước khi phân công, biểu diễn trạng thái mục tiêu:

\[
S_t=(t,\tau_t,X_t^{\mathrm{traffic}},X_t^{\mathrm{ambulance}},X_t^{\mathrm{call}},Z_t).
\]

Trong đó \(\tau_t\) là đồng hồ/pha giao thông và \(Z_t\) là các biến nội bộ còn cần thiết cho chuyển trạng thái, cập nhật reward và số liệu theo đúng quy tắc cài đặt. Không cần đưa seed vào trạng thái MDP lý thuyết; seed phục vụ tái lập việc lấy mẫu trong chương trình.

### 3.1. Thời gian

| Trường | Ý nghĩa | Đơn vị |
|---|---|---|
| `current_minute` | Thời điểm ra quyết định | phút |
| `traffic_time_step` | Đồng hồ dùng xác định pha giao thông | bước/phút |
| Giai đoạn nhận cuộc gọi | Suy ra từ thời gian và `max_time` | nhị phân |

Thời gian cần được giữ trong trạng thái vì sinh cuộc gọi dừng tại `max_time` và xác suất giao thông phụ thuộc pha chu kỳ. Khi đồng hồ giao thông luôn đồng bộ, pha có thể suy ra từ thời gian; trong code hiện tại cả hai được ghi nhận để kiểm tra tính nhất quán.

### 3.2. Giao thông

Với mỗi cung \(e\), \(g_{e,t}\in\{1,2,3\}\) lần lượt tương ứng thông thoáng, bình thường và tắc. Vector toàn mạng là \(X_t^{\mathrm{traffic}}=(g_{e,t})_{e\in E}\), theo thứ tự cung cố định.

| Khu vực | Thông thoáng | Bình thường | Tắc |
|---|---:|---:|---:|
| Trung tâm | 40 | 25 | 12 |
| Ngoại ô | 60 | 40 | 20 |

Đơn vị vận tốc: km/giờ. Đặc tính vùng được xác định bởi thuộc tính `is_center` trên cung. Khi chuyển sang km/phút, vận tốc bằng vận tốc km/giờ chia cho 60.

### 3.3. Xe cứu thương

Với mỗi xe \(i\), trạng thái mục tiêu cần chứa hoặc cho phép suy ra:

| Biến | Nội dung / đơn vị | Tác dụng |
|---|---|---|
| `status` | rảnh, đến cuộc gọi, sơ cứu, đến bệnh viện, bàn giao | xác định các chuyển trạng thái hợp lệ |
| `current_edge`, `distance_from_start` | cung có hướng và vị trí trên cung, km | xác định vị trí thực tế |
| `current_call_id` | cuộc gọi liên kết, nếu có | liên kết xe và nhiệm vụ |
| `target_type`, `target_id`, `target_edge`, `target_dist` | loại và vị trí đích | mô tả điểm đến |
| `route_edges` | các cung còn lại trong tuyến | điều khiển di chuyển thực tế |
| `scene_time_remaining` | thời gian sơ cứu còn lại, phút | quyết định khi nào kết thúc sơ cứu |
| `hospital_time_remaining` | thời gian bàn giao còn lại, phút | quyết định khi nào xe rảnh |
| `assigned_hospital_id` | bệnh viện đã chọn | xác định đích và thời gian bàn giao |

`get_observation()` hiện đã trả vị trí, trạng thái, ID cuộc gọi, tuyến và một số thông tin tốc độ; nhưng chưa trả đầy đủ thời gian sơ cứu/bàn giao còn lại và các trường đích ở trên. Danh sách tuyến hiện được trả trực tiếp từ object xe, nên observation cũ có thể bị thay đổi khi tuyến nội bộ thay đổi. Xem `dispatch_env/environment.py:get_observation` và `dispatch_env/ambulance.py:Ambulance`.

### 3.4. Cuộc gọi

Với mỗi cuộc gọi còn hoạt động, các thông tin cần có hoặc suy ra được gồm: ID hoặc slot; vị trí trên cung; mức độ; trạng thái; thời điểm sinh; deadline; ngưỡng hoàn hảo; thời gian sơ cứu; thời gian đã chờ; và xe được phân công.

| Mức | Xác suất sinh | Thời gian chờ cho phép | Thời gian sơ cứu |
|---|---:|---:|---:|
| 1 | 0,40 | 60 phút | 4 phút |
| 2 | 0,35 | 40 phút | 6 phút |
| 3 | 0,25 | 30 phút | 8 phút |

Các xác suất và thời gian là giả định trong code. Deadline là thời điểm sinh cộng thời gian chờ cho phép. Các trạng thái cuộc gọi gồm `PENDING`, `WAITING_FOR_AMBULANCE`, `AT_SCENE`, `COMPLETED`, `FAILED`.

Tối đa 3 cuộc gọi **chưa được phân công** có thể ở `PENDING`. Ngoài ra còn có các cuộc gọi đang được xe tiếp cận hoặc sơ cứu. Ba slot chờ không đại diện cho toàn bộ cuộc gọi hoạt động.

Nếu observation giữ các ID tăng dần, cần kiểm tra biến `next_call_id` và quy tắc liên kết có đủ để observation thực sự Markov; một phương án khác là xây dựng biểu diễn slot/ID chuẩn hóa và chứng minh việc ánh xạ không làm mất thông tin. Các biến lịch sử hoàn thành hoặc cờ đã tính reward cần được rà soát theo các bất biến của code, không mặc định rằng chỉ thêm vài trường đã đủ.

## 4. Không gian hành động \(\mathcal A\)

### 4.1. Hành động phân công đồng thời

Tại đầu phút, một hành động là tập ghép \(a_t\) gồm các cặp \((i,j)\), trong đó xe \(i\) nhận cuộc gọi `PENDING` ở slot \(j\). Ràng buộc:

1. Xe được chọn đang `IDLE` và không có `current_call_id`.
2. Slot chứa một cuộc gọi đang `PENDING`.
3. Một xe và một cuộc gọi xuất hiện tối đa một lần.
4. Tập rỗng là hành động không phân công, luôn hợp lệ.

Các slot được sắp xếp tăng dần theo `(created_time, call_id)`. Ánh xạ từ slot sang ID phải được hiểu tại thời điểm chọn hành động.

Với \(N\) xe và \(M\) slot, số cách ghép tiềm năng là:

\[
|\mathcal A_{\mathrm{potential}}|=\sum_{k=0}^{\min(N,M)}\binom Nk\binom Mk k!.
\]

| Cấu hình | Số cách ghép, bao gồm không phân công |
|---|---:|
| 6 xe, 3 slot — mặc định hiện tại | 229 |
| 4 xe, 3 slot | 73 |

`get_action_mask()` đánh dấu các cách ghép hiện hợp lệ. Số cách ghép tăng khi tăng số xe/slot, do đó cần đánh giá khả năng mở rộng nếu thay đổi cấu hình.

### 4.2. Hành vi code hiện tại

`step()` nhận action ID, một cặp, danh sách các cặp, hoặc `None`. `auto_dispatch=True` và không truyền hành động kích hoạt heuristic: ưu tiên mức độ cao hơn, sau đó chọn xe có ETA đến hiện trường ngắn nhất. Đây là baseline, không phải hành động của agent học.

Đối với hành động sai, `decode_action()` hiện có thể **lọc bỏ cặp không hợp lệ**, biến hành động thành ghép một phần hoặc không phân công. Vì vậy, trong thí nghiệm nên dùng action mask và cần quy định rõ/kiểm thử lại cách xử lý hành động sai. Code hiện chưa có `action_space` chuẩn Gymnasium.

## 5. Quy luật chuyển trạng thái \(P\)

Môi trường sử dụng bộ sinh chuyển trạng thái:

\[
S_{t+1}=f(S_t,A_t,\xi_t;\theta),
\]

với \(\xi_t\) bao gồm các mẫu ngẫu nhiên về giao thông và cuộc gọi. Vì trạng thái lớn và có các biến vị trí/thời gian dạng số thực, báo cáo không liệt kê ma trận \(P\) đầy đủ. Về mặt tổng quát:

\[
P(B\mid s,a)=\Pr(S_{t+1}\in B\mid S_t=s,A_t=a)
\]

cho một tập trạng thái \(B\). Bộ mô phỏng cho phép lấy mẫu chuyển trạng thái; không cần tạo bảng toàn bộ \(P\) để dùng phương pháp model-free. Các phân phối dưới đây là **quy tắc tổng hợp trong code**, không phải xác suất được suy ra từ dữ liệu bệnh viện thực.

### 5.1. Thứ tự một bước

1. Lấy slot và mask; giải mã hành động và phân công xe hợp lệ.
2. Mỗi xe di chuyển/xử lý liên tục trong khoảng một phút, dùng thời gian còn lại nếu chuyển giai đoạn giữa phút.
3. Cập nhật phạt chờ, kiểm tra thất bại và ghi nhận các cuộc gọi vừa hoàn thành sơ cứu.
4. Tăng thời gian; cập nhật giao thông và tính lại tuyến của xe đang di chuyển.
5. Sinh cuộc gọi tại đầu phút mới nếu chưa đến `max_time`; cuộc gọi vượt giới hạn `PENDING` bị từ chối.
6. Tính reward và kiểm tra kết thúc hoặc truncation.

Cần phân biệt thứ tự trên là **hành vi cài đặt hiện tại**; một số trường hợp deadline và reward ban đầu còn sai lệch so với ý định, xem Mục 10.

### 5.2. Cuộc gọi mới

Tại các thời điểm được phép sinh cuộc gọi:

\[
N_t\sim\operatorname{Poisson}(\lambda),\qquad
\Pr(N_t=k)=e^{-\lambda}\frac{\lambda^k}{k!}.
\]

Mặc định \(\lambda=0,1\) cuộc gọi/phút. Với `generate_initial_calls=True`, code lấy mẫu tại \(t=0\), rồi tại các phút \(1,\ldots,359\) với `max_time=360`; không lấy mẫu tại \(t\ge360\). Mỗi cuộc gọi được chọn đều theo danh sách các cung có hướng; vị trí trên cung được lấy mẫu trong khoảng code quy định rồi làm tròn 0,01 km. Đây không phải phân phối theo dân số, mật độ nhu cầu hoặc chiều dài đường. Mức độ được lấy theo xác suất trong Mục 3.4.

Trong code, bộ sinh của môi trường và bộ sinh giao thông là các generator riêng; khi cung cấp cùng seed, không nên suy diễn rằng các kịch bản đầu vào đã được thiết kế độc lập về mặt thống kê cho mọi mục đích đánh giá. Nếu so sánh nhiều chính sách, nên thiết kế luồng ngẫu nhiên hoặc kịch bản chung có kiểm soát.

### 5.3. Chuyển trạng thái giao thông

Mỗi phút mỗi cung cập nhật theo ma trận phụ thuộc vùng và pha giờ cao điểm. Hàng là mức tắc hiện tại, cột là mức ở phút sau, thứ tự mức `1, 2, 3`. Cao điểm khi \(10\le(\tau\bmod40)\le25\), với \(\tau\) là đồng hồ sau khi tăng trong bước cập nhật.

**Trung tâm, cao điểm:**

\[
\begin{bmatrix}
0.30&0.40&0.30\\
0.10&0.45&0.45\\
0.02&0.28&0.70
\end{bmatrix}
\]

**Trung tâm, bình thường:**

\[
\begin{bmatrix}
0.45&0.35&0.20\\
0.20&0.50&0.30\\
0.05&0.35&0.60
\end{bmatrix}
\]

**Ngoại ô, cao điểm:**

\[
\begin{bmatrix}
0.55&0.30&0.15\\
0.25&0.45&0.30\\
0.10&0.45&0.45
\end{bmatrix}
\]

**Ngoại ô, bình thường:**

\[
\begin{bmatrix}
0.70&0.22&0.08\\
0.35&0.45&0.20\\
0.20&0.50&0.30
\end{bmatrix}
\]

Nguồn: `map_env/edge.py:update_congestion_markov` và `map_env/city_map.py:step_traffic`.

### 5.4. Xe và bệnh viện

Xe được điều tới hiện trường bằng tuyến thời gian ngắn nhất theo giao thông hiện tại. Khi sơ cứu xong, code tự chọn bệnh viện có ETA ngắn nhất; agent hiện không quyết định bệnh viện hay tuyến đường. Xe tiếp tục vận chuyển, bàn giao và trở về `IDLE`. Thời gian bàn giao của bốn bệnh viện mặc định lần lượt là 3, 4, 5, 4 phút.

## 6. Phần thưởng \(R\)

### 6.1. Công thức dự định và cấu hình đang thực thi

\[
r_t=-\sum_c w_c\Delta W_{c,t}
    -\sum_{c\in F_t}p_c
    -0.05\sum_i\Delta d_{i,t}
    +\sum_{c\in C_t}b_c.
\]

| Ký hiệu | Đơn vị / nội dung |
|---|---|
| \(\Delta W_{c,t}\) | thời gian chờ được tính trong bước, phút |
| \(w_c\) | mức phạt chờ theo mức độ, điểm/phút |
| \(F_t\) | sự kiện thất bại được ghi nhận trong bước |
| \(p_c\) | phạt thất bại, điểm/lần |
| \(\Delta d_{i,t}\) | quãng đường xe đi trong bước, km |
| \(C_t\) | các cuộc gọi vừa hoàn thành sơ cứu trong bước |
| \(b_c\) | điểm hoàn thành thông thường hoặc hoàn hảo |

**Bảng giá trị đang thực thi tại `dispatch_env/types.py:REWARD_CONFIG`:**

| Mức | Phạt chờ/phút | Phạt thất bại/lần | Hoàn thành | Hoàn hảo |
|---|---:|---:|---:|---:|
| 1 | 0,5 | 30 | 50 | 70 |
| 2 | 1,0 | 60 | 65 | 80 |
| 3 | 1,5 | 120 | 70 | 100 |

Phạt quãng đường là 0,05 điểm/km. Điểm hoàn hảo trong bảng là **tổng điểm hoàn thành**, không cộng thêm điểm hoàn thành thông thường. Cuộc gọi được đánh dấu hoàn thành khi sơ cứu xong, không phải khi đã bàn giao tại bệnh viện. Hoàn hảo nếu thời điểm xong sơ cứu không muộn hơn `created_time + 0.5 * failure_time`.

**Khác biệt với bảng người dùng từng đề xuất:** phạt chờ `1/2/4`, thưởng hoàn thành `80` và hoàn hảo `100` cho mọi mức. Đó là một cấu hình đề xuất khác, **chưa phải giá trị code đang chạy**. Nếu chọn áp dụng, cần đổi code, test, README và ghi lại phiên bản thí nghiệm.

Trong điều kiện thông thường, các cuộc gọi bị từ chối tại cuối bước được ghi vào phạt thất bại. Tuy nhiên, những sai lệch tại \(t=0\) và deadline được mô tả tại Mục 10; vì vậy không được coi công thức lý tưởng tự động mô tả chính xác mọi tình huống hiện tại.

### 6.2. Log

Mỗi bước hiện ghi reward, các thành phần phạt chờ, thất bại, quãng đường, điểm hoàn thành và tổng tích lũy trong `simulation.log`. Khi `reset`, các trường reward được đặt về 0; vấn đề thất bại lúc khởi tạo được nêu riêng bên dưới.

## 7. Phân phối khởi tạo \(\rho_0\) và reset

Ở cấu hình mặc định, khi bắt đầu episode:

- Thời gian mô phỏng và đồng hồ giao thông về 0; thống kê và reward được xóa.
- Giao thông ban đầu được lấy mẫu với xác suất mức 1/2/3 tương ứng `0,60/0,30/0,10` theo quy tắc hiện tại của `CityMap`.
- Xe ban đầu rảnh; các cung khởi đầu được lấy mẫu không lặp, sau đó lấy vị trí trên cung theo code và làm tròn 0,01 km.
- Có thể sinh các cuộc gọi đầu tiên ngay tại \(t=0\).
- Bản đồ truyền vào được giữ về cấu trúc; `reset_traffic` có thể thay đổi mức tắc ban đầu theo quy tắc của môi trường.

`reset(seed=x)` khởi tạo lại các bộ sinh theo seed; `reset()` tiếp tục quá trình lấy mẫu để tạo episode mới. Observation có cả `episode_count`, nên kiểm thử tái lập nên so sánh những trường thuộc kịch bản và chuỗi kết quả, hoặc xử lý riêng bộ đếm episode. Việc cùng seed không đảm bảo hai chính sách khác nhau tiêu thụ ngẫu nhiên theo cùng thứ tự; để so sánh công bằng nên dùng kịch bản đầu vào hoặc luồng ngẫu nhiên được tách và kiểm soát.

## 8. Kết thúc, truncation và giao diện

Sau `max_time`, kết thúc thông thường khi đồng thời:

1. Không còn cuộc gọi hoạt động (`PENDING`, `WAITING_FOR_AMBULANCE`, `AT_SCENE`).
2. Mọi xe đều `IDLE`, bao gồm đã xong vận chuyển và bàn giao.

Nếu chưa kết thúc bình thường mà đạt `max_time + max_overtime` — mặc định phút 660 — episode bị dừng và báo `truncated`, kèm backlog. Dừng vì giới hạn kỹ thuật không đồng nghĩa tất cả ca đã được phục vụ.

API code hiện tại:

```python
obs = env.reset(seed=42)
obs, reward, done, info = env.step(action)
```

`info["truncated"]` phân biệt trường hợp dừng quá giờ. Đây là API Python riêng, chưa phải Gymnasium chuẩn `reset -> (obs, info)` và `step -> (obs, reward, terminated, truncated, info)`; observation cũng chưa có không gian kiểu/kích thước cố định được khai báo.

## 9. Dữ liệu, kiểm chứng và đánh giá

### 9.1. Nguồn và mức độ tin cậy

| Dữ liệu/tham số | Nguồn hiện tại | Tình trạng |
|---|---|---|
| Đồ thị 80 nút, bệnh viện | Mã nguồn `map_env/city_map.py` | bản đồ tổng hợp |
| Ma trận giao thông, tốc độ | `map_env/edge.py` | giả định mô phỏng |
| Tần suất, vị trí, mức độ cuộc gọi | `environment.py`, `types.py` | phân phối tổng hợp |
| Thời gian sơ cứu, deadline và reward | `types.py`, `call.py` | cấu hình nghiên cứu |
| KPI và reward từ mỗi lần chạy | `step()`, `simulation.log` | đầu ra mô phỏng |

Chưa có bằng chứng trong dự án cho thấy các tham số đã được hiệu chuẩn bằng dữ liệu vận hành thực tế. Không dùng các chỉ số mô phỏng để khẳng định hiệu quả cấp cứu thực tế khi chưa có dữ liệu, hiệu chuẩn và thẩm định phù hợp.

### 9.2. Kiểm thử đã chạy

Lệnh kiểm tra ngày 30/10/2026: `python -m unittest discover -s tests -v` với trình thông dịch Python cài trên máy và `PYTHONDONTWRITEBYTECODE=1`. Kết quả: **31/31 test đạt**. Điều này xác nhận các test hiện có vượt qua, không chứng minh mọi trường hợp biên hay sự phù hợp dữ liệu thực. Cần bổ sung test cho các mục tại Mục 10.

### 9.3. Kế hoạch đánh giá chính sách

So sánh ít nhất một chính sách ngẫu nhiên hợp lệ và heuristic ưu tiên mức độ/ETA hiện có với agent. Dùng cùng cấu hình, tập kịch bản đánh giá được kiểm soát, nhiều episode và các tập train/validation/test tách biệt. Báo cáo:

- Thời gian từ phát sinh cuộc gọi đến xe tới hiện trường: trung bình, trung vị và P90/P95.
- Tỷ lệ hoàn thành sơ cứu, quá hạn và bị từ chối, phân tách theo mức độ.
- Thời gian bàn giao, quãng đường, tỷ lệ xe bận và backlog khi truncated.
- Tổng reward và các thành phần reward, cùng số episode và mức độ phân tán.

Khi so sánh, không chỉ dựa vào reward: trọng số reward là lựa chọn thiết kế và có thể không phản ánh đầy đủ mục tiêu nghiệp vụ.

## 10. Những điểm cần sửa hoặc quyết định trước khi tuyên bố môi trường hoàn thiện

1. **Bảng reward chưa khớp bảng người dùng đề xuất.** Cần chốt bảng nào là chuẩn và cập nhật code, test, tài liệu đồng bộ.
2. **Quan sát chưa chứng minh đủ Markov.** Cần bổ sung/kiểm chứng thời gian sơ cứu, bàn giao, đích và các biến liên quan; sao chép tuyến đường khi tạo snapshot; kiểm tra ID tăng dần và các cờ sự kiện.
3. **Thất bại ở \(t=0\).** `reset()` có thể từ chối cuộc gọi do đầy hàng chờ, nhưng reward ban đầu vẫn 0 và danh sách sự kiện có thể bị xóa trước khi tính bước đầu. Cần chốt cách hạch toán và kiểm thử.
4. **Deadline và thời gian chờ.** Đến đúng deadline được cho là thành công; kiểm tra thất bại chỉ khi đã vượt deadline có thể làm tính thêm một phút chờ. Trường hợp xe dự kiến đến muộn có nhánh di chuyển đến đích trước khi hủy. Cần đặc tả thời điểm timeout và test đến trước/đúng/sau deadline, quãng đường và reward.
5. **Hành động sai.** Code hiện lọc cặp không hợp lệ thay vì luôn báo lỗi; cần chốt hợp đồng cho agent, đồng bộ action mask và test.
6. **API và thí nghiệm.** Chưa có Gymnasium spaces chuẩn, chương trình huấn luyện hoặc đánh giá baseline nhiều seed được xác nhận trong dự án. Cần triển khai nếu muốn tích hợp thư viện RL.
7. **Thiếu dữ liệu hiệu chuẩn.** Các giả định về nhu cầu, giao thông, xử lý và bệnh viện cần dữ liệu nguồn và quy trình đánh giá trước khi dùng cho kết luận ứng dụng thực tế.
8. **Số cung trong README khác runtime.** README ghi 176 cung/88 tuyến; khi khởi tạo bản đồ mặc định, code thực tế có 172 cung/86 tuyến. Cần đối chiếu và cập nhật tài liệu hoặc cấu trúc bản đồ theo yêu cầu.

## 11. Tham chiếu mã nguồn

- `dispatch_env/environment.py`: reset, observation, slot, mask, step, reward, log và kết thúc.
- `dispatch_env/types.py`: cấu hình mặc định, deadline, thời gian sơ cứu và reward.
- `dispatch_env/ambulance.py`: di chuyển, sơ cứu, chọn bệnh viện và bàn giao.
- `dispatch_env/call.py`: deadline, hoàn hảo và trạng thái cuộc gọi.
- `map_env/city_map.py`, `map_env/edge.py`: cấu trúc đường, reset/cập nhật giao thông và ma trận chuyển.
- `run_simulation.py`, `tests/`: chạy heuristic, log và bộ kiểm thử.
