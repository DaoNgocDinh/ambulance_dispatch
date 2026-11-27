# RL Algorithm Development & Evaluation Standard v1.0

**Dự án:** Ambulance Dispatch Simulation
**Repository:** `E:\ambulance_dispatch`
**Đối tượng áp dụng:** ba thành viên phát triển thuật toán Reinforcement Learning và AI Agent hỗ trợ từng thành viên
**Trạng thái tài liệu:** quy chuẩn chung của nhóm (G0). Không thay thế tài liệu MDP, không thay thế báo cáo Environment.
**Tài liệu tham chiếu đã đọc:** `README.md`, `BAO_CAO_MDP.md`, `M1_AUDIT_SIMULATION_ENVIRONMENT.md`, `M2_MDP_SPECIFICATION.md`, `M3_IMPLEMENTATION_REPORT.md`, `M4_DOCUMENTATION_HANDOFF.md`, source code `dispatch_env/`, `map_env/`, `tests/`.

---

## 0. Cách đọc tài liệu này

Tài liệu phân biệt ba mức yêu cầu:

| Nhãn | Ý nghĩa |
|---|---|
| **MUST – Bắt buộc** | Vi phạm thì kết quả không được dùng để so sánh hoặc kết luận. |
| **SHOULD – Khuyến nghị** | Nên làm; nếu bỏ qua phải ghi rõ lý do trong báo cáo. |
| **MAY – Được tự do** | Mỗi thuật toán tự quyết định theo đặc trưng của mình. |

Nguyên tắc xuyên suốt:

- Đây là **quy chuẩn chung của nhóm**, không phải yêu cầu thay thế thuật toán hiện có.
- Ba thuật toán **không bị áp đặt cùng một cách triển khai**.
- Không tự ý thay đổi phạm vi hoặc kiến trúc dự án.
- Environment/MDP hiện tại đã qua các giai đoạn audit (M1), specification (M2), stabilization (M3) và documentation (M4). **Không xem đây là yêu cầu xây dựng lại Environment.**
- Dùng tiếng Việt; giữ nguyên thuật ngữ kỹ thuật tiếng Anh khi cần (Observation, Action Mask, Reward, Episode, Truncation, Checkpoint, Seed, Policy, ...).

---

## 1. Mục tiêu tài liệu

Tài liệu này truyền đạt thống nhất định hướng và yêu cầu của người phụ trách Environment/MDP tới cả ba thành viên đang phát triển thuật toán RL. Mỗi thành viên và AI Agent hỗ trợ phải đọc, hiểu và tuân thủ tài liệu này khi phát triển, kiểm tra, huấn luyện và đánh giá thuật toán.

Tài liệu phải:

- Rõ ràng, có tính quy chuẩn và có thể thực hiện được.
- Phân biệt yêu cầu bắt buộc với đề xuất khuyến nghị.
- Không áp đặt ba thuật toán phải có cùng cách triển khai.
- Không tự ý thay đổi phạm vi hoặc kiến trúc dự án.

---

## 2. Bối cảnh dự án

Dự án mô phỏng điều phối xe cứu thương sử dụng Reinforcement Learning.

Hiện có ba thành viên phát triển ba thuật toán RL khác nhau. Các nhánh thuật toán **chưa được merge vào `main`** vì kết quả hiện tại chưa đạt kỳ vọng và cần được đánh giá, cải thiện thêm.

Mục tiêu chung là xây dựng Agent có khả năng:

- Phục vụ được nhiều cuộc gọi cấp cứu hơn.
- Giảm số cuộc gọi thất bại.
- Ưu tiên xử lý các cuộc gọi nghiêm trọng và khẩn cấp.
- Giảm thời gian chờ và thời gian phản hồi.
- Sử dụng đội xe cứu thương hiệu quả.
- Hoạt động ổn định trên nhiều tình huống mô phỏng.

Heuristic hiện tại (`env.step(auto_dispatch=True)`, ưu tiên mức độ nghiêm trọng cao hơn rồi chọn xe có ETA đến hiện trường ngắn nhất) là **Baseline để so sánh**, không phải một thuật toán RL.

> **Trạng thái repository (theo M4):** Environment MDP bằng Python/numpy đã có; heuristic baseline đã có; **chưa có** RL Agent, pipeline huấn luyện/đánh giá RL, Gymnasium `action_space`/`observation_space`, và chưa tách `terminated`/`truncated`. Nhóm RL tự chịu trách nhiệm xây vòng huấn luyện/đánh giá.

---

## 3. Quy chuẩn môi trường và MDP

Cả ba thuật toán **MUST** sử dụng chung:

- Environment và phiên bản mã nguồn được nhóm thống nhất.
- Cấu hình bản đồ, đường đi, giao thông và bệnh viện.
- Cấu hình số lượng xe cứu thương.
- Cơ chế sinh cuộc gọi.
- Observation do Environment cung cấp.
- Action Space và cách mã hóa Action ID.
- Action Mask.
- Transition và Reward do Environment cung cấp.
- Điều kiện kết thúc Episode và Truncation.

Cấu hình mặc định đang thực thi (đối chiếu `dispatch_env/types.py`, `map_env/city_map.py`, `README.md`):

| Tham số | Mặc định | Ghi chú |
|---|---:|---|
| `num_ambulances` | 6 | Có thể cấu hình; không tự đổi cho thí nghiệm so sánh |
| `max_time` | 360 phút | Ngừng sinh cuộc gọi mới sau mốc này |
| `max_overtime` | 300 phút | Truncation tại `max_time + max_overtime = 660` |
| `poisson_lambda` | 0.1 | Cuộc gọi/phút (tham số Poisson, không phải xác suất) |
| `max_pending_calls` | 3 | Số cuộc gọi `PENDING` tối đa; tương ứng 3 slot |
| Số nút / cung có hướng | 80 / 172 | `edge_id = 1..172` |
| Số bệnh viện | 4 | Handover time lần lượt 3, 4, 5, 4 phút |
| Bước quyết định | 1 phút | Mỗi `step()` mô phỏng `[t, t+1)` |

Các quy định:

1. **MUST** – Agent chỉ nhận thông tin từ Environment thông qua giao diện đã có (`reset()`, `step()`, `get_observation()`, `get_action_mask()`, `encode_action()`, `decode_action()`).
2. **MUST** – Agent không được tự sửa đổi trạng thái Environment (không ghi vào `env.ambulances`, `env.calls`, `env.rng`, `city_map`, ...).
3. **MUST** – Agent không được tự tính lại hoặc thay thế Reward của Environment. Reward dùng để học phải là `reward` do `step()` trả về (và xử lý minh bạch khoản `t=0` – xem Mục 10).
4. **MUST** – Agent phải tôn trọng Action Mask, không lựa chọn Action bị Mask.
5. **MUST** – Không tự ý thay đổi Reward, Observation, Action Space hoặc luật mô phỏng trong nhánh thuật toán.
6. **MUST** – Nếu phát hiện lỗi hoặc hạn chế của Environment, ghi nhận thành **đề xuất có bằng chứng** để nhóm đánh giá chung; không tự sửa riêng.

**Hợp đồng invalid action hiện tại [Verified]:** `action_id` ngoài khoảng `[0, num_potential_actions - 1]` trả `[]` (no-op); `action_id` trong khoảng nhưng `action_mask[id] == False` vẫn được decode từng cặp, cặp sai bị lọc và các cặp còn lại có thể được thực thi (**partial execution**); `step()` **không enforce** mask và **không raise**. Vì vậy yêu cầu số 4 là trách nhiệm của Agent, không phải của Environment.

---

## 4. Quy chuẩn triển khai thuật toán

Ba thành viên **MAY** tự do triển khai theo đặc trưng thuật toán của mình.

Các thành phần **MAY** khác nhau:

- Thuật toán RL.
- State Encoder / State Representation.
- Learning Method.
- Exploration Strategy.
- Policy.
- Hyperparameters.
- Cấu trúc lưu trữ dữ liệu học như Q-table, trajectory hoặc model.
- Cơ chế cập nhật Policy.

Tuy nhiên **MUST** đáp ứng:

- Đầu vào phù hợp với Observation và Action Mask.
- Đầu ra là Action ID hợp lệ (hoặc cặp `(ambulance_id, call_id)` được encode thành Action ID hợp lệ).
- Không chọn Action bị Mask.
- Không can thiệp trực tiếp vào Environment.
- Có thể phân biệt Training và Evaluation.
- Có khả năng lưu và tải Policy hoặc trạng thái học để đánh giá lại.

### 4.1. Yêu cầu đối với State Encoder

Mỗi thành viên **MUST** giải thích rõ trong báo cáo:

- State được xây dựng từ những thông tin nào trong Observation?
- Thông tin nào được giữ lại?
- Thông tin nào bị loại bỏ hoặc biến đổi?
- Việc Discretization hoặc Feature Engineering được thực hiện như thế nào?
- Vì sao cách biểu diễn đó phù hợp với thuật toán?
- Có nguy cơ mất thông tin ảnh hưởng đến quyết định điều phối hay không?

**Không mặc định State Encoder của ba thuật toán phải giống nhau.** Tuy vậy, State Encoder **MUST** xử lý đúng các khóa thực có trong Observation (xem Mục 14.1) và không được giả định các trường không tồn tại.

Lưu ý về Observation thực tế:

- `ambulances` là `dict[int, dict]`; thứ tự ambulance ổn định theo `ambulance_id`.
- `call_slots` là list độ dài 3; phần tử `PENDING` hoặc `None`; slot được sắp xếp canonical theo `(created_time, call_id)`.
- `action_mask` là `np.ndarray[bool]` shape `(num_potential_actions,)` = `(229,)` với 6 xe / 3 slot.
- Raw `call_id` và `ambulance_id` nên được coi là **identifier**, không mặc định là policy feature có ý nghĩa tổng quát.
- Tên trạng thái hiển thị trong Observation là tiếng Việt (ví dụ `"Đang đến cuộc gọi"`), trong khi enum nội bộ là tiếng Anh (`EN_ROUTE_TO_CALL`, ...). Encoder phải ánh xạ nhất quán, không so sánh chuỗi hiển thị một cách ngầm định.

---

## 5. Quy chuẩn Training và Evaluation

**MUST** tách biệt rõ hai chế độ.

### 5.1. Training

- Agent được phép học và cập nhật Policy.
- Exploration được phép sử dụng.
- Hyperparameters được phép điều chỉnh.
- Kết quả được sử dụng để phân tích quá trình học.
- **MUST** ghi lại Training Seeds và cấu hình Environment.
- **SHOULD** ghi lại Learning Curve theo episode hoặc theo số bước.

### 5.2. Evaluation

- **MUST** sử dụng Policy/Checkpoint đã huấn luyện.
- **MUST** không cập nhật kiến thức hoặc tham số học.
- **MUST** không sử dụng Exploration ngẫu nhiên nếu mục tiêu là đánh giá Policy Greedy/Deterministic.
- **MUST** ghi nhận KPI theo cùng một cách cho mọi thuật toán và cho Heuristic.
- **MUST** không sử dụng Evaluation Seeds để điều chỉnh Policy rồi báo cáo lại như dữ liệu đánh giá độc lập.

### 5.3. Quy chuẩn so sánh

**SHOULD** sử dụng:

- 10 Evaluation Seeds chung cho ba thuật toán và Heuristic.
- Cùng cấu hình Environment (`num_ambulances`, `max_time`, `max_overtime`, `poisson_lambda`, `max_pending_calls`).
- Training Seeds tách biệt với Evaluation Seeds.
- Báo cáo từng Seed và số liệu tổng hợp.

Nếu cần đánh giá độ ổn định của Training, **MAY** chạy nhiều lần huấn luyện độc lập. **MUST** ghi rõ số lần chạy và cách chọn Checkpoint.

**MUST** không lựa chọn riêng những Seed có kết quả tốt để đại diện cho toàn bộ thuật toán.

**Lưu ý tái lập [Verified]:** Environment có hai bộ sinh độc lập `env.rng` và `city_map.rng`. `reset(seed=x)` đặt lại cả hai; `reset()` không seed ở các episode sau tiếp tục luồng RNG và sinh seed traffic mới từ `env.rng`. Cùng seed **không đảm bảo** hai policy khác nhau tiêu thụ RNG theo cùng thứ tự. Vì vậy để so sánh công bằng phải dùng cùng tập Evaluation Seeds được kiểm soát và mô tả rõ giao thức reset. Trường `episode` trong Observation tăng qua các lần `reset()`; khi đối chiếu tái lập nên bỏ qua trường này.

---

## 6. Quy chuẩn KPI

**MUST** chia KPI thành các nhóm dưới đây và dùng **cùng định nghĩa, đơn vị và mẫu số** cho mọi thuật toán và Heuristic.

### A. Service Performance

| KPI | Định nghĩa |
|---|---|
| Served Rate | Số cuộc gọi `COMPLETED` / Tổng số cuộc gọi phát sinh |
| Failed Rate | Số cuộc gọi `FAILED` / Tổng số cuộc gọi phát sinh |
| High-Severity Served Rate | Tỷ lệ `COMPLETED` trong nhóm cuộc gọi nghiêm trọng |
| High-Severity Failed Rate | Tỷ lệ `FAILED` trong nhóm cuộc gọi nghiêm trọng |
| Average Waiting Time | Thời gian chờ trung bình |

### B. Priority Performance

| KPI | Ý nghĩa |
|---|---|
| High-Severity Response Time | Thời gian phản hồi đối với cuộc gọi nghiêm trọng |
| High-Severity Deadline Success Rate | Tỷ lệ đáp ứng Deadline của nhóm nghiêm trọng |
| Average Slack Time at Dispatch | Deadline còn lại trừ ETA tại thời điểm điều phối |
| Priority Violation | Quyết định điều phối không phù hợp với mức độ ưu tiên và tính khả thi |

**MUST** – Không được định nghĩa Priority Violation đơn giản là “chọn cuộc gọi nhẹ hơn trong khi có cuộc gọi nặng hơn”. Cần xét ETA, Deadline, trạng thái xe và tính khả thi. Định nghĩa cụ thể **MUST** được công bố trước khi so sánh và giữ nguyên cho cả ba thuật toán.

### C. Learning & Resource Performance

| KPI | Ý nghĩa |
|---|---|
| Episode Reward | Tổng Reward trong Episode |
| Reward Components | Các thành phần thưởng/phạt (`wait_penalty`, `failure_penalty`, `distance_penalty`, `completion_bonus`) |
| Total Distance | Tổng quãng đường di chuyển |
| Ambulance Utilization | Mức độ sử dụng đội xe |
| Learning Curve | Xu hướng cải thiện trong Training |
| Policy Stability | Độ ổn định giữa các lần chạy |
| Generalization | Khả năng hoạt động trên Seed chưa dùng khi Training |

### 6.1. Quy định tính KPI

- **MUST** dùng chung định nghĩa, đơn vị và mẫu số.
- **MUST** không tự thay đổi cách tính KPI giữa các thuật toán.
- **MUST** báo cáo rõ cách xử lý Episode bị Truncation.
- **MUST** – Các cuộc gọi chưa hoàn tất tại thời điểm Episode kết thúc phải được thống kê riêng là `Unresolved`, không tự động gộp vào `FAILED` hoặc `COMPLETED`.
- **MUST** – Không dùng một KPI duy nhất để kết luận thuật toán tốt hơn.

### 6.2. Ghi chú kỹ thuật khi tính KPI từ Environment hiện tại

Các điểm sau **MUST** được xử lý nhất quán và ghi rõ trong báo cáo:

1. **`COMPLETED` nghĩa là xong sơ cứu tại hiện trường**, trước khi bàn giao tại bệnh viện. Đây là semantics hiện tại; không đổi nghĩa khi tính KPI.
2. **`Unresolved` chưa có counter riêng trong Environment.** Khi Truncation, `info["backlog"]` và `env.get_backlog()` cung cấp `pending_calls`, `waiting_calls`, `at_scene_calls`. Chỉ các cuộc gọi trong ba nhóm này là `Unresolved`; **không** tính `en_route_hospital_ambulances` / `at_hospital_ambulances` vì cuộc gọi tương ứng đã `COMPLETED`. Việc cộng cả 5 nhóm của `total_backlog_count` có thể **đếm trùng** với `served_calls` – phải kiểm tra trước khi dùng làm mẫu số.
3. **`failed_calls` bao gồm cả cuộc gọi bị từ chối do `queue_overflow`** (vượt quá `max_pending_calls`), không chỉ cuộc gọi quá Deadline.
4. **Tổng số cuộc gọi phát sinh** không có sẵn dưới dạng một trường Observation. Phải được đếm nhất quán từ log/sự kiện (ví dụ `served + failed + Unresolved`, có đối soát). Không được tự suy ra bằng một hằng số giả định.
5. **KPI theo từng mức độ nghiêm trọng** (severity) không thể tái dựng đầy đủ chỉ từ `served_calls`/`failed_calls` cuối Episode, vì `active_calls` không còn chứa `COMPLETED`/`FAILED`. Evaluator **MAY** đọc read-only lịch sử `env.calls` hoặc ghi log sự kiện của riêng mình; không được sửa Environment.
6. **Response Time** nên định nghĩa từ các mốc có thật: `arrived_at_scene_time - created_time`. `deadline = created_time + failure_time`; `perfect_threshold = created_time + 0.5 * failure_time`. Phải công bố chính xác mốc dùng (tới hiện trường hay xong sơ cứu).
7. **Average Slack Time at Dispatch** cần ETA tại thời điểm điều phối; Environment không trả trực tiếp “slack” đã tính sẵn. Phải công bố công thức (ví dụ `deadline - (current_minute + ETA_to_scene)`) và cách lấy ETA.
8. **Ambulance Utilization** phải được định nghĩa tường minh (ví dụ tỷ lệ thời gian xe không `IDLE` trên tổng thời gian Episode) và áp dụng như nhau cho mọi thuật toán.
9. **Total Distance** có thể lấy từ `total_distance_traveled` của từng xe trong Observation; phải nêu rõ tổng trên toàn đội hay trung bình mỗi xe.
10. **Reward Components** lấy từ `info["reward_breakdown"]` gồm `wait_penalty`, `failure_penalty`, `distance_penalty`, `completion_bonus` và `total`.

---

## 7. Correctness Gate

Trước khi đánh giá hiệu suất, thuật toán **MUST** được kiểm tra tính đúng đắn:

1. State Encoder xử lý Observation đúng (không giả định trường không tồn tại, không dùng raw ID ngoài ý muốn).
2. Action Selection tuân thủ Action Mask.
3. Learning Update đúng với cơ chế thuật toán.
4. Reward được nhận và ghi nhận nhất quán.
5. Evaluation không cập nhật Policy.
6. Không có hành vi sửa đổi Environment trái quy định.
7. Có thể tái lập kết quả với cấu hình và Seed đã ghi nhận.

Nếu chưa đạt Correctness Gate, **MUST NOT** kết luận thuật toán có hiệu suất tốt hoặc kém dựa trên kết quả đó.

**SHOULD** – Mỗi thành viên nộp kèm bằng chứng kiểm tra (log, test nhỏ, script kiểm tra mask, so sánh hai lần chạy cùng seed). Không chỉ mô tả bằng lời.

---

## 8. Tiêu chí đánh giá thành công

**MUST NOT** đặt ngưỡng Served Rate, Failed Rate hoặc Episode Reward tùy ý khi chưa có dữ liệu Baseline đủ tin cậy.

Đánh giá theo ba mức:

- **P0 – Correctness:** Thuật toán hoạt động đúng và kết quả có thể tin cậy.
- **P1 – Service Performance:** Cải thiện hiệu quả phục vụ so với Heuristic, đặc biệt với các cuộc gọi nghiêm trọng.
- **P2 – Stability & Efficiency:** Độ ổn định, khả năng tổng quát hóa và sử dụng tài nguyên.

Một thuật toán có triển vọng **MUST** vượt qua P0 và thể hiện lợi ích thực tế ở P1. P2 được dùng để đánh giá chất lượng bổ sung.

**Reward dương không phải điều kiện bắt buộc để thuật toán thành công. Reward âm cũng không tự động đồng nghĩa thuật toán thất bại.** Cần phân tích các thành phần Reward và KPI phục vụ.

---

## 9. Quy chuẩn báo cáo

Mỗi thành viên **MUST** nộp báo cáo theo cấu trúc chung:

1. Algorithm Overview.
2. Implementation Architecture.
3. State Representation / State Encoder.
4. Training Configuration.
5. Correctness Verification.
6. Evaluation Results.
7. Heuristic Baseline Comparison.
8. Problems and Root Cause Analysis.
9. Improvement Proposals.
10. Conclusion.

Báo cáo Evaluation **MUST** có:

- Mean.
- Standard Deviation.
- Min / Max.
- Kết quả từng Seed.
- Training Configuration.
- Checkpoint được sử dụng.
- Reward Components.
- Các KPI quan trọng.

**MUST NOT** chỉ báo cáo ảnh chụp màn hình hoặc một Episode có kết quả tốt.

---

## 10. Các giới hạn hiện tại cần lưu ý

Đây là những vấn đề đã được ghi nhận trong quá trình audit/stabilization Environment/MDP. **MUST NOT** tự ý sửa riêng trong nhánh thuật toán:

- Observation hiện tại **chưa được chứng minh có đầy đủ tính Markov** (thiếu `next_call_id`, một số cấu hình tĩnh; raw `call_id`/ambulance ID nên coi là identifier).
- State Encoder cần được đánh giá về mức độ bảo toàn thông tin.
- Invalid Action hiện có hành vi xử lý theo hợp đồng hiện tại (no-op ngoài range, partial decode/execute, không enforce mask); Agent **MUST** tránh chọn Action không hợp lệ.
- Có điểm chưa nhất quán trong việc ghi nhận penalty của cuộc gọi phát sinh tại thời điểm Reset `t=0`: penalty được cộng vào `last_step_reward`/`total_reward` nhưng `reset()` không trả reward; `step()` đầu tiên xóa `calls_failed_this_step`. Vòng huấn luyện chỉ cộng reward từ `step()` sẽ bỏ sót khoản này.
- Reward và Episode Return cần được ghi nhận nhất quán (đặc biệt cách xử lý khoản `t=0` ở trên).
- Environment đánh dấu `COMPLETED` khi hoàn tất điều trị tại hiện trường, **không phải** khi đưa bệnh nhân đến bệnh viện.
- Episode có thể kết thúc bằng Truncation; `done=True` dùng chung cho kết thúc tự nhiên và Truncation, phải đọc `info["truncated"]` để phân biệt. `info["backlog"]` chỉ có khi `truncated=True`.

Nếu cần thay đổi các vấn đề trên, **MUST** đề xuất và thống nhất ở cấp nhóm trước.

---

## 11. Phạm vi chưa triển khai

Hiện tại:

- **Chưa yêu cầu** Common Agent Interface.
- **Chưa yêu cầu** ba thành viên hợp nhất cấu trúc Agent.
- **Chưa yêu cầu** merge các nhánh thuật toán vào `main`.
- **Chưa yêu cầu** thay đổi số lượng xe cứu thương.
- **Chưa yêu cầu** thay đổi Reward hoặc Observation.
- **Chưa yêu cầu** bổ sung cơ chế mô phỏng mới.
- **Chưa yêu cầu** xây dựng Visualization/Dashboard mới.

Common Agent Interface là **đề xuất kiến trúc** có thể xem xét **sau khi** các thuật toán đã ổn định. **MUST NOT** triển khai trước thời điểm đó.

---

## 12. Lộ trình làm việc

- **G0:** Thống nhất quy chuẩn này.
- **G1:** Mỗi thành viên audit thuật toán hiện tại.
- **G2:** Chuẩn hóa Training và Evaluation.
- **G3:** Đánh giá trên cùng Evaluation Seeds và so sánh Heuristic.
- **G4:** Phân tích nguyên nhân, đề xuất cải tiến có bằng chứng.
- **G5:** Ổn định thuật toán, xem xét Common Agent Interface và tích hợp.

Trong G1–G4, ưu tiên sửa lỗi trong thuật toán của chính mình. **MUST NOT** tự ý thay đổi Environment/MDP chung.

---

## 13. Hướng dẫn dành cho AI Agent hỗ trợ từng thành viên

Khi đọc tài liệu này, AI Agent **MUST**:

- Hiểu đây là quy chuẩn chung của nhóm, không phải yêu cầu thay thế thuật toán hiện tại.
- Kiểm tra hiện trạng code trước khi đưa ra đề xuất.
- Không giả định cấu trúc hoặc tên file khi chưa kiểm tra repository.
- Không tự ý thay đổi Environment, Reward, Observation, Action Space hoặc số lượng xe.
- Phân biệt rõ lỗi triển khai, hạn chế thuật toán và hạn chế Environment.
- Đưa ra bằng chứng cho các kết luận.
- Ưu tiên sửa lỗi thực tế, tránh overengineering.
- Không triển khai Common Agent Interface trong giai đoạn hiện tại.
- Báo cáo rõ các thay đổi, kết quả kiểm thử và vấn đề còn tồn tại.

---

## 14. Điểm cần xác minh so với tài liệu hiện tại

Phần này ghi lại các điểm **khác biệt hoặc chưa đủ thông tin** giữa yêu cầu quy chuẩn, tài liệu hiện có và implementation. Đây là các điểm **cần xác minh/thống nhất ở cấp nhóm**, không phải lỗi đã sửa.

| ID | Nội dung cần xác minh | Bằng chứng / tài liệu |
|---|---|---|
| V-01 | Environment hiện **không** có `terminated`/`truncated` tách rời và không có Gymnasium spaces. Quy chuẩn nói “Điều kiện kết thúc Episode và Truncation”; trong code phải dùng `done` + `info["truncated"]`. | M2 Mục 8; M4 Mục 11; `dispatch_env/environment.py` |
| V-02 | Không có Counter `Unresolved` sẵn. Chỉ suy ra từ `info["backlog"]` / `env.get_backlog()` khi Truncation; và `total_backlog_count` gộp cả xe đang tới bệnh viện (cuộc gọi đã `COMPLETED`) nên có nguy cơ đếm trùng. | `dispatch_env/environment.py:1040-1062`; Mục 6.2 |
| V-03 | Chưa có định nghĩa chuẩn “High-Severity”. Cấu hình severity 1/2/3 với deadline 60/40/30 phút; cần chốt nhóm “nghiêm trọng” là mức 2–3 hay chỉ mức 3 và dùng thống nhất cho mọi báo cáo. | `dispatch_env/types.py:SEVERITY_CONFIG`; `BAO_CAO_MDP.md` Mục 3.4 |
| V-04 | Chưa có KPI “Priority Violation”, “Average Slack Time at Dispatch”, “Ambulance Utilization” trong Environment. Phải định nghĩa ở evaluator của nhóm, công bố trước khi so sánh. | M2 Mục 6; M4 Mục 13 |
| V-05 | “Tổng số cuộc gọi phát sinh” không phải trường Observation; phải đếm nhất quán từ log/sự kiện và đối soát với `served + failed + Unresolved`. | `README.md` Mục 3.1; Mục 6.2 |
| V-06 | Khoản penalty phát sinh trong `reset()` tại `t=0` không được trả qua API; cần chốt cách hạch toán Episode Return (cộng thủ công từ `obs["last_step_reward"]` sau reset, hoặc bỏ qua có công bố). | M2 Mục 6.5; M4 Mục 8; Mục 10 |
| V-07 | Action ID mang nghĩa theo `slot_idx`; cùng một `action_id` ở hai thời điểm có thể trỏ tới hai cuộc gọi khác nhau vì slot được sắp xếp lại theo `(created_time, call_id)`. State Encoder/Policy phải hiểu điều này. | M2 Mục 4.3; `dispatch_env/environment.py` |
| V-08 | Observation chưa được chứng minh đủ Markov; State Encoder dùng raw `call_id`/ambulance ID có thể học phụ thuộc history không quan sát được. | M2 Mục 3.4; M4 Mục 6 |
| V-09 | Mẫu số KPI theo severity không tái dựng được chỉ từ Observation cuối Episode (không chứa `COMPLETED`/`FAILED`). Cần log sự kiện hoặc đọc read-only `env.calls`. | M2 Mục 3.1; Mục 6.2 |
| V-10 | Bảng Reward mặc định đang thực thi là wait `0.5/1.0/1.5`, fail `30/60/120`, complete `50/65/70`, perfect `70/80/100`, distance `0.05`/km; bảng đề xuất cũ `1/2/4`, `80`, `100` **không** được dùng. | `dispatch_env/types.py:REWARD_CONFIG`; `BAO_CAO_MDP.md` Mục 6.1 |
| V-11 | Số test được ghi nhận là **37** (9 map + 28 dispatch) trong M3/M4; nếu nhóm thay code phải chạy lại `python -m unittest discover -s tests -v`. | M4 Mục 12 |
| V-12 | Trạng thái hiển thị tiếng Việt trong Observation khác tên enum tiếng Anh; cần ánh xạ nhất quán trong State Encoder. | `README.md` Mục 3.1; `dispatch_env/ambulance.py:to_dict()` |

---

## 15. Phụ lục – Bảng đối chiếu nhanh giao diện Environment

### 15.1. API hiện tại

```python
obs = env.reset(seed=42)
obs, reward, done, info = env.step(action)
```

- `action` có thể là `None`, `action_id` (int), một cặp `(amb_id, call_id)`, hoặc danh sách cặp.
- `step(auto_dispatch=True)` là **Heuristic baseline**, không phải RL Agent.
- `done=True` dùng chung; `info["truncated"]` phân biệt Truncation; `info["backlog"]` chỉ có khi truncated.
- `info["reward_breakdown"]` gồm `wait_penalty`, `failure_penalty`, `distance_penalty`, `completion_bonus`, `total`.

### 15.2. Khóa Observation thực tế

`episode`, `current_minute`, `traffic_time_step`, `edge_congestion_vector`, `edge_congestion`, `edge_congestion_by_id`, `edges`, `edges_by_id`, `edge_id_to_nodes`, `nodes_to_edge_id`, `call_slots`, `action_mask`, `ambulances`, `active_calls`, `pending_call_count`, `served_calls`, `failed_calls`, `perfect_calls`, `total_reward`, `last_step_reward`.

### 15.3. Điều kiện kết thúc

- **Natural termination:** `current_minute >= max_time` **và** không còn cuộc gọi hoạt động (`PENDING`, `WAITING_FOR_AMBULANCE`, `AT_SCENE`) **và** tất cả ambulance `IDLE`.
- **Truncation:** chưa natural done và `current_minute >= max_time + max_overtime` (mặc định phút 660): `done=True`, `info["truncated"]=True`, `info["backlog"]` có giá trị.
