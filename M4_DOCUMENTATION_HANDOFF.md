# M4 – DOCUMENTATION & HANDOFF

**Dự án:** Ambulance Dispatch Simulation  
**Repository:** `E:\ambulance_dispatch`  
**Branch:** `environment`  
**Ngày hoàn tất:** 11/11/2026  
**Tài liệu đầu vào:** `M1_AUDIT_SIMULATION_ENVIRONMENT.md`, `M2_MDP_SPECIFICATION.md`, `M3_IMPLEMENTATION_REPORT.md`, `README.md`, `BAO_CAO_MDP.md`, source code hiện tại.

---

## 1. Mục tiêu và phạm vi

M4 là bước **tài liệu hóa và bàn giao**, không phải bước phát triển tính năng. Mục tiêu:

- Đồng bộ `README.md`, `BAO_CAO_MDP.md` và các điểm bị M3 làm thay đổi trong `M2_MDP_SPECIFICATION.md` với source code hiện tại.
- Tạo tài liệu bàn giao thống nhất cho nhóm RL.
- Ghi rõ phần nào đã xác minh, phần nào là giả định, giới hạn và quyết định còn mở.

**Không nằm trong phạm vi M4:** sửa source code, sửa test, huấn luyện RL Agent, thêm Gymnasium wrapper, đổi reward/terminal/seed/action semantics, commit, push, merge hoặc checkout branch khác.

---

## 2. Trạng thái bàn giao

| Hạng mục | Trạng thái |
|---|---|
| Environment MDP bằng Python/numpy | **Đã có và chạy được** |
| Heuristic baseline (`auto_dispatch=True`) | **Đã có** |
| RL Agent / policy học | **Chưa có** |
| Pipeline huấn luyện/đánh giá RL | **Chưa có** |
| Gymnasium `action_space`/`observation_space` | **Chưa có** |
| Tách `terminated`/`truncated` | **Chưa có**; `done` gộp, `info["truncated"]` phân biệt |
| Test suite | **37/37 đạt** trong lần chạy M3 |
| Hiệu chuẩn bằng dữ liệu thực | **Chưa có** |

**Tài liệu M4 đã cập nhật/tạo:**

| File | Trạng thái M4 |
|---|---|
| `README.md` | Cập nhật observation (`int` counters, route snapshot), invalid action, reset/randomness, limit `t=0`, lệnh chạy, **37 test**, mục giới hạn/quyết định mở |
| `BAO_CAO_MDP.md` | Cập nhật observation/timer, reset, test 37/37, deadline, invalid action, bảng reward cũ, giới hạn `t=0`, phân biệt Environment với RL Agent |
| `M2_MDP_SPECIFICATION.md` | Cập nhật route snapshot, test count và trạng thái các mục bị M3/M4 ảnh hưởng |
| `M4_DOCUMENTATION_HANDOFF.md` | Tạo mới |

**Không thay đổi trong M4:** source code, test, commit/push/merge/checkout. Các thay đổi chưa commit có sẵn từ trước được giữ nguyên.

**Kết luận bàn giao:** nhóm RL có thể dùng Environment như một bộ mô phỏng MDP hiện có, nhưng phải tự xây vòng huấn luyện/đánh giá và chốt các quyết định ở Mục 13 trước khi so sánh chính sách.

---

## 3. Cấu hình mặc định

| Tham số | Mặc định | Ghi chú |
|---|---:|---|
| `num_ambulances` | 6 | Số xe |
| `max_time` | 360 phút | Ngừng sinh cuộc gọi mới sau mốc này |
| `max_overtime` | 300 phút | Truncation tại `max_time + max_overtime = 660` |
| `poisson_lambda` | 0.1 | Số cuộc gọi/phút theo Poisson, không phải xác suất |
| `max_pending_calls` | 3 | Số cuộc gọi `PENDING` tối đa; tương ứng 3 slot |
| `seed` | 42 | Seed mặc định |
| Số nút | 80 | Nút trung tâm 1..20, ngoại ô 21..80 |
| Tuyến hai chiều | 86 | Tương ứng 172 cung có hướng |
| `edge_id` | 1..172 | Ánh xạ 1-1 với cung có hướng |
| Số bệnh viện | 4 | Handover time lần lượt 3, 4, 5, 4 phút |
| Bước quyết định | 1 phút | Mỗi `step()` mô phỏng `[t, t+1)` |

**Severity `(failure_time, processing_time)`:** mức 1 `(60, 4)`, mức 2 `(40, 6)`, mức 3 `(30, 8)` phút.

---

## 4. Bản đồ file và trách nhiệm

| File/thư mục | Nội dung |
|---|---|
| `dispatch_env/environment.py` | `reset`, `step`, observation, slot, action mask, reward, terminal, log |
| `dispatch_env/ambulance.py` | Di chuyển, sơ cứu, chọn bệnh viện, handover, `to_dict()` |
| `dispatch_env/call.py` | Trạng thái cuộc gọi, deadline, perfect threshold, failure reason |
| `dispatch_env/types.py` | Enum, defaults, `SEVERITY_CONFIG`, `REWARD_CONFIG` |
| `map_env/city_map.py` | Đồ thị 80 nút, 4 bệnh viện, traffic reset/update, routing |
| `map_env/edge.py` | Cung có hướng, tốc độ, congestion Markov matrix |
| `run_simulation.py` | Chạy heuristic baseline và ghi `simulation.log` |
| `tests/test_map.py` | 9 test bản đồ/giao thông/bệnh viện |
| `tests/test_dispatch_env.py` | 28 test Environment, gồm 3 test M3 |
| `M1_AUDIT_SIMULATION_ENVIRONMENT.md` | Audit Environment |
| `M2_MDP_SPECIFICATION.md` | Đặc tả MDP và hợp đồng Environment–Agent |
| `M3_IMPLEMENTATION_REPORT.md` | Thay đổi, test và giới hạn của M3 |
| `M4_DOCUMENTATION_HANDOFF.md` | Tài liệu bàn giao này |

---

## 5. API Environment

### 5.1. Khởi tạo

```python
from map_env import CityMap
from dispatch_env import AmbulanceDispatchEnv

city_map = CityMap(seed=42)
env = AmbulanceDispatchEnv(
    city_map=city_map,
    num_ambulances=6,
    max_time=360,
    max_overtime=300,
    poisson_lambda=0.1,
    max_pending_calls=3,
    seed=42,
)
```

### 5.2. `reset()`

```python
obs = env.reset(seed=42)
obs = env.reset()                  # episode mới, tiếp tục luồng RNG
obs = env.reset(generate_initial_calls=False)  # không sinh cuộc gọi tại t=0
```

- `reset()` chỉ trả `obs`, **không trả reward**.
- `reset(seed=...)` đặt lại `env.rng` và traffic map.
- `reset()` không seed **sau episode đầu** lấy seed traffic mới từ `env.rng`.
- Episode đầu tiên ngay sau khởi tạo không gọi `reset_traffic()`; traffic hiện có của `CityMap` được giữ nguyên.

### 5.3. `step()`

```python
obs, reward, done, info = env.step(action)
obs, reward, done, info = env.step(action_id)          # action ID
obs, reward, done, info = env.step([(1, 10), (3, 12)]) # cặp (amb_id, call_id)
obs, reward, done, info = env.step(auto_dispatch=True) # heuristic baseline
```

- `action=None` và `auto_dispatch=False`: không phân công.
- `done=True` dùng chung cho kết thúc tự nhiên và truncation.
- `info["truncated"]` phân biệt truncation; `info["backlog"]` có khi truncated.
- `info` còn có `action_id`, `decoded_action`, `valid_action_count`, `served`, `failed`, `perfect`, `step_reward`, `total_reward`, `reward_breakdown`.

---

## 6. Observation

`get_observation()` trả dictionary gồm:

- Thời gian/episode: `episode`, `current_minute`, `traffic_time_step`.
- Giao thông: `edge_congestion_vector`, `edge_congestion`, `edge_congestion_by_id`, `edges`, `edges_by_id`, `edge_id_to_nodes`, `nodes_to_edge_id`.
- Slot/mask: `call_slots` (3 slot), `action_mask`.
- Xe: `ambulances` (`ambulance_id -> dict`).
- Cuộc gọi: `active_calls`, `pending_call_count`.
- Counter/reward: `served_calls` (**int**), `failed_calls` (**int**), `perfect_calls` (**int**), `total_reward`, `last_step_reward`.

**Ambulance fields:** `id`, `ambulance_id`, `status`, `current_edge`, `current_edge_id`, `distance_from_start`, `coord`, `current_speed`, `road_speed`, `current_call_id`, `target_type`, `targer_type` (alias), `target_id`, `target_edge`, `target_edge_id`, `target_dist`, `target_name`, `assigned_hospital_id`, `hospital_handover_time`, `scene_time_remaining`, `hospital_time_remaining`, `route`, `route_edge_ids`, `total_distance_traveled`.

**Call fields:** `id`, `call_id`, `severity`, `edge`, `edge_id`, `dist`, `distance_from_start`, `coord`, `status`, `created_time`, `failure_time`, `processing_time`, `assigned_ambulance_id`, `arrived_at_scene_time`, `completed_time`, `wait_time`, `deadline`, `perfect_threshold`, `is_perfect`. `call_slots` thêm `slot`.

**[Verified]** Từ M3, `route` là bản copy `list(self.route_edges)` tại `dispatch_env/ambulance.py:130`; observation cũ không bị thay đổi khi Environment tiếp tục `step()`. Có test `test_observation_route_is_snapshot`.

**[Open decision]** Observation chưa được chứng minh là sufficient statistic/Markov đầy đủ: không có `next_call_id`, không có toàn bộ cấu hình tĩnh. Raw `call_id` và ambulance ID nên được xem là identifier, không mặc định là policy feature.

---

## 7. Không gian hành động và action mask

Với `N` xe và `M` slot:

\[
|\mathcal A_{\text{potential}}|=\sum_{k=0}^{\min(N,M)}\binom Nk\binom Mk k!
\]

| Cấu hình | Số action tiềm năng |
|---|---:|
| 6 xe, 3 slot | **229** |
| 4 xe, 3 slot | **73** |

- Action ID `0` là không phân công `()`, luôn hợp lệ.
- `get_call_slots()` sắp xếp `PENDING` theo `(created_time, call_id)`.
- `get_action_mask()` đánh dấu action hợp lệ tại trạng thái hiện tại.
- `encode_action()` / `decode_action()` chuyển đổi giữa cặp `(amb_id, call_id)` và action ID.

**[Verified] Hợp đồng invalid action hiện tại:**

- `action_id` ngoài `[0, num_potential_actions - 1]` → `[]` (no-op).
- `action_id` trong range nhưng mask `False` vẫn được decode từng cặp; cặp sai bị lọc, cặp còn lại có thể thực thi (**partial execution**).
- `step()` **không enforce** action mask và **không raise** cho các trường hợp trên.

**[Open decision]** Cần chốt strict raise / no-op toàn action / partial có cảnh báo. Trong thí nghiệm hiện tại nên luôn dùng action mask.

---

## 8. Reward

Công thức một bước:

\[
r_t=-\sum_c w_c\Delta W_{c,t}
    -\sum_{c\in F_t}p_c
    -0.05\sum_i\Delta d_{i,t}
    +\sum_{c\in C_t}b_c
\]

| Mức | Wait penalty/phút | Fail penalty/lần | Complete reward | Perfect reward |
|---|---:|---:|---:|---:|
| 1 | 0.5 | 30 | 50 | 70 |
| 2 | 1.0 | 60 | 65 | 80 |
| 3 | 1.5 | 120 | 70 | 100 |

- Distance penalty: `0.05` điểm/km.
- Perfect reward **thay thế** complete reward, không cộng thêm.
- Completion reward được cộng đúng một lần nhờ `completion_rewarded`; có test `test_completion_rewarded_only_once`.
- `COMPLETED` nghĩa là **đã xong sơ cứu tại hiện trường**, trước khi bàn giao tại bệnh viện.
- Perfect nếu xong sơ cứu không muộn hơn `created_time + 0.5 * failure_time`.

**[Limitation/Open decision]** Nếu `reset()` từ chối cuộc gọi do đầy hàng chờ tại `t=0`, penalty được cộng vào `last_step_reward`/`total_reward` nhưng `reset()` không trả reward; `step()` đầu tiên xóa `calls_failed_this_step`. Vòng huấn luyện chỉ cộng reward từ `step()` sẽ bỏ sót khoản này.

---

## 9. Quy luật chuyển trạng thái

Mỗi `step()` mô phỏng một phút `[t, t+1)` theo thứ tự:

1. Lấy slot và action mask tại `t`.
2. Decode và thực thi phân công đồng thời.
3. Xe di chuyển/xử lý liên tục trong `[t, t+1)`.
4. Cập nhật wait penalty, kiểm tra timeout, xử lý xe đang tiếp cận khi call fail.
5. Ghi nhận completion và reward completion một lần.
6. Tăng `current_minute`, cập nhật traffic Markov.
7. Reroute xe đang di chuyển bằng traffic mới.
8. Sinh cuộc gọi mới tại `t+1` nếu `current_minute < max_time`.
9. Tính reward và kiểm tra `done`/truncation.

**Deadline [Verified]:** đến đúng deadline là thành công; chỉ thất bại khi `current_time > deadline + 1e-6`. Khi xe đã tới hiện trường (`AT_SCENE`), deadline không còn áp dụng.

**Traffic [Verified]:** movement trong `[t, t+1)` dùng traffic trước cập nhật; reroute sau cập nhật dùng traffic mới. Chu kỳ cao điểm: `10 <= (traffic_time_step mod 40) <= 25`.

**Call arrival [Verified]:** tại `t=0` nếu `generate_initial_calls=True`, và tại `t+1` cuối mỗi step khi chưa đạt `max_time`. Cuộc gọi vượt quá 3 slot `PENDING` bị từ chối ngay và tính thất bại.

**Hospital [Verified]:** agent không chọn bệnh viện hay tuyến; code tự chọn bệnh viện có ETA nhỏ nhất sau khi sơ cứu xong.

---

## 10. Reset, randomness và tái lập

- Có **hai bộ sinh độc lập**: `env.rng` và `city_map.rng`.
- `reset(seed=42)` reset cả hai; cùng seed cho phép tái lập kịch bản ban đầu.
- `reset()` không seed ở các episode sau tiếp tục luồng RNG và sinh seed traffic mới từ `env.rng`.
- Episode đầu tiên sau khởi tạo giữ traffic hiện có của `CityMap`, không gọi `reset_traffic()`.
- Vị trí xe ban đầu ngẫu nhiên trên các cung không lặp; xe luôn bắt đầu `IDLE`.
- Traffic ban đầu khi được reset: mức 1/2/3 với xác suất `0.60/0.30/0.10`.
- `episode` trong observation tăng qua các lần `reset()`; khi so sánh tái lập nên bỏ qua trường này.
- Cùng seed không đảm bảo hai policy khác nhau tiêu thụ RNG theo cùng thứ tự. Để so sánh công bằng, dùng cùng tập kịch bản/seed đánh giá được kiểm soát.

---

## 11. Terminal và truncation

Episode kết thúc tự nhiên khi đồng thời:

1. `current_minute >= max_time`.
2. Không còn cuộc gọi hoạt động (`PENDING`, `WAITING_FOR_AMBULANCE`, `AT_SCENE`).
3. Tất cả xe `IDLE`.

Nếu chưa kết thúc tự nhiên và `current_minute >= max_time + max_overtime` (mặc định 660), episode dừng với:

```python
done = True
info["truncated"] = True
info["backlog"] = env.get_backlog()
```

**[Verified]** `done=True` không phân biệt natural termination và truncation; phải đọc `info["truncated"]`.

---

## 12. Test và xác minh

Lệnh chạy:

```powershell
python -m unittest discover -s tests -v
```

- Số test hiện tại: **37** (9 trong `tests/test_map.py`, 28 trong `tests/test_dispatch_env.py`).
- Lần chạy M3: **37/37 đạt** trong khoảng 0.599 giây.
- 3 test M3:
  - `test_observation_route_is_snapshot`
  - `test_invalid_action_contract_current_behavior`
  - `test_completion_rewarded_only_once`
- M4 chỉ kiểm tra sự hiện diện của 37 test method và đối chiếu source; **không sửa code/test, không chạy lại toàn bộ suite**.

**[Assumption]** 37/37 vẫn đúng sau M4 vì M4 không thay đổi source hoặc test. Nếu nhóm thay code, phải chạy lại toàn bộ suite.

---

## 13. Giới hạn và quyết định còn mở

| ID | Nội dung | Nhãn | Trạng thái |
|---|---|---|---|
| H-01 | Invalid action: no-op ngoài range, partial decode/execute, không enforce mask | **Open decision** | Code hiện tại đã có test contract; chưa chốt strict/no-op/partial |
| H-02 | Penalty `t=0` trong `reset()` không trả qua API training | **Limitation/Open decision** | Cần chốt hạch toán |
| H-03 | Observation chưa chứng minh đủ Markov; thiếu `next_call_id`, cấu hình tĩnh | **Limitation/Open decision** | Cần chốt policy feature, đặc biệt raw ID |
| H-04 | `COMPLETED` tại hiện trường, không phải handover | **Verified/Limitation** | Không đổi KPI semantics nếu chưa quyết định |
| H-05 | Chưa có Gymnasium spaces/wrapper, chưa tách `terminated`/`truncated` | **Open decision** | Chỉ làm nếu nhóm chọn tích hợp thư viện RL |
| H-06 | Chưa có RL Agent, training pipeline, baseline nhiều seed | **Limitation** | Nhóm RL chịu trách nhiệm |
| H-07 | Tham số mô phỏng chưa hiệu chuẩn bằng dữ liệu thực | **Limitation** | Không dùng kết quả để kết luận y tế thực tế |
| H-08 | Reward/terminal/seed/event ordering | **Verified** | Giữ nguyên; thay đổi phải version hóa và chạy lại test |

**Quyết định nhóm cần chốt trước khi huấn luyện nghiêm túc:**

1. Hợp đồng invalid action.
2. Cách hạch toán penalty `t=0`.
3. Policy có dùng raw `call_id`/ambulance ID hay không; có cần thêm field vào observation.
4. Completion KPI dùng thời điểm xong sơ cứu hay handover (hoặc báo cáo cả hai).
5. Có cần Gymnasium wrapper và tách `terminated`/`truncated`.
6. Giá trị `gamma` và giao thức đánh giá nhiều seed.

---

## 14. Hướng dẫn sử dụng, trách nhiệm nhóm RL và acceptance

### 14.1. Vòng lặp mẫu

```python
obs = env.reset(seed=42)
done = False

while not done:
    mask = obs["action_mask"]
    action = 0  # thay bằng policy; phải bảo đảm mask[action] nếu muốn hợp lệ
    obs, reward, done, info = env.step(action)

    if info["truncated"]:
        print("Kết thúc do truncation, backlog:", info["backlog"])
```

Với heuristic baseline:

```python
obs, reward, done, info = env.step(auto_dispatch=True)
```

### 14.2. Trách nhiệm nhóm RL

- Xây vòng huấn luyện/đánh giá và chọn thuật toán, `gamma`, hyperparameter.
- Quyết định dùng action mask như một phần của policy hay môi trường.
- Chốt các quyết định H-01, H-02, H-03, H-05 tại Mục 13.
- Thiết kế tập seed/kịch bản đánh giá công bằng; không chỉ báo cáo reward mà còn KPI vận hành.
- Không sửa semantics Environment tùy tiện; nếu sửa, cập nhật test và version tài liệu.

### 14.3. Checklist bàn giao

- [x] README đồng bộ observation, action, reward, baseline và **37 test**.
- [x] `BAO_CAO_MDP.md` đồng bộ observation/timer, test, reset, deadline, invalid action và bảng reward cũ.
- [x] `M2_MDP_SPECIFICATION.md` cập nhật route snapshot, test count và trạng thái các quyết định bị M3 ảnh hưởng.
- [x] Ghi rõ M3 route snapshot và 3 test mới.
- [x] Ghi rõ Environment khác RL Agent/training.
- [x] Ghi rõ các giới hạn và quyết định còn mở.
- [ ] Nhóm chốt invalid action contract.
- [ ] Nhóm chốt hạch toán penalty `t=0`.
- [ ] Nhóm chốt Markov/ID policy và Gymnasium wrapper nếu cần.

### 14.4. Tiêu chí acceptance của M4

| Tiêu chí | Kết quả |
|---|---|
| Không sửa source/test | Đạt |
| Không commit/push/merge/checkout | Đạt |
| Tài liệu khớp source hiện tại | Đạt |
| Số test được đồng bộ về 37 | Đạt |
| Có hướng dẫn sử dụng và trách nhiệm nhóm RL | Đạt |
| Có danh sách giới hạn/quyết định mở | Đạt |
| Có thể dùng làm điểm bắt đầu cho M5/nhóm RL | Đạt |
