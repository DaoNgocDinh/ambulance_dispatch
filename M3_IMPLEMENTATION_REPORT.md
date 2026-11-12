# M3 – IMPLEMENTATION REPORT

**Dự án:** Ambulance Dispatch Simulation  
**Repository:** `E:\ambulance_dispatch`  
**Branch:** `environment`  
**Tài liệu đầu vào:** `M1_AUDIT_SIMULATION_ENVIRONMENT.md`, `M2_MDP_SPECIFICATION.md`, source code hiện tại  
**Phạm vi:** Ổn định và xác minh Environment hiện có; không mở rộng MDP, không xây RL Agent, không thêm Gymnasium wrapper.

---

## 1. Executive Summary

M3 đã đọc lại M1, M2 và source code hiện tại trước khi thay đổi. Kết quả chính:

- Xác định và sửa **một MUST FIX thực sự**: `Ambulance.to_dict()` trả về `route` bằng tham chiếu tới list nội bộ `route_edges`, khiến observation cũ bị thay đổi khi Environment tiếp tục `step()`.
- **Không thay đổi** semantics invalid action, reward, completion, terminal, seed hay event ordering vì chưa có bằng chứng đây là bug.
- Làm rõ contract invalid action hiện tại trong docstring và thêm test hồi quy.
- Thêm 3 test mục tiêu: observation route snapshot, invalid action contract, completion reward chỉ cộng một lần.
- Chạy test mục tiêu: **3/3 pass**.
- Chạy lại toàn bộ test suite một lần: **37/37 pass** trong khoảng 0.599 giây.
- Không commit, push, merge, checkout branch khác, không cài dependency, không sửa M1/M2.

Thay đổi hành vi duy nhất là defensive copy cho `route`; các thay đổi còn lại chỉ là tài liệu trong docstring và test.

---

## 2. Phân loại vấn đề

| ID | Vấn đề | Bằng chứng | Phân loại | Xử lý M3 |
|---|---|---|---|---|
| M3-01 | Observation `route` là mutable reference tới `Ambulance.route_edges` | `dispatch_env/ambulance.py` dòng 130 trước sửa; tái hiện được bằng `env.step()` sau khi lấy observation | **MUST FIX** | Đã sửa bằng `list(self.route_edges)` |
| M3-02 | Invalid action: ngoài range trả `[]`; action mask `False` có thể partial decode; `step()` không enforce mask | `decode_action()`, `is_action_valid()`, `step()` | **CLARIFY ONLY** | Giữ hành vi; bổ sung docstring và test contract |
| M3-03 | Penalty phát sinh trong `reset()` tại `t=0` không được trả qua `step()` đầu tiên | `reset()` cập nhật `last_step_reward`/`total_reward`; `reset()` không trả reward; `step()` clear `calls_failed_this_step` | **CLARIFY ONLY** | Không đổi API; ghi rõ trong báo cáo |
| M3-04 | Chưa chứng minh observation là sufficient statistic / Markov đầy đủ | M2 mục 3.4; thiếu `next_call_id`, cấu hình tĩnh, raw ID policy | **CLARIFY ONLY / DEFER** | Không mở rộng observation |
| M3-05 | Reward table và completion KPI | `REWARD_CONFIG`; `mark_completed()` khi xong sơ cứu | **CLARIFY ONLY** | Giữ nguyên; test suite hiện có xác nhận |
| M3-06 | Lifecycle, deadline, event ordering, terminal, seed | M1/M2 và test hiện có | **CLARIFY ONLY** | Giữ nguyên; không thấy bug thực tế |
| M3-07 | Strict rejection / enforce action mask | Chưa có quyết định contract | **DEFER** | Không tự thay đổi semantics |
| M3-08 | Trả penalty `t=0` qua API khác (info/reset reward/step đầu) | Chưa có yêu cầu chốt | **DEFER** | Không đổi chữ ký `reset()` |
| M3-09 | Gymnasium wrapper, `terminated`/`truncated` tách rời | M2 D-05 | **DEFER** | Ngoài scope M3 |
| M3-10 | Tài liệu README/BAO_CAO còn mô tả cũ | M2 mục 1.3 | **DEFER** | Chuyển sang M4 để tránh mở rộng M3 |
| M3-11 | `next_call_id`, cấu hình bệnh viện/`max_time` không có trong observation | M2 D-13, D-14 | **DEFER** | Chưa có bằng chứng cần cho policy hiện tại |
| M3-12 | `edges` và `edges_by_id` dùng chung dict trong cùng một observation | `get_observation()` tạo một `e_info` rồi gán cho hai khóa | **DEFER** | Không phải internal state leak; không sửa khi chưa có ảnh hưởng thực tế |

---

## 3. Thay đổi thực tế đã triển khai

### 3.1. M3-C1 – Defensive copy cho `route` trong observation

**Trước khi sửa:** `Ambulance.to_dict()` trả `"route": self.route_edges`, tức cùng object list nội bộ.

**Sau khi sửa:** trả `"route": list(self.route_edges)` tại `dispatch_env/ambulance.py:130`.

**Lý do:** Khi xe di chuyển qua cung trung gian, `Ambulance.step_continuous()` gọi `self.route_edges.pop(0)`. Observation cũ giữ cùng list nên bị thay đổi ngoài ý muốn. Đã tái hiện trước khi sửa: observation ban đầu `[(1, 18), (18, 2)]` trở thành `[(18, 2)]` sau một `step()`.

**Phạm vi:** Chỉ copy list; cấu trúc observation và các trường khác không đổi.

**Test xác nhận:** `test_observation_route_is_snapshot` – kiểm tra route snapshot không đổi sau `env.step()` và không phải cùng object với `amb.route_edges`.

### 3.2. M3-C2 – Làm rõ invalid action contract trong docstring

**Trước khi sửa:** `decode_action()` và `step()` chưa mô tả rõ no-op ngoài range và partial decode khi mask `False`.

**Sau khi sửa:** bổ sung mô tả tại:

- `dispatch_env/environment.py:397` – `decode_action()`: action ID ngoài range trả `[]`; action trong range nhưng mask `False` vẫn decode từng cặp; cặp không hợp lệ bị bỏ qua; không raise.
- `dispatch_env/environment.py:692` – `step()`: không enforce action mask; không raise invalid action; có thể partial execution.

**Lý do:** M3 yêu cầu làm rõ contract trước khi có quyết định strict/no-op/partial. Chưa có bằng chứng hành vi hiện tại là bug, nên không thay đổi dispatch semantics.

**Test xác nhận:** `test_invalid_action_contract_current_behavior` – kiểm tra action ngoài range, action mask `False`, decode partial và partial execution thực tế.

### 3.3. M3-C3 – Ghi rõ observation là snapshot trong docstring

**Sau khi sửa:** `get_observation()` mô tả tại `dispatch_env/environment.py:516` rằng `route` được copy để observation cũ không bị thay đổi khi tiếp tục `step()`.

**Lý do:** Tài liệu hóa hành vi mới của M3-C1, không thay đổi thêm code.

### 3.4. M3-C4 – Bổ sung test mục tiêu

Đã thêm 3 test vào `tests/test_dispatch_env.py`:

1. `test_observation_route_is_snapshot` – dòng 818.
2. `test_invalid_action_contract_current_behavior` – dòng 853.
3. `test_completion_rewarded_only_once` – dòng 877.

Không thêm test trùng lặp cho deadline, truncation hay reset seed vì các hành vi này đã có test hiện hữu.

---

## 4. Danh sách file thay đổi và lý do

| File | Trạng thái | Thay đổi | Lý do |
|---|---|---|---|
| `dispatch_env/ambulance.py` | Modified | `route` trả về bản copy | MUST FIX observation immutability |
| `dispatch_env/environment.py` | Modified | Cập nhật docstring `decode_action()`, `step()`, `get_observation()` | Làm rõ contract, không đổi hành vi |
| `tests/test_dispatch_env.py` | Modified | Thêm 3 test mục tiêu | Xác nhận fix và contract hiện tại |
| `M3_IMPLEMENTATION_REPORT.md` | New | Báo cáo M3 | Yêu cầu bàn giao M3 |

Các file `M1_AUDIT_SIMULATION_ENVIRONMENT.md`, `M2_MDP_SPECIFICATION.md` và các thay đổi chưa commit có sẵn không bị ghi đè.

---

## 5. Test đã chạy và kết quả

### 5.1. Test mục tiêu

Lệnh đã chạy:

```powershell
$env:PYTHONPATH='E:\ambulance_dispatch'
& "C:\Users\DELL\AppData\Local\Programs\Python\Python310\python.exe" -m unittest -v `
  tests.test_dispatch_env.TestDispatchEnv.test_observation_route_is_snapshot `
  tests.test_dispatch_env.TestDispatchEnv.test_invalid_action_contract_current_behavior `
  tests.test_dispatch_env.TestDispatchEnv.test_completion_rewarded_only_once
```

Kết quả: **3/3 pass**, không lỗi.

### 5.2. Toàn bộ test suite

Lệnh đã chạy một lần sau thay đổi:

```powershell
$env:PYTHONPATH='E:\ambulance_dispatch'
& "C:\Users\DELL\AppData\Local\Programs\Python\Python310\python.exe" -m unittest discover -s tests -v
```

Kết quả: **37/37 pass**, thời gian khoảng **0.599 giây**, không có failure/error.

Số test tăng từ 34 lên 37 do 3 test mới. Đây là kết quả thực tế; tài liệu README vẫn ghi 34 và `BAO_CAO_MDP.md` ghi 31, cần đồng bộ ở M4.

---

## 6. Những vấn đề được giữ nguyên và lý do

1. **Invalid action semantics:** giữ no-op ngoài range và partial decode. Chưa có bằng chứng đây là bug; M3 ưu tiên không thay đổi hành vi Agent đang dùng. Contract đã được làm rõ và có test.
2. **Action mask không được enforce trong `step()`:** giữ nguyên. Đây là quyết định thiết kế/contract, không phải lỗi thực thi đã chứng minh.
3. **Penalty `t=0` trong `reset()`:** giữ nguyên chữ ký `reset()` và cách ghi nhận vào `last_step_reward`/`total_reward`. Không trả reward từ `reset()`, nên tổng reward từ các `step()` không bao gồm khoản penalty này. Đây là hạn chế đã biết, cần nhóm quyết định cách hạch toán.
4. **Reward table, perfect reward, completion tại hiện trường:** giữ nguyên đúng bảng hiện hành và định nghĩa completion hiện tại. Không đổi KPI sang hospital handover.
5. **Event ordering, traffic update, reroute, call generation, terminal/truncation, seed:** giữ nguyên. Test suite hiện có và kiểm tra source không cho thấy bug thực tế trong phạm vi M3.
6. **Tài liệu README/BAO_CAO lỗi thời:** không sửa trong M3 để tránh mở rộng phạm vi; chuyển thành đầu vào M4.
7. **Gymnasium wrapper, `terminated`/`truncated`, RL Agent/training:** ngoài scope M3.

---

## 7. Giới hạn và quyết định còn chờ nhóm

- **Invalid action contract cuối cùng:** chọn strict raise, no-op toàn action, partial có cảnh báo, hay chỉ dựa vào mask. Hiện tại code là partial decode/no-op ngoài range.
- **Hạch toán penalty `t=0`:** quyết định có đưa vào return training, `info` của `reset()`, cộng vào step đầu, hay bỏ penalty `t=0`.
- **Markov sufficiency:** chưa chứng minh observation đầy đủ cho mọi policy. Cần chốt cấu hình tĩnh là θ và policy có được dùng raw `call_id` hay không.
- **Đồng bộ tài liệu:** README, `BAO_CAO_MDP.md` và số lượng test cần được cập nhật ở M4.
- **API RL:** chưa có Gymnasium wrapper hoặc tách `terminated`/`truncated`; chỉ làm nếu nhóm chọn hướng tích hợp thư viện RL.

---

## 8. Đánh giá mức độ hoàn thành M3

| Tiêu chí acceptance | Kết quả |
|---|---|
| Không mở rộng ngoài phạm vi | Đạt |
| Bug thực sự được sửa bằng thay đổi tối thiểu | Đạt – chỉ copy `route` |
| Observation không bị mutable reference làm thay đổi ngoài ý muốn | Đạt – có test hồi quy |
| Contract quan trọng được làm rõ | Đạt – invalid action, route snapshot, t=0 penalty trong báo cáo |
| Test mục tiêu và test suite được chạy, báo cáo trung thực | Đạt – 3/3 và 37/37 pass |
| Không làm mất thay đổi có sẵn | Đạt – branch và các file modified/untracked được giữ nguyên |
| Có `M3_IMPLEMENTATION_REPORT.md` đầy đủ | Đạt |

**Kết luận:** M3 đạt mục tiêu ổn định và xác minh Environment trong phạm vi cho phép. Một MUST FIX đã được xử lý; các quyết định contract/API còn lại được giữ nguyên có chủ đích và chuyển thành đầu vào cho nhóm/M4.

---

## 9. Đề xuất đầu vào cho M4 – Documentation & Handoff

1. Cập nhật README:
   - `served_calls`, `failed_calls`, `perfect_calls` là `int`, không phải list.
   - Số test thực tế là 37.
   - Ghi rõ invalid action contract hiện tại: no-op ngoài range, partial decode, mask không enforce trong `step()`.
   - Ghi rõ `route` trong observation là snapshot sau M3.
2. Cập nhật `BAO_CAO_MDP.md`:
   - Xóa/điều chỉnh mô tả lỗi thời về observation thiếu timer/target fields.
   - Đồng bộ số test và bảng reward; đánh dấu bảng đề xuất cũ `1/2/4` là chưa áp dụng.
   - Ghi rõ completion KPI đang tính tại thời điểm xong sơ cứu.
3. Chốt và ghi tài liệu cho các quyết định còn mở:
   - Invalid action contract.
   - Hạch toán penalty `t=0`.
   - Markov sufficiency và policy ID-invariant.
   - Gymnasium wrapper nếu cần.
4. Đưa M3-C1 và 3 test mới vào checklist handoff để nhóm RL biết observation cũ an toàn khi lưu giữ qua nhiều bước.
5. Không cần thay đổi thêm core Environment trừ khi nhóm chốt thay đổi contract/reward/API.

---

## Git status tại thời điểm kết thúc M3

- Branch: `environment`.
- Không commit, push, merge hoặc checkout branch khác.
- Các thay đổi chưa commit có sẵn vẫn được giữ nguyên.
- File modified bởi M3: `dispatch_env/ambulance.py`, `dispatch_env/environment.py`, `tests/test_dispatch_env.py`.
- File mới của M3: `M3_IMPLEMENTATION_REPORT.md`.
- File M1/M2 vẫn là untracked và không bị sửa.
