"""
Bộ kiểm thử tự động (Unit Tests) cho Môi trường Điều phối Xe Cứu Thương (dispatch_env).
"""

from __future__ import annotations

import unittest
from map_env.city_map import CityMap
from dispatch_env.types import (
    AmbulanceStatus,
    CallStatus,
    DEFAULT_NUM_AMBULANCES,
    DEFAULT_MAX_TIME,
    DEFAULT_MAX_OVERTIME,
    DEFAULT_POISSON_LAMBDA,
    DEFAULT_MAX_PENDING_CALLS,
    SEVERITY_CONFIG,
    REWARD_CONFIG,
    DISTANCE_PENALTY_PER_KM,
)
from dispatch_env.call import EmergencyCall
from dispatch_env.ambulance import Ambulance
from dispatch_env.environment import AmbulanceDispatchEnv


class TestDispatchEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.env = AmbulanceDispatchEnv(seed=123)

    def test_initial_ambulance_state(self) -> None:
        """Kiểm tra 6 xe cứu thương (mặc định đồng bộ với run_simulation), vị trí ngẫu nhiên, trạng thái luôn Rảnh (IDLE)."""
        self.assertEqual(self.env.num_ambulances, DEFAULT_NUM_AMBULANCES)
        self.assertEqual(self.env.max_time, DEFAULT_MAX_TIME)
        self.assertEqual(self.env.max_overtime, DEFAULT_MAX_OVERTIME)
        self.assertEqual(self.env.poisson_lambda, DEFAULT_POISSON_LAMBDA)
        self.assertEqual(self.env.max_pending_calls, DEFAULT_MAX_PENDING_CALLS)
        self.assertEqual(len(self.env.ambulances), DEFAULT_NUM_AMBULANCES)
        for aid, amb in self.env.ambulances.items():
            self.assertEqual(amb.status, AmbulanceStatus.IDLE)
            self.assertIsNone(amb.current_call_id)
            self.assertIn(amb.current_edge, self.env.city_map.edges)
            self.assertGreaterEqual(amb.distance_from_start, 0.0)

    def test_call_severity_and_failure_deadlines(self) -> None:
        """Kiểm tra cấu hình mức độ nghiêm trọng (1, 2, 3), thời gian thất bại và xử lý."""
        for sev, (fail_t, proc_t) in SEVERITY_CONFIG.items():
            call = EmergencyCall(
                call_id=sev,
                edge=(1, 2),
                distance_from_start=0.5,
                coord=(10.0, 10.0),
                severity=sev,
                created_time=0.0,
            )
            self.assertEqual(call.failure_time, fail_t)
            self.assertEqual(call.processing_time, proc_t)
            self.assertEqual(call.deadline, fail_t)
            self.assertEqual(call.perfect_threshold, 0.5 * fail_t)

    def test_call_failure_timeout(self) -> None:
        """Kiểm tra cuộc gọi thất bại khi quá thời gian thất bại."""
        call = EmergencyCall(
            call_id=10,
            edge=(1, 2),
            distance_from_start=0.5,
            coord=(10.0, 10.0),
            severity=3,  # Mức 3: failure_time = 30
            created_time=5.0,
        )
        # deadline = 5.0 + 30.0 = 35.0
        # Xe đến đúng deadline (current_time == 35.0) tính là thành công, không tính là thất bại
        self.assertFalse(call.check_failure(current_time=35.0))
        # Khi vượt quá deadline (current_time > 35.0) mới tính là thất bại
        self.assertTrue(call.check_failure(current_time=35.1))
        self.assertEqual(call.status, CallStatus.FAILED)

    def test_exact_deadline_arrival_is_success(self) -> None:
        """Kiểm tra: Xe đến đúng deadline (chuẩn thời gian thất bại) tính là thành công."""
        call = EmergencyCall(
            call_id=11,
            edge=(1, 2),
            distance_from_start=0.5,
            coord=(10.0, 10.0),
            severity=3,  # failure_time = 30
            created_time=0.0,
        )
        self.assertEqual(call.deadline, 30.0)
        # Tại đúng mốc 30.0
        self.assertFalse(call.check_failure(current_time=30.0))
        call.mark_at_scene(arrival_time=30.0)
        self.assertEqual(call.status, CallStatus.AT_SCENE)
        # Sau khi đã ở hiện trường, không thể thất bại
        self.assertFalse(call.check_failure(current_time=35.0))

    def test_scene_arrival_removes_failure_deadline(self) -> None:
        """Khi xe tới hiện trường (AT_SCENE), mốc thời gian thất bại bị loại bỏ."""
        call = EmergencyCall(
            call_id=20,
            edge=(1, 2),
            distance_from_start=0.5,
            coord=(10.0, 10.0),
            severity=1,
            created_time=0.0,
        )
        call.mark_at_scene(arrival_time=15.0)
        self.assertEqual(call.status, CallStatus.AT_SCENE)
        # Quá mốc 30 phút nhưng vì đã ở hiện trường nên không thể bị FAILED do timeout
        self.assertFalse(call.check_failure(current_time=50.0))
        self.assertEqual(call.status, CallStatus.AT_SCENE)

    def test_max_3_pending_calls_limit(self) -> None:
        """Chỉ cho phép tối đa 3 cuộc gọi ở trạng thái PENDING, các cuộc gọi khác bị từ chối (FAILED)."""
        env = AmbulanceDispatchEnv(seed=999)
        env.reset(generate_initial_calls=False)

        # Tạo thủ công 3 cuộc gọi PENDING
        for cid in range(1, 4):
            c = EmergencyCall(
                call_id=cid,
                edge=(1, 2),
                distance_from_start=0.5,
                coord=(10.0, 10.0),
                severity=1,
                created_time=0.0,
                status=CallStatus.PENDING,
            )
            env.calls[cid] = c
            env.next_call_id = 4

        self.assertEqual(len(env.pending_calls), 3)

        # Sinh cuộc gọi tiếp theo
        env.rng = self.env.rng
        new_calls = env._generate_new_calls()
        if new_calls:
            for nc in new_calls:
                self.assertEqual(nc.status, CallStatus.FAILED)
                self.assertEqual(nc.failure_type, "queue_overflow")
                self.assertIn("từ chối", nc.failure_reason.lower())
            self.assertGreater(env.failed_calls_count, 0)

    def test_continuous_movement_and_event_handling(self) -> None:
        """Kiểm tra xe cứu thương di chuyển liên tục qua nhiều đoạn và xử lý sự kiện trong 1 phút."""
        env = AmbulanceDispatchEnv(seed=42)
        env.reset(generate_initial_calls=False)

        # Đặt xe 1 tại cung (1, 18) ở cự ly 0.95km (cung dài 1.0km)
        amb = env.ambulances[1]
        amb.current_edge = (1, 18)
        amb.distance_from_start = 0.95
        amb.status = AmbulanceStatus.IDLE

        # Tạo cuộc gọi tại cung (18, 2) cách đầu cung 0.2km
        call = EmergencyCall(
            call_id=99,
            edge=(18, 2),
            distance_from_start=0.2,
            coord=(18.0, 17.0),
            severity=1,
            created_time=0.0,
            status=CallStatus.PENDING,
        )
        env.calls[99] = call

        # Phân công xe 1 cho cuộc gọi 99
        amb.dispatch_to_call(call, env.city_map)
        self.assertEqual(amb.status, AmbulanceStatus.EN_ROUTE_TO_CALL)

        # Bước di chuyển 1 phút (vận tốc ~40-60km/h = ~0.67-1.0km/phút)
        # Khoảng cách cần đi: 0.05km (hết cung 1->18) + 0.2km (trên cung 18->2) = 0.25km
        # Trong 1 phút xe hoàn toàn có thể đi hết 0.25km và chuyển sang AT_SCENE
        amb.step_continuous(
            dt_minute=1.0,
            current_minute=0.0,
            city_map=env.city_map,
            calls_dict=env.calls,
        )

        self.assertEqual(amb.status, AmbulanceStatus.AT_SCENE)
        self.assertEqual(call.status, CallStatus.AT_SCENE)
        self.assertLess(amb.scene_time_remaining, call.processing_time)

    def test_abort_to_idle_on_call_failure(self) -> None:
        """Khi cuộc gọi bị xoá (FAILED), xe đang tiếp cận lập tức về trạng thái rảnh."""
        env = AmbulanceDispatchEnv(seed=42)
        env.reset(generate_initial_calls=False)

        call = EmergencyCall(
            call_id=88,
            edge=(21, 24),
            distance_from_start=1.0,
            coord=(18.0, 25.0),
            severity=3,  # Mức 3: failure_time = 30
            created_time=0.0,
            status=CallStatus.PENDING,
        )
        env.calls[88] = call

        amb = env.ambulances[1]
        amb.dispatch_to_call(call, env.city_map)
        self.assertEqual(amb.status, AmbulanceStatus.EN_ROUTE_TO_CALL)

        # Giả lập thời gian trôi qua quá deadline làm cuộc gọi thất bại
        call.check_failure(current_time=35.0)
        self.assertEqual(call.status, CallStatus.FAILED)

        # Gọi step_continuous: xe phát hiện cuộc gọi đã thất bại và lập tức về IDLE
        amb.step_continuous(dt_minute=1.0, current_minute=35.0, city_map=env.city_map, calls_dict=env.calls)
        self.assertEqual(amb.status, AmbulanceStatus.IDLE)
        self.assertIsNone(amb.current_call_id)

        # Giả lập thời gian trôi qua quá deadline làm cuộc gọi thất bại
        call.check_failure(current_time=35.0)
        self.assertEqual(call.status, CallStatus.FAILED)

        # Gọi step_continuous: xe phát hiện cuộc gọi đã thất bại và lập tức về IDLE
        amb.step_continuous(dt_minute=1.0, current_minute=35.0, city_map=env.city_map, calls_dict=env.calls)
        self.assertEqual(amb.status, AmbulanceStatus.IDLE)
        self.assertIsNone(amb.current_call_id)

    def test_hospital_handover_times(self) -> None:
        """Kiểm tra 4 bệnh viện có thời gian gửi bệnh nhân khác nhau với biên độ hợp lý (3-5 phút)."""
        handover_times = [h.handover_time for h in self.env.city_map.hospitals.values()]
        self.assertEqual(len(handover_times), 4)
        self.assertGreaterEqual(min(handover_times), 3.0)
        self.assertLessEqual(max(handover_times), 5.0)
        self.assertLessEqual(max(handover_times) - min(handover_times), 2.5)

    def test_center_vs_outer_speeds_and_congestion(self) -> None:
        """Kiểm tra vận tốc trong trung tâm thấp hơn ngoại ô."""
        center_edge = self.env.city_map.edges[(1, 18)]
        outer_edge = self.env.city_map.edges[(67, 74)]

        self.assertTrue(center_edge.is_center)
        self.assertFalse(outer_edge.is_center)

        # Với cùng trạng thái thông thoáng:
        center_edge.set_congestion(1)
        outer_edge.set_congestion(1)
        self.assertLess(center_edge.speed, outer_edge.speed)

        # Với cùng trạng thái tắc:
        center_edge.set_congestion(3)
        outer_edge.set_congestion(3)
        self.assertLess(center_edge.speed, outer_edge.speed)

    def test_log_step_formatting(self) -> None:
        """Kiểm tra định dạng log đúng mẫu yêu cầu."""
        env = AmbulanceDispatchEnv(seed=42)
        env.reset()
        env.step(auto_dispatch=True)
        log_str = env.format_log_step()

        self.assertIn("Phút thứ 1:", log_str)
        self.assertIn("Số cuộc gọi sinh ra:", log_str)
        self.assertIn("Xe 1 đang di chuyển trên tuyến đường", log_str)
        self.assertIn("Vận tốc:", log_str)
        self.assertIn("Trạng thái:", log_str)
        self.assertIn("Cách tuyến đường chuyển trạng thái:", log_str)
        self.assertIn("Từ tắc -> bình thường:", log_str)
        self.assertIn("Từ thông thoáng -> bình thường:", log_str)
        self.assertIn("Từ bình thường -> tắc:", log_str)
        self.assertIn("Từ bình thường -> thông thoáng:", log_str)
        self.assertIn("Từ tắc -> thông thoáng:", log_str)
        self.assertIn("Từ thông thoáng -> tắc:", log_str)
        self.assertIn("Số cuộc gọi thành công:", log_str)
        self.assertIn("Số cuộc gọi thất bại:", log_str)
        self.assertIn("Reward bước này (r_t):", log_str)
        self.assertIn("Tổng Reward tích lũy:", log_str)
        self.assertIn("Lý do cuộc gọi thất bại trong phút này:", log_str)

    def test_failure_reason_logged_once(self) -> None:
        """Kiểm tra lý do thất bại được ghi rõ ràng và chỉ ghi 1 lần duy nhất khi thất bại."""
        env = AmbulanceDispatchEnv(seed=42)
        env.reset(generate_initial_calls=False)

        # Tạo 1 cuộc gọi sắp hết hạn (mức 3: deadline = 30)
        call = EmergencyCall(
            call_id=50,
            edge=(1, 2),
            distance_from_start=0.5,
            coord=(10.0, 10.0),
            severity=3,  # deadline = 30
            created_time=0.0,
            status=CallStatus.PENDING,
        )
        env.calls[50] = call
        env.current_minute = 30

        # Bước thời gian sang phút 31 (current_time = 31.0 > deadline 30.0) -> call 50 hết hạn thất bại
        env.step()
        log_step_31 = env.format_log_step()
        self.assertIn("Lý do cuộc gọi thất bại trong phút này:", log_step_31)
        self.assertIn("Cuộc gọi #50", log_step_31)
        self.assertIn("Quá hạn thời gian chờ cấp cứu", log_step_31)

        # Bước thời gian tiếp theo (phút 32) -> call 50 không được lặp lại trong danh sách thất bại của phút mới
        env.step()
        log_step_32 = env.format_log_step()
        self.assertNotIn("Cuộc gọi #50", log_step_32)

    def test_initial_calls_at_t0(self) -> None:
        """Kiểm tra tại t=0 có thể sinh cuộc gọi mới vào đầu phút."""
        env = AmbulanceDispatchEnv(seed=123, poisson_lambda=2.0)
        env.reset(generate_initial_calls=True)
        self.assertEqual(env.current_minute, 0)
        # Với lambda = 2.0, Poisson sẽ sinh ít nhất 1 cuộc gọi tại t=0
        self.assertGreater(len(env.calls), 0)
        self.assertEqual(len(env.calls_generated_this_step), len(env.calls))
        log_0 = env.format_log_step()
        self.assertIn("Phút thứ 0:", log_0)
        self.assertIn(f"Số cuộc gọi sinh ra: {len(env.calls_generated_this_step)}", log_0)

    def test_single_assignment_rules(self) -> None:
        """
        Kiểm tra các quy tắc phân công:
        - Xe chỉ nhận tối đa 1 cuộc gọi (chỉ xe đang Rảnh IDLE).
        - Một cuộc gọi được giao tối đa cho 1 xe.
        - Chỉ cuộc gọi đang PENDING mới được chọn.
        """
        env = AmbulanceDispatchEnv(seed=42)
        env.reset(generate_initial_calls=False)

        edges = list(env.city_map.edges.keys())
        e1, e2, e3 = edges[0], edges[1], edges[2]

        # Tạo 2 cuộc gọi PENDING và 1 cuộc gọi AT_SCENE
        c1 = EmergencyCall(1, e1, 0.2, (10.0, 10.0), 1, 0.0, status=CallStatus.PENDING)
        c2 = EmergencyCall(2, e2, 0.2, (10.0, 10.0), 1, 0.0, status=CallStatus.PENDING)
        c3 = EmergencyCall(3, e3, 0.2, (10.0, 10.0), 1, 0.0, status=CallStatus.AT_SCENE)
        env.calls[1] = c1
        env.calls[2] = c2
        env.calls[3] = c3

        # Thử phân công sai: Xe 1 được giao cả c1 và c2 -> Xe 1 chỉ được nhận c1
        # Thử phân công c1 cho cả Xe 1 và Xe 2 -> Chỉ Xe 1 nhận c1
        # Thử phân công c3 (AT_SCENE) cho Xe 3 -> Bị từ chối vì c3 không PENDING
        actions = [
            (1, 1),
            (1, 2),  # Xe 1 nhận thêm c2 -> Bỏ qua
            (2, 1),  # c1 được giao thêm cho Xe 2 -> Bỏ qua
            (3, 3),  # c3 không PENDING -> Bỏ qua
        ]
        env.step(action=actions)

        # Xe 1 nhận c1
        self.assertEqual(env.ambulances[1].current_call_id, 1)
        self.assertEqual(env.ambulances[1].status, AmbulanceStatus.EN_ROUTE_TO_CALL)
        self.assertEqual(c1.assigned_ambulance_id, 1)

        # Xe 2 không bị gán c1 hay c2
        self.assertIsNone(env.ambulances[2].current_call_id)
        self.assertEqual(env.ambulances[2].status, AmbulanceStatus.IDLE)

        # Xe 3 không nhận c3
        self.assertIsNone(env.ambulances[3].current_call_id)
        self.assertEqual(env.ambulances[3].status, AmbulanceStatus.IDLE)

    def test_overtime_truncation_and_backlog(self) -> None:
        """Kiểm tra: Nếu chạm ngưỡng quá giờ 300 phút (max_time + 300), dừng mô phỏng và báo Truncated kèm backlog."""
        env = AmbulanceDispatchEnv(seed=42, max_time=360, max_overtime=300)
        env.reset(generate_initial_calls=False)

        # Đặt thời gian mô phỏng đến sát ngưỡng: 659 phút
        env.current_minute = 659

        # Tạo 1 cuộc gọi còn tồn đọng PENDING
        c = EmergencyCall(101, (1, 2), 0.5, (10.0, 10.0), 1, 650.0, status=CallStatus.PENDING)
        env.calls[101] = c

        # Bước tiếp theo chạm 660 phút (= 360 + 300)
        obs, reward, done, info = env.step()

        self.assertEqual(env.current_minute, 660)
        self.assertTrue(done)
        self.assertTrue(env.is_truncated)
        self.assertTrue(info["truncated"])
        self.assertIsNotNone(info["backlog"])
        self.assertGreaterEqual(info["backlog"]["total_backlog_count"], 1)

        # Kiểm tra nội dung báo cáo backlog
        report = env.format_backlog_report()
        self.assertIn("TRUNCATED", report)
        self.assertIn("300 PHÚT QUÁ GIỜ", report)
        self.assertIn("Cuộc gọi #101", report)

        log_str = env.format_log_step()
        self.assertIn("TRUNCATED", log_str)
        self.assertIn("DANH SÁCH CÁC CÔNG VIỆC TỒN ĐỌNG CHI TIẾT:", log_str)

    def test_reward_table_configuration(self) -> None:
        """Kiểm tra bảng tham số Reward trong REWARD_CONFIG."""
        self.assertEqual(DISTANCE_PENALTY_PER_KM, 0.05)

        for sev in (1, 2, 3):
            self.assertIn(sev, REWARD_CONFIG)
            self.assertGreater(REWARD_CONFIG[sev]["wait_penalty_per_min"], 0.0)
            self.assertGreater(REWARD_CONFIG[sev]["fail_penalty"], 0.0)
            self.assertGreater(REWARD_CONFIG[sev]["complete_reward"], 0.0)
            self.assertGreater(REWARD_CONFIG[sev]["perfect_reward"], REWARD_CONFIG[sev]["complete_reward"])

    def test_reward_formula_calculation(self) -> None:
        """
        Kiểm tra công thức tính phần thưởng r_t tại mỗi bước thời gian t:
        r_t = - sum_c w_c * dW_{c,t} - sum_{c in F_t} p_c - 0.05 * sum_i d_{i,t} + sum_{c in C_t} b_c
        """
        env = AmbulanceDispatchEnv(seed=42, poisson_lambda=0.0)
        env.reset(generate_initial_calls=False)

        # 1. Kiểm tra phạt chờ w_c * dW_{c,t} cho 3 cuộc gọi mức 1, 2, 3 cùng chờ 1 phút (không xe nào di chuyển)
        edges = list(env.city_map.edges.keys())
        env.calls[1] = EmergencyCall(1, edges[0], 0.2, (0.0, 0.0), severity=1, created_time=0.0)
        env.calls[2] = EmergencyCall(2, edges[1], 0.2, (0.0, 0.0), severity=2, created_time=0.0)
        env.calls[3] = EmergencyCall(3, edges[2], 0.2, (0.0, 0.0), severity=3, created_time=0.0)

        obs, reward, done, info = env.step()
        expected_wait = (
            REWARD_CONFIG[1]["wait_penalty_per_min"]
            + REWARD_CONFIG[2]["wait_penalty_per_min"]
            + REWARD_CONFIG[3]["wait_penalty_per_min"]
        )
        self.assertAlmostEqual(reward, -expected_wait, places=5)
        self.assertAlmostEqual(info["reward_breakdown"]["wait_penalty"], expected_wait, places=5)

        # 2. Kiểm tra phạt thất bại p_c khi cuộc gọi mức 3 quá hạn
        env.reset(generate_initial_calls=False)
        c_fail = EmergencyCall(10, edges[0], 0.2, (0.0, 0.0), severity=3, created_time=0.0)
        env.calls[10] = c_fail
        env.current_minute = 30  # sang phút 31 sẽ quá hạn 30p
        obs, reward, done, info = env.step()
        w3 = REWARD_CONFIG[3]["wait_penalty_per_min"]
        p3 = REWARD_CONFIG[3]["fail_penalty"]
        self.assertAlmostEqual(reward, -(w3 + p3), places=5)
        self.assertAlmostEqual(info["reward_breakdown"]["wait_penalty"], w3, places=5)
        self.assertAlmostEqual(info["reward_breakdown"]["failure_penalty"], p3, places=5)

        # 3. Kiểm tra thưởng hoàn thành hoàn hảo (b_c) và phạt quãng đường (-0.05 * d)
        env.reset(generate_initial_calls=False)
        amb = env.ambulances[1]
        amb.current_edge = (1, 18)
        amb.distance_from_start = 0.5
        # Đặt cuộc gọi mức 2 ngay tại vị trí xe (khoảng cách = 0km) để xe lập tức AT_SCENE (dW = 0)
        c_perf = EmergencyCall(20, (1, 18), 0.5, (0.0, 0.0), severity=2, created_time=0.0)
        env.calls[20] = c_perf
        amb.dispatch_to_call(c_perf, env.city_map)
        # Đặt thời gian xử lý tại hiện trường = 0.5 phút để hoàn thành ngay trong phút đầu tiên (is_perfect = True)
        c_perf.processing_time = 0.5

        obs, reward, done, info = env.step()
        self.assertTrue(c_perf.is_perfect)
        # Thưởng hoàn hảo mức 2: b_c = 80.0
        self.assertAlmostEqual(info["reward_breakdown"]["completion_bonus"], 80.0, places=5)
        # Trong 0.5 phút còn lại của bước, xe đi về phía bệnh viện với quãng đường amb.last_step_distance
        expected_dist_pen = 0.05 * amb.last_step_distance
        self.assertAlmostEqual(info["reward_breakdown"]["distance_penalty"], expected_dist_pen, places=5)
        self.assertAlmostEqual(reward, 80.0 - expected_dist_pen, places=5)

    def test_call_slots_ordering(self) -> None:
        """
        Kiểm tra quy tắc sắp xếp Slot thống nhất (Canonical FIFO Slot Rule):
        - Chỉ xét các cuộc gọi đang PENDING.
        - Sắp xếp tăng dần theo (created_time, call_id) vào Slot 0, Slot 1, Slot 2.
        - Các slot chưa có cuộc gọi mang giá trị None.
        """
        env = AmbulanceDispatchEnv(seed=42, max_pending_calls=3)
        env.reset(generate_initial_calls=False)
        edges = list(env.city_map.edges.keys())

        # Ban đầu chưa có cuộc gọi nào -> cả 3 slot đều None
        slots = env.get_call_slots()
        self.assertEqual(len(slots), 3)
        self.assertEqual(slots, [None, None, None])

        # Tạo 3 cuộc gọi:
        # - Call #15 phát sinh lúc t=5.0 (PENDING)
        # - Call #10 phát sinh lúc t=2.0 (PENDING)
        # - Call #12 phát sinh lúc t=2.0 (PENDING)
        # - Call #8 phát sinh lúc t=1.0 nhưng đang AT_SCENE (không vào slot PENDING)
        c15 = EmergencyCall(15, edges[0], 0.2, (0.0, 0.0), 1, created_time=5.0, status=CallStatus.PENDING)
        c10 = EmergencyCall(10, edges[1], 0.2, (0.0, 0.0), 2, created_time=2.0, status=CallStatus.PENDING)
        c12 = EmergencyCall(12, edges[2], 0.2, (0.0, 0.0), 3, created_time=2.0, status=CallStatus.PENDING)
        c8 = EmergencyCall(8, edges[3], 0.2, (0.0, 0.0), 1, created_time=1.0, status=CallStatus.AT_SCENE)

        env.calls[15] = c15
        env.calls[10] = c10
        env.calls[12] = c12
        env.calls[8] = c8

        slots = env.get_call_slots()
        self.assertIs(slots[0], c10)  # (2.0, 10)
        self.assertIs(slots[1], c12)  # (2.0, 12)
        self.assertIs(slots[2], c15)  # (5.0, 15)

        # Nếu Call #10 được giao cho xe (WAITING_FOR_AMBULANCE), chỉ còn 2 cuộc gọi PENDING -> Slot 2 là None
        c10.status = CallStatus.WAITING_FOR_AMBULANCE
        slots_after = env.get_call_slots()
        self.assertIs(slots_after[0], c12)
        self.assertIs(slots_after[1], c15)
        self.assertIsNone(slots_after[2])

    def test_potential_actions_count_and_uniqueness(self) -> None:
        """
        Kiểm tra tạo tất cả các cách ghép hành động tiềm năng (gồm cả không phân công):
        Với 4 xe và 3 slot:
        k=0: 1, k=1: 12, k=2: 36, k=3: 24 => Tổng = 73 cách ghép duy nhất.
        """
        env = AmbulanceDispatchEnv(num_ambulances=4, max_pending_calls=3, seed=42)
        self.assertEqual(env.num_potential_actions, 73)
        self.assertEqual(len(env.potential_actions), 73)
        self.assertEqual(len(set(env.potential_actions)), 73)
        # Hành động 0 là không phân công ()
        self.assertEqual(env.potential_actions[0], ())

        # Kiểm tra số lượng theo từng bậc k (số cặp phân công đồng thời)
        count_by_k = {0: 0, 1: 0, 2: 0, 3: 0}
        for joint_action in env.potential_actions:
            k = len(joint_action)
            count_by_k[k] += 1
            ambs = [a for a, _ in joint_action]
            slots = [s for _, s in joint_action]
            # Không trùng xe, không trùng slot, và sắp xếp tăng dần theo mã xe
            self.assertEqual(len(ambs), len(set(ambs)))
            self.assertEqual(len(slots), len(set(slots)))
            self.assertEqual(ambs, sorted(ambs))

        self.assertEqual(count_by_k[0], 1)
        self.assertEqual(count_by_k[1], 12)
        self.assertEqual(count_by_k[2], 36)
        self.assertEqual(count_by_k[3], 24)

    def test_action_mask_and_decode_action(self) -> None:
        """
        Kiểm tra Action Mask và hàm giải mã hành động decode_action / encode_action:
        - Kiểm tra mask chính xác khi chỉ có một số xe rảnh và một số slot có cuộc gọi.
        - Giải mã chỉ số action_id sang tập các cặp đồng thời [(amb_id, call_id), ...].
        """
        env = AmbulanceDispatchEnv(num_ambulances=4, max_pending_calls=3, seed=42)
        env.reset(generate_initial_calls=False)
        edges = list(env.city_map.edges.keys())

        # Trường hợp 0: Không có cuộc gọi PENDING nào -> Chỉ có Action 0 (không phân công) hợp lệ
        mask0 = env.get_action_mask()
        self.assertEqual(mask0.shape, (73,))
        self.assertTrue(mask0[0])
        self.assertEqual(int(mask0.sum()), 1)

        # Trường hợp 1: Có 2 cuộc gọi PENDING (ID 10 ở Slot 0, ID 12 ở Slot 1),
        # và có 3 xe RẢNH (Xe 1, 2, 3), còn Xe 4 đang bận (AT_SCENE).
        env.calls[10] = EmergencyCall(10, edges[0], 0.2, (0.0, 0.0), 2, created_time=1.0, status=CallStatus.PENDING)
        env.calls[12] = EmergencyCall(12, edges[1], 0.2, (0.0, 0.0), 3, created_time=2.0, status=CallStatus.PENDING)
        env.ambulances[4].status = AmbulanceStatus.AT_SCENE
        env.ambulances[4].current_call_id = 99

        # Với 3 xe rảnh {1, 2, 3} và 2 slot hợp lệ {0, 1}:
        # k=0: 1 cách
        # k=1: C(3,1) * P(2,1) = 3 * 2 = 6 cách
        # k=2: C(3,2) * P(2,2) = 3 * 2 = 6 cách
        # k=3: 0 cách (vì chỉ có 2 slot có cuộc gọi)
        # => Tổng số hành động hợp lệ = 1 + 6 + 6 = 13
        mask1 = env.get_action_mask()
        self.assertEqual(int(mask1.sum()), 13)

        # Kiểm tra mã hóa & giải mã tập phân công đồng thời [(1, 10), (3, 12)]
        # Trong đó Call 10 ở Slot 0, Call 12 ở Slot 1 => tương ứng slot action ((1, 0), (3, 1))
        act_id = env.encode_action([(1, 10), (3, 12)])
        self.assertGreater(act_id, 0)
        self.assertTrue(mask1[act_id])
        self.assertTrue(env.is_action_valid(act_id))
        self.assertTrue(env.is_action_valid([(1, 10), (3, 12)]))

        decoded = env.decode_action(act_id)
        self.assertEqual(decoded, [(1, 10), (3, 12)])

        # Hành động gán Xe 4 (đang bận) hoặc Slot 2 (đang trống) phải bị mask = False
        invalid_slot_action = ((1, 0), (4, 1))
        invalid_act_id = env.slot_action_to_id[invalid_slot_action]
        self.assertFalse(mask1[invalid_act_id])
        self.assertFalse(env.is_action_valid(invalid_act_id))
        self.assertFalse(env.is_action_valid([(1, 10), (4, 12)]))

    def test_simultaneous_joint_action_step(self) -> None:
        """
        Kiểm tra thực thi bước mô phỏng step() với tập các cặp phân công đồng thời
        dưới cả 2 dạng: danh sách cặp [(1, 10), (3, 12)] và chỉ số action_id.
        """
        env = AmbulanceDispatchEnv(num_ambulances=4, max_pending_calls=3, seed=42, poisson_lambda=0.0)
        env.reset(generate_initial_calls=False)
        edges = list(env.city_map.edges.keys())

        env.calls[10] = EmergencyCall(10, edges[0], 0.5, (0.0, 0.0), 1, created_time=0.0, status=CallStatus.PENDING)
        env.calls[12] = EmergencyCall(12, edges[1], 0.5, (0.0, 0.0), 2, created_time=0.0, status=CallStatus.PENDING)

        # Agent truyền tập các cặp phân công đồng thời [(1, 10), (3, 12)]
        obs, reward, done, info = env.step(action=[(1, 10), (3, 12)])

        self.assertEqual(info["decoded_action"], [(1, 10), (3, 12)])
        self.assertGreater(info["action_id"], 0)
        self.assertEqual(env.ambulances[1].current_call_id, 10)
        self.assertEqual(env.ambulances[3].current_call_id, 12)
        self.assertIsNone(env.ambulances[2].current_call_id)
        self.assertIsNone(env.ambulances[4].current_call_id)

    def test_observation_traffic_time_step_and_edge_congestion(self) -> None:
        """
        Kiểm tra Observation trả về thêm:
        - traffic_time_step chạy song song với current_minute.
        - Tình trạng tắc trong mỗi cung đường (edge_congestion, edge_congestion_vector, edges).
        """
        env = AmbulanceDispatchEnv(seed=42, poisson_lambda=0.0)
        obs0 = env.reset(generate_initial_calls=False)

        # Tại t = 0: current_minute == 0 và traffic_time_step == 0
        self.assertEqual(obs0["current_minute"], 0)
        self.assertEqual(obs0["traffic_time_step"], 0)
        self.assertEqual(env.traffic_time_step, 0)
        self.assertEqual(env.city_map.traffic_time_step, 0)

        # Kiểm tra thông tin tình trạng tắc trên từng cung đường
        num_edges = len(env.city_map.edges)
        self.assertEqual(len(obs0["edge_congestion"]), num_edges)
        self.assertEqual(obs0["edge_congestion_vector"].shape, (num_edges,))
        self.assertEqual(len(obs0["edges"]), num_edges)

        for ek, edge_obj in env.city_map.edges.items():
            self.assertEqual(obs0["edge_congestion"][ek], edge_obj.congestion)
            self.assertIn(obs0["edge_congestion"][ek], (1, 2, 3))
            self.assertEqual(obs0["edges"][ek]["congestion"], edge_obj.congestion)
            self.assertEqual(obs0["edges"][ek]["congestion_name"], edge_obj.congestion_name)
            self.assertEqual(obs0["edges"][ek]["speed"], edge_obj.speed)

        # Chạy qua 3 bước thời gian: traffic_time_step phải tăng song song với current_minute
        for expected_step in (1, 2, 3):
            obs, reward, done, info = env.step()
            self.assertEqual(obs["current_minute"], expected_step)
            self.assertEqual(obs["traffic_time_step"], expected_step)
            self.assertEqual(obs["traffic_time_step"], obs["current_minute"])
            self.assertEqual(info["traffic_time_step"], expected_step)
            self.assertEqual(env.traffic_time_step, env.current_minute)
            self.assertEqual(env.city_map.traffic_time_step, expected_step)
            # Kiểm tra đồng bộ độ tắc sau khi step_traffic()
            for ek, edge_obj in env.city_map.edges.items():
                self.assertEqual(obs["edge_congestion"][ek], edge_obj.congestion)

    def test_multi_episode_reset_diversity(self) -> None:
        """
        Kiểm tra khi chạy với nhiều episode liên tiếp (gọi env.reset() không truyền seed),
        các thành phần sinh ngẫu nhiên (vị trí xe ban đầu, tình trạng tắc đường ban đầu,
        và chuỗi cuộc gọi phát sinh) mang tính đa dạng, khác nhau giữa các episode.
        """
        env = AmbulanceDispatchEnv(seed=42, poisson_lambda=0.5)

        episode_amb_positions = []
        episode_traffic_vectors = []
        episode_call_signatures = []

        for ep_idx in range(1, 5):
            if ep_idx > 1:
                obs = env.reset()
            else:
                obs = env.get_observation()

            self.assertEqual(obs["episode"], ep_idx)
            self.assertEqual(obs["current_minute"], 0)
            self.assertEqual(obs["traffic_time_step"], 0)

            # 1. Ghi nhận vị trí khởi tạo của các xe cứu thương trong episode này
            amb_pos = tuple(
                (aid, a.current_edge, a.distance_from_start)
                for aid, a in sorted(env.ambulances.items())
            )
            episode_amb_positions.append(amb_pos)

            # 2. Ghi nhận vector độ tắc đường ban đầu của 176 cung đường trong episode này
            episode_traffic_vectors.append(tuple(obs["edge_congestion_vector"].tolist()))

            # 3. Chạy thử 10 bước để thu thập chuỗi cuộc gọi phát sinh trong episode
            for _ in range(10):
                env.step(auto_dispatch=True)
            call_sig = tuple(
                (c.call_id, c.edge, c.distance_from_start, c.severity, c.created_time)
                for c in env.calls.values()
            )
            episode_call_signatures.append(call_sig)

        # Kiểm tra cả 4 episode đều có vị trí xe, trạng thái giao thông và cuộc gọi khác nhau
        self.assertEqual(len(set(episode_amb_positions)), 4)
        self.assertEqual(len(set(episode_traffic_vectors)), 4)
        self.assertEqual(len(set(episode_call_signatures)), 4)

        # Kiểm tra khi truyền seed tường minh vào reset(seed=99) thì tái lập chính xác trạng thái đó
        obs_seed_a = env.reset(seed=99)
        obs_seed_b = env.reset(seed=99)
        self.assertEqual(
            obs_seed_a["edge_congestion_vector"].tolist(),
            obs_seed_b["edge_congestion_vector"].tolist(),
        )
        for aid in env.ambulances:
            self.assertEqual(
                obs_seed_a["ambulances"][aid]["current_edge"],
                obs_seed_b["ambulances"][aid]["current_edge"],
            )

    def test_observation_detailed_ambulance_and_call_fields(self) -> None:
        """
        Kiểm tra đầy đủ các biến mới trả về trong Observation theo yêu cầu:
        1. Top-level: edge_congestion_vector, call_slots, action_mask, edges_by_id, edge_congestion_by_id.
        2. Ambulance: target_type (và targer_type), target_edge, target_edge_id, target_dist,
           assigned_hospital_id, hospital_handover_time, scene_time_remaining, hospital_time_remaining,
           current_edge_id, route_edge_ids.
        3. Call: created_time, failure_time, processing_time, assigned_ambulance_id, arrived_at_scene_time, edge_id.
        """
        env = AmbulanceDispatchEnv(seed=42, poisson_lambda=0.0)
        obs = env.reset(generate_initial_calls=False)

        # 1. Kiểm tra top-level observation
        self.assertIn("edge_congestion_vector", obs)
        self.assertIn("call_slots", obs)
        self.assertIn("action_mask", obs)
        self.assertIn("edges_by_id", obs)
        self.assertIn("edge_congestion_by_id", obs)
        self.assertEqual(obs["edge_congestion_vector"].shape, (172,))
        self.assertEqual(len(obs["edges_by_id"]), 172)
        self.assertEqual(len(obs["edge_congestion_by_id"]), 172)

        # Kiểm tra cung đường 1 trong edges_by_id
        edge_1 = obs["edges_by_id"][1]
        self.assertEqual(edge_1["edge_id"], 1)
        self.assertEqual(edge_1["id"], 1)
        self.assertIn("edge", edge_1)
        self.assertIn("from_node", edge_1)
        self.assertIn("to_node", edge_1)
        self.assertIn("congestion", edge_1)
        self.assertIn("speed", edge_1)

        # 2. Kiểm tra các biến của Ambulance ở trạng thái Rảnh ban đầu
        for aid, amb_info in obs["ambulances"].items():
            self.assertIn("current_edge_id", amb_info)
            self.assertGreater(amb_info["current_edge_id"], 0)
            self.assertIn("target_type", amb_info)
            self.assertIn("targer_type", amb_info)
            self.assertIsNone(amb_info["target_type"])
            self.assertIsNone(amb_info["targer_type"])
            self.assertIsNone(amb_info["target_edge"])
            self.assertIsNone(amb_info["target_edge_id"])
            self.assertIsNone(amb_info["target_dist"])
            self.assertIsNone(amb_info["assigned_hospital_id"])
            self.assertEqual(amb_info["hospital_handover_time"], 0.0)
            self.assertEqual(amb_info["scene_time_remaining"], 0.0)
            self.assertEqual(amb_info["hospital_time_remaining"], 0.0)
            self.assertIn("route_edge_ids", amb_info)

        # 3. Tạo một cuộc gọi thử nghiệm để kiểm tra các biến của Call
        edges = list(env.city_map.edges.keys())
        first_edge = edges[0]
        first_edge_id = env.city_map.edge_to_id[first_edge]

        test_call = EmergencyCall(
            call_id=999,
            edge=first_edge,
            distance_from_start=0.5,
            coord=(15.0, 15.0),
            severity=2,  # failure_time = 40, processing_time = 6
            created_time=0.0,
            edge_id=first_edge_id,
        )
        env.calls[999] = test_call

        obs = env.get_observation()
        slot0_call = obs["call_slots"][0]
        self.assertIsNotNone(slot0_call)
        self.assertEqual(slot0_call["id"], 999)
        self.assertEqual(slot0_call["created_time"], 0.0)
        self.assertEqual(slot0_call["failure_time"], 40.0)
        self.assertEqual(slot0_call["processing_time"], 6.0)
        self.assertEqual(slot0_call["edge_id"], first_edge_id)
        self.assertIsNone(slot0_call["assigned_ambulance_id"])
        self.assertIsNone(slot0_call["arrived_at_scene_time"])

        # 4. Phân công Xe 1 nhận cuộc gọi 999 và kiểm tra biến trong Ambulance
        obs, reward, done, info = env.step(action=[(1, 999)])
        amb1_info = obs["ambulances"][1]
        self.assertEqual(amb1_info["target_type"], "call")
        self.assertEqual(amb1_info["targer_type"], "call")
        self.assertEqual(amb1_info["target_edge"], first_edge)
        self.assertEqual(amb1_info["target_edge_id"], first_edge_id)
        self.assertEqual(amb1_info["target_dist"], 0.5)

        # Cuộc gọi trong call_slots đã ghi nhận assigned_ambulance_id = 1
        # Lưu ý: Khi xe nhận cuộc gọi thì cuộc gọi chuyển sang WAITING_FOR_AMBULANCE (rời khỏi hàng chờ PENDING)
        active_call_info = next(c for c in obs["active_calls"] if c["id"] == 999)
        self.assertEqual(active_call_info["assigned_ambulance_id"], 1)

        # 5. Kiểm tra trạng thái tới hiện trường AT_SCENE và thời gian sơ cứu scene_time_remaining
        test_call.mark_at_scene(arrival_time=3.5)
        amb1 = env.ambulances[1]
        amb1.status = AmbulanceStatus.AT_SCENE
        amb1.scene_time_remaining = 6.0

        obs = env.get_observation()
        active_call_info = next(c for c in obs["active_calls"] if c["id"] == 999)
        self.assertEqual(active_call_info["arrived_at_scene_time"], 3.5)
        self.assertEqual(obs["ambulances"][1]["scene_time_remaining"], 6.0)

        # 6. Kiểm tra chuyển sang bệnh viện và các biến bệnh viện bàn giao
        best_hid = amb1.select_best_hospital_and_route(env.city_map)
        obs = env.get_observation()
        amb1_info = obs["ambulances"][1]
        self.assertEqual(amb1_info["target_type"], "hospital")
        self.assertEqual(amb1_info["assigned_hospital_id"], best_hid)
        self.assertGreater(amb1_info["hospital_handover_time"], 0.0)

        # 7. Kiểm tra trạng thái tại bệnh viện AT_HOSPITAL và hospital_time_remaining
        amb1.status = AmbulanceStatus.AT_HOSPITAL
        amb1.hospital_time_remaining = 4.0
        obs = env.get_observation()
        self.assertEqual(obs["ambulances"][1]["hospital_time_remaining"], 4.0)

    def test_t0_failed_calls_penalized(self) -> None:
        """
        Kiểm tra: Tại t=0 các cuộc gọi thất bại vẫn bị trừ điểm thưởng như thường.
        - Khi t=0 sinh ra cuộc gọi vượt quá 3 (bị từ chối do queue overflow),
          các cuộc gọi thất bại phải bị trừ điểm phạt ngay tại t=0 (last_step_reward, total_reward).
        - Phút 0 hiển thị đầy đủ phạt thất bại trong format_log_step.
        - Khi sang phút 1 (step), điểm phạt thất bại t=0 không bị mất và không bị trừ lặp lại.
        - Hàm penalize_call_failure hoạt động chính xác và chống trừ trùng lặp.
        """
        # 1. Kiểm tra sinh cuộc gọi tại t=0 vượt quá 3 cuộc gọi (seed=1, poisson_lambda=5.0)
        env = AmbulanceDispatchEnv(seed=1, poisson_lambda=5.0)
        self.assertEqual(env.current_minute, 0)
        self.assertEqual(len(env.calls), 4)
        self.assertEqual(env.failed_calls_count, 1)

        c4 = env.calls[4]
        self.assertEqual(c4.status, CallStatus.FAILED)
        self.assertEqual(c4.failure_type, "queue_overflow")
        expected_penalty = REWARD_CONFIG[c4.severity]["fail_penalty"]

        # Điểm thưởng tại t=0 phải bị trừ điểm phạt thất bại ngay lập tức
        self.assertEqual(env.last_step_reward, -expected_penalty)
        self.assertEqual(env.total_reward, -expected_penalty)
        self.assertEqual(env.last_reward_breakdown["failure_penalty"], expected_penalty)
        self.assertEqual(env.last_reward_breakdown["total"], -expected_penalty)

        # Observation tại t=0 cũng phản ánh total_reward và last_step_reward
        obs0 = env.get_observation()
        self.assertEqual(obs0["total_reward"], -expected_penalty)
        self.assertEqual(obs0["last_step_reward"], -expected_penalty)
        self.assertEqual(obs0["failed_calls"], 1)

        # Kiểm tra file log tại phút 0 ghi nhận điểm phạt thất bại
        log_0 = env.format_log_step()
        self.assertIn("Phút thứ 0:", log_0)
        self.assertIn("Số cuộc gọi thất bại: 1", log_0)
        self.assertIn(f"Phạt thất bại: -{expected_penalty:.2f}", log_0)
        self.assertIn(f"Tổng Reward tích lũy: -{expected_penalty:.2f}", log_0)
        self.assertIn("Cuộc gọi #4", log_0)

        # 2. Bước sang phút 1: điểm phạt t=0 được bảo toàn trong total_reward
        obs1, reward1, done1, info1 = env.step()
        self.assertEqual(env.total_reward, -expected_penalty + reward1)
        self.assertEqual(info1["total_reward"], env.total_reward)

        # 3. Kiểm tra hàm penalize_call_failure trực tiếp
        clean_env = AmbulanceDispatchEnv(seed=42, poisson_lambda=0.0)
        clean_env.reset(generate_initial_calls=False)
        self.assertEqual(clean_env.total_reward, 0.0)

        edges = list(clean_env.city_map.edges.keys())
        manual_call = EmergencyCall(100, edges[0], 0.5, (0.0, 0.0), severity=3, created_time=0.0)
        clean_env.calls[100] = manual_call
        pen = clean_env.penalize_call_failure(manual_call)
        self.assertEqual(pen, 120.0)
        self.assertEqual(clean_env.total_reward, -120.0)
        self.assertEqual(clean_env.last_step_reward, -120.0)
        self.assertEqual(clean_env.last_reward_breakdown["failure_penalty"], 120.0)

        # Thử phạt lại cuộc gọi đã phạt: không bị phạt lặp lại
        pen2 = clean_env.penalize_call_failure(manual_call)
        self.assertEqual(pen2, 0.0)
        self.assertEqual(clean_env.total_reward, -120.0)


if __name__ == "__main__":
    unittest.main()
