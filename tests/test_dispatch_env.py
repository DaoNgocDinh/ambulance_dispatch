"""
Bộ kiểm thử tự động (Unit Tests) cho Môi trường Điều phối Xe Cứu Thương (dispatch_env).
"""

from __future__ import annotations

import unittest
from map_env.city_map import CityMap
from dispatch_env.types import (
    AmbulanceStatus,
    CallStatus,
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
        """Kiểm tra 4 xe cứu thương, vị trí ban đầu ngẫu nhiên, trạng thái luôn Rảnh (IDLE)."""
        self.assertEqual(len(self.env.ambulances), 4)
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
        """Kiểm tra bảng tham số Reward đúng chuẩn theo hình ảnh đặc tả."""
        self.assertEqual(DISTANCE_PENALTY_PER_KM, 0.05)

        # Mức độ 1: Đợi=1, Thất bại=30, Hoàn thành=50, Hoàn hảo=70
        self.assertEqual(REWARD_CONFIG[1]["wait_penalty_per_min"], 1.0)
        self.assertEqual(REWARD_CONFIG[1]["fail_penalty"], 30.0)
        self.assertEqual(REWARD_CONFIG[1]["complete_reward"], 50.0)
        self.assertEqual(REWARD_CONFIG[1]["perfect_reward"], 70.0)

        # Mức độ 2: Đợi=2, Thất bại=60, Hoàn thành=65, Hoàn hảo=80
        self.assertEqual(REWARD_CONFIG[2]["wait_penalty_per_min"], 2.0)
        self.assertEqual(REWARD_CONFIG[2]["fail_penalty"], 60.0)
        self.assertEqual(REWARD_CONFIG[2]["complete_reward"], 65.0)
        self.assertEqual(REWARD_CONFIG[2]["perfect_reward"], 80.0)

        # Mức độ 3: Đợi=4, Thất bại=120, Hoàn thành=70, Hoàn hảo=100
        self.assertEqual(REWARD_CONFIG[3]["wait_penalty_per_min"], 4.0)
        self.assertEqual(REWARD_CONFIG[3]["fail_penalty"], 120.0)
        self.assertEqual(REWARD_CONFIG[3]["complete_reward"], 70.0)
        self.assertEqual(REWARD_CONFIG[3]["perfect_reward"], 100.0)

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
        # w_1 + w_2 + w_3 = 1 + 2 + 4 = 7.0 -> r_t = -7.0
        self.assertAlmostEqual(reward, -7.0, places=5)
        self.assertAlmostEqual(info["reward_breakdown"]["wait_penalty"], 7.0, places=5)

        # 2. Kiểm tra phạt thất bại p_c khi cuộc gọi mức 3 quá hạn (p_3 = 120)
        env.reset(generate_initial_calls=False)
        c_fail = EmergencyCall(10, edges[0], 0.2, (0.0, 0.0), severity=3, created_time=0.0)
        env.calls[10] = c_fail
        env.current_minute = 30  # sang phút 31 sẽ quá hạn 30p
        obs, reward, done, info = env.step()
        # Phạt chờ 1 phút (w_3 = 4) + Phạt thất bại (p_3 = 120) = -124.0
        self.assertAlmostEqual(reward, -124.0, places=5)
        self.assertAlmostEqual(info["reward_breakdown"]["wait_penalty"], 4.0, places=5)
        self.assertAlmostEqual(info["reward_breakdown"]["failure_penalty"], 120.0, places=5)

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


if __name__ == "__main__":
    unittest.main()
