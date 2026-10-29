"""
Bộ kiểm thử tự động (Unit Tests) cho Môi trường Điều phối Xe Cứu Thương (dispatch_env).
"""

from __future__ import annotations

import unittest
from map_env.city_map import CityMap
from dispatch_env.types import AmbulanceStatus, CallStatus, SEVERITY_CONFIG
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
            severity=1,  # failure_time = 30
            created_time=5.0,
        )
        self.assertFalse(call.check_failure(current_time=30.0))
        self.assertTrue(call.check_failure(current_time=35.1))
        self.assertEqual(call.status, CallStatus.FAILED)

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
        env.reset()

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
        env.reset()

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
        env.reset()

        call = EmergencyCall(
            call_id=88,
            edge=(21, 24),
            distance_from_start=1.0,
            coord=(18.0, 25.0),
            severity=1,  # failure_time = 30
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
        self.assertIn("Lý do cuộc gọi thất bại trong phút này:", log_str)

    def test_failure_reason_logged_once(self) -> None:
        """Kiểm tra lý do thất bại được ghi rõ ràng và chỉ ghi 1 lần duy nhất khi thất bại."""
        env = AmbulanceDispatchEnv(seed=42)
        env.reset()

        # Tạo 1 cuộc gọi sắp hết hạn
        call = EmergencyCall(
            call_id=50,
            edge=(1, 2),
            distance_from_start=0.5,
            coord=(10.0, 10.0),
            severity=1,  # deadline = 30
            created_time=0.0,
            status=CallStatus.PENDING,
        )
        env.calls[50] = call
        env.current_minute = 29

        # Bước thời gian sang phút 30 -> call 50 hết hạn thất bại
        env.step()
        log_step_30 = env.format_log_step()
        self.assertIn("Lý do cuộc gọi thất bại trong phút này:", log_step_30)
        self.assertIn("Cuộc gọi #50", log_step_30)
        self.assertIn("Quá hạn thời gian chờ cấp cứu", log_step_30)

        # Bước thời gian tiếp theo (phút 31) -> call 50 không được lặp lại
        env.step()
        log_step_31 = env.format_log_step()
        self.assertNotIn("Cuộc gọi #50", log_step_31)


if __name__ == "__main__":
    unittest.main()
