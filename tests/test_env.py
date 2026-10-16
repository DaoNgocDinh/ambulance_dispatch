"""
Bộ kiểm thử tự động (Unit Tests) cho môi trường SmartCityAmbulanceEnv.
Kiểm tra tính toàn vẹn của mô hình MDP:
- 15 nút giao thông (|V| = 15)
- 3 xe cứu thương (K = 3)
- Tối đa 3 cuộc gọi đồng thời (M = 3)
- 5 tuyến đường huyết mạch (|E_arterial| = 5)
- Không gian trạng thái S, Hành động A, Phần thưởng R và Chuyển trạng thái P.
"""

from __future__ import annotations

import unittest

import numpy as np

from environment import (
    AMB_STATE_IDLE,
    AMB_STATE_ON_SCENE,
    AMB_STATE_RELOCATING,
    AMB_STATE_TO_SCENE,
    EnvConfig,
    PRIO_RED,
    SmartCityAmbulanceEnv,
    TRAFFIC_CONGESTED,
)


class TestSmartCityAmbulanceEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.env = SmartCityAmbulanceEnv(config=EnvConfig(max_steps=50), seed=123)
        self.obs = self.env.reset(seed=123)

    def test_model_dimensions(self) -> None:
        """Kiểm tra đúng cấu hình 15 nút, 3 xe, tối đa 3 cuộc gọi, 5 tuyến huyết mạch."""
        self.assertEqual(self.env.num_nodes, 15)
        self.assertEqual(self.env.num_ambulances, 3)
        self.assertEqual(self.env.max_pending_requests, 3)
        self.assertEqual(self.env.num_arterial_roads, 5)
        self.assertEqual(len(self.env.graph.nodes), 15)
        self.assertEqual(len(self.env.graph.arterial_edges), 5)

    def test_state_vector_structure(self) -> None:
        """Kiểm tra S_t = [X_amb(6), X_req(9), X_traffic(5)] có đúng 20 phần tử và miền giá trị."""
        s_vec = self.obs["state_vector"]
        self.assertEqual(s_vec.shape, (20,))

        # X_amb = [p1, st1, p2, st2, p3, st3]
        x_amb = self.obs["X_amb"]
        self.assertEqual(len(x_amb), 6)
        for k in range(3):
            p_k = x_amb[2 * k]
            st_k = x_amb[2 * k + 1]
            self.assertTrue(1 <= p_k <= 15)
            self.assertTrue(0 <= st_k <= 4)

        # X_req = 3 * [loc_m, prio_m, wait_m]
        x_req = self.obs["X_req"]
        self.assertEqual(len(x_req), 9)
        for m in range(3):
            loc_m, prio_m, wait_m = x_req[3 * m : 3 * m + 3]
            self.assertTrue(0 <= loc_m <= 15)
            self.assertTrue(0 <= prio_m <= 3)
            self.assertTrue(wait_m >= 0)

        # X_traffic = [c1, c2, c3, c4, c5]
        x_traffic = self.obs["X_traffic"]
        self.assertEqual(len(x_traffic), 5)
        for c_e in x_traffic:
            self.assertIn(int(c_e), (0, 1, 2))

    def test_action_encoding_and_decoding(self) -> None:
        """Kiểm tra giải mã đầy đủ 25 hành động trong A = {dispatch, relocate, route, standby}."""
        self.assertEqual(self.env.number_of_actions, 25)
        for a in range(25):
            info = self.env.decode_action(a)
            self.assertIn(info["type"], ("dispatch", "relocate", "route", "standby"))

    def test_dispatch_and_golden_time_reward(self) -> None:
        """Kiểm tra phân bổ xe và phần thưởng Thời gian vàng (+50)."""
        from environment.entities import EmergencyRequest

        self.env.reset(seed=99)
        # Đặt cuộc gọi Đỏ ngay tại nút 1 (nơi Xe 1 đang trực)
        self.env.pending_requests = [
            EmergencyRequest(req_id=1, loc=1, prio=PRIO_RED, wait_time=0),
            None,
            None,
        ]
        action = self.env.encode_dispatch_action(amb_id=0, req_slot=0)
        _, reward, _, info = self.env.step(action)

        # Xe 1 ở nút 1 tiếp cận ngay lập tức -> nhận thưởng Thời gian vàng +50
        self.assertEqual(info["reward_breakdown"]["golden_time_bonus"], 50.0)
        self.assertGreaterEqual(reward, 45.0)

    def test_relocate_and_route_actions(self) -> None:
        """Kiểm tra hành động tái định vị a_relocate(k, j) và định tuyến tránh kẹt xe a_route(k, u)."""
        self.env.reset(seed=77)
        # Tái định vị Xe 1 (từ trạm nút 1) sang trạm nút 8 (station_idx=1)
        reloc_action = self.env.encode_relocate_action(amb_id=0, station_idx=1)
        self.env.step(reloc_action)
        self.assertIn(self.env.ambulances[0].status, (AMB_STATE_RELOCATING, AMB_STATE_IDLE))

    def test_congested_road_and_empty_station_penalties(self) -> None:
        """Kiểm tra phạt đường ùn tắc (-2) và phạt làm trống hoàn toàn trạm trực (-5)."""
        self.env.reset(seed=55)
        # Đưa cả 3 xe ra khỏi trạm trực để kiểm tra phạt mất độ phủ an toàn (-5)
        self.env.ambulances[0].node = 2
        self.env.ambulances[0].status = AMB_STATE_TO_SCENE
        self.env.ambulances[0].target_node = 3

        self.env.ambulances[1].node = 7
        self.env.ambulances[1].status = AMB_STATE_TO_SCENE
        self.env.ambulances[1].target_node = 6

        self.env.ambulances[2].node = 14
        self.env.ambulances[2].status = AMB_STATE_TO_SCENE
        self.env.ambulances[2].target_node = 12

        # Đặt tuyến huyết mạch 1 (nút 2 <-> nút 3) ở trạng thái Ùn tắc (2)
        self.env.graph.arterial_traffic[0] = TRAFFIC_CONGESTED

        _, _, _, info = self.env.step(self.env.STANDBY_ACTION)
        self.assertEqual(info["reward_breakdown"]["congested_road_penalty"], -2.0)
        self.assertEqual(info["reward_breakdown"]["empty_station_penalty"], -5.0)


if __name__ == "__main__":
    unittest.main()
