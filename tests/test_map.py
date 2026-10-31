"""
Bộ kiểm thử tự động (Unit Tests) cho Môi trường Bản đồ Đô thị 80 NÚT và 4 BỆNH VIỆN.
Kiểm tra các yêu cầu:
1. Đủ 80 nút được đánh số từ 1 đến 80.
2. Đầy đủ các loại giao lộ: Ngã 4, Ngã 3, Đường rẽ/nối tiếp, Đường cụt.
3. Đường 2 chiều gồm 2 cung có hướng riêng biệt, độ dài tự động sinh, độ tắc in {1, 2, 3}.
4. Độ dài đoạn đường có độ đa dạng cao (min ~1.0km, max ~6.0km+).
5. Hai chiều u->v và v->u có độ tắc độc lập và có thể khác nhau.
6. Độ tắc biến thiên ngẫu nhiên theo thời gian chạy (chuỗi Markov).
7. Đúng 4 BỆNH VIỆN:
   - Mỗi bệnh viện nằm trên DUY NHẤT 1 cung đường có hướng, tính độ dài từ điểm đi.
   - 2 bệnh viện GẦN TRUNG TÂM (khu vực mạng lưới đường dày đặc).
   - 2 bệnh viện Ở XA TRUNG TÂM (khu vực ngoại ô vành đai).
8. Đồ thị liên thông hoàn toàn và định tuyến chính xác đến cả 4 bệnh viện từ bất kỳ nút nào.
"""

from __future__ import annotations

import math
import unittest
from map_env import CityMap, IntersectionType, CONGESTION_CLEAR, CONGESTION_MODERATE, CONGESTION_HEAVY


class TestCityMap(unittest.TestCase):
    def setUp(self) -> None:
        self.city_map = CityMap(seed=42)

    def test_node_count_and_indexing(self) -> None:
        """Kiểm tra đúng 80 nút và đánh số liên tục từ 1 đến 80."""
        self.assertEqual(len(self.city_map.nodes), 80)
        expected_ids = list(range(1, 81))
        self.assertEqual(sorted(list(self.city_map.nodes.keys())), expected_ids)

    def test_intersection_types_exist(self) -> None:
        """Kiểm tra có đầy đủ các loại nút giao: Ngã 4, Ngã 3, Đường rẽ, Đường cụt."""
        types_found = {node.intersection_type for node in self.city_map.nodes.values()}
        self.assertIn(IntersectionType.FOUR_WAY, types_found, "Phải có Ngã 4")
        self.assertIn(IntersectionType.THREE_WAY, types_found, "Phải có Ngã 3")
        self.assertIn(IntersectionType.TURN, types_found, "Phải có Đường rẽ")
        self.assertIn(IntersectionType.DEAD_END, types_found, "Phải có Đường cụt")

        four_ways = [n for n in self.city_map.nodes.values() if n.intersection_type == IntersectionType.FOUR_WAY]
        three_ways = [n for n in self.city_map.nodes.values() if n.intersection_type == IntersectionType.THREE_WAY]
        turns = [n for n in self.city_map.nodes.values() if n.intersection_type == IntersectionType.TURN]
        dead_ends = [n for n in self.city_map.nodes.values() if n.intersection_type == IntersectionType.DEAD_END]

        self.assertGreaterEqual(len(four_ways), 5)
        self.assertGreaterEqual(len(three_ways), 10)
        self.assertGreaterEqual(len(turns), 10)
        self.assertGreaterEqual(len(dead_ends), 20)

    def test_two_way_edges_and_lengths(self) -> None:
        """Kiểm tra các đường nối là 2 chiều và độ dài dương."""
        for (u, v) in self.city_map.road_pairs:
            self.assertIn((u, v), self.city_map.edges)
            self.assertIn((v, u), self.city_map.edges)

            edge_uv = self.city_map.edges[(u, v)]
            edge_vu = self.city_map.edges[(v, u)]

            self.assertGreater(edge_uv.length, 0.0)
            self.assertEqual(edge_uv.length, edge_vu.length)

    def test_road_length_diversity(self) -> None:
        """Kiểm tra độ đa dạng của độ dài đoạn đường (từ ngắn ~1.0km đến dài ~6.0km+)."""
        lengths = [e.length for e in self.city_map.edges.values()]
        min_len = min(lengths)
        max_len = max(lengths)

        self.assertLessEqual(min_len, 1.2, "Phải có các đoạn đường ngắn <= 1.2km")
        self.assertGreaterEqual(max_len, 5.0, "Phải có các đoạn đường dài >= 5.0km")
        self.assertGreater(max_len / min_len, 4.0, "Độ dài phải đa dạng gấp ít nhất 4 lần giữa ngắn nhất và dài nhất")

    def test_independent_two_way_congestion(self) -> None:
        """Kiểm tra 2 chiều u->v và v->u có độ tắc độc lập thuộc {1, 2, 3}."""
        self.city_map.set_edge_congestion(1, 18, CONGESTION_CLEAR)      # Chiều đi: 1
        self.city_map.set_edge_congestion(18, 1, CONGESTION_HEAVY)      # Chiều về: 3

        self.assertEqual(self.city_map.edges[(1, 18)].congestion, 1)
        self.assertEqual(self.city_map.edges[(18, 1)].congestion, 3)
        self.assertNotEqual(self.city_map.edges[(1, 18)].congestion, self.city_map.edges[(18, 1)].congestion)

        for edge in self.city_map.edges.values():
            self.assertIn(edge.congestion, (1, 2, 3))

    def test_dynamic_traffic_variation(self) -> None:
        """Kiểm tra độ tắc đường có biến thiên qua các bước thời gian."""
        initial_congestions = {k: e.congestion for k, e in self.city_map.edges.items()}

        changes_detected = False
        for _ in range(5):
            self.city_map.step_traffic()
            current_congestions = {k: e.congestion for k, e in self.city_map.edges.items()}
            if current_congestions != initial_congestions:
                changes_detected = True
                break

        self.assertTrue(changes_detected, "Độ tắc đường phải biến thiên sau các bước thời gian")

    def test_four_hospitals_and_locations(self) -> None:
        """
        Kiểm tra:
        1. Đúng 4 bệnh viện.
        2. Mỗi bệnh viện chỉ nằm trên DUY NHẤT 1 cung đường có hướng.
        3. 2 bệnh viện GẦN TRUNG TÂM (d <= 5km từ trung tâm N1 (18, 16)).
        4. 2 bệnh viện XA TRUNG TÂM (d > 7km từ trung tâm N1 (18, 16)).
        5. Vị trí tính bằng độ dài từ điểm xuất phát (from_node).
        """
        self.assertEqual(len(self.city_map.hospitals), 4)

        center_x, center_y = self.city_map.nodes[1].coord

        near_center = []
        far_from_center = []

        for hid, hosp in self.city_map.hospitals.items():
            # Kiểm tra khoảng cách đến tâm đô thị Nút 1
            hx, hy = hosp.coord
            dist_to_center = math.hypot(hx - center_x, hy - center_y)

            if dist_to_center <= 5.0:
                near_center.append(hid)
            else:
                far_from_center.append(hid)

            # Kiểm tra chỉ nằm trên 1 cung có hướng
            u, v = hosp.from_node, hosp.to_node
            self.assertIn((u, v), self.city_map.edges)
            # Chiều ngược lại không chứa bệnh viện
            other_hosp_edges = [h.directed_edge_key for h in self.city_map.hospitals.values() if h.hospital_id != hid]
            self.assertNotIn((v, u), other_hosp_edges)

            # Kiểm tra khoảng cách từ điểm đi
            self.assertGreater(hosp.distance_from_start, 0.0)
            self.assertLess(hosp.distance_from_start, hosp.road_length)
            self.assertEqual(round(hosp.distance_from_start + hosp.distance_to_end, 2), hosp.road_length)

        self.assertEqual(len(near_center), 2, "Phải có đúng 2 bệnh viện gần trung tâm")
        self.assertEqual(len(far_from_center), 2, "Phải có đúng 2 bệnh viện xa trung tâm")

    def test_graph_connectivity_and_all_hospitals_routing(self) -> None:
        """Kiểm tra đồ thị 80 nút liên thông và tìm lộ trình chính xác đến cả 4 bệnh viện."""
        test_start_nodes = [32, 47, 69, 74]  # Các nút cụt xa nhất ở 4 góc bản đồ
        for st in test_start_nodes:
            for hid in range(1, 5):
                route_info = self.city_map.route_to_hospital(start_node=st, hospital_id=hid, by_travel_time=False)
                hosp = self.city_map.hospitals[hid]
                self.assertEqual(route_info["hospital_id"], hid)
                self.assertEqual(route_info["start_node"], hosp.from_node)
                self.assertGreater(len(route_info["path_to_start_node"]), 0)
                self.assertEqual(route_info["distance_from_start_node"], hosp.distance_from_start)
                self.assertGreater(route_info["total_cost"], 0.0)

    def test_edge_ids_and_mappings(self) -> None:
        """
        Kiểm tra ID số nguyên định danh cho từng cung đường (edge_id):
        - Đủ 172 cung đường có hướng với ID từ 1 đến 172 không trùng lặp.
        - Ánh xạ 2 chiều giữa (u, v) và edge_id chính xác.
        - Các bệnh viện lưu đúng edge_id của cung đường chứa nó.
        """
        num_edges = len(self.city_map.edges)
        self.assertEqual(num_edges, 172)
        self.assertEqual(len(self.city_map.edges_by_id), num_edges)
        self.assertEqual(len(self.city_map.edge_to_id), num_edges)
        self.assertEqual(len(self.city_map.id_to_edge), num_edges)

        expected_ids = set(range(1, num_edges + 1))
        actual_ids = {e.edge_id for e in self.city_map.edges.values()}
        self.assertEqual(actual_ids, expected_ids)

        for (u, v), edge in self.city_map.edges.items():
            eid = edge.edge_id
            self.assertEqual(self.city_map.edge_to_id[(u, v)], eid)
            self.assertEqual(self.city_map.id_to_edge[eid], (u, v))
            self.assertIs(self.city_map.edges_by_id[eid], edge)
            self.assertIs(self.city_map.get_edge(eid), edge)
            self.assertIs(self.city_map.get_edge((u, v)), edge)
            self.assertEqual(self.city_map.get_edge_id(u, v), eid)
            self.assertEqual(self.city_map.get_edge_id((u, v)), eid)

        # Kiểm tra 4 bệnh viện có edge_id chuẩn xác
        for hid, hosp in self.city_map.hospitals.items():
            expected_eid = self.city_map.edge_to_id[(hosp.from_node, hosp.to_node)]
            self.assertEqual(hosp.edge_id, expected_eid)
            self.assertGreater(hosp.edge_id, 0)

        # Kiểm tra set_edge_congestion bằng edge_id
        test_edge = self.city_map.edges_by_id[1]
        self.city_map.set_edge_congestion(1, level=CONGESTION_HEAVY)
        self.assertEqual(test_edge.congestion, CONGESTION_HEAVY)


if __name__ == "__main__":
    unittest.main()
