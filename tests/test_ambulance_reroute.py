import unittest
from types import SimpleNamespace

from dispatch_env.ambulance import Ambulance
from dispatch_env.types import AmbulanceStatus


class FakeCityMap:
    def __init__(self, new_eta):
        self.edges = {
            (1, 2): SimpleNamespace(length=10.0, speed=60.0),
            (3, 4): SimpleNamespace(length=10.0, speed=60.0),
            (5, 6): SimpleNamespace(length=10.0, speed=60.0),
        }
        self.new_eta = new_eta

    def route_from_road_to_road(self, **kwargs):
        return {
            "edges_path": [(1, 2), (5, 6), (3, 4)],
            "total_time_minutes": self.new_eta,
        }


def _ambulance():
    return Ambulance(
        ambulance_id=1,
        current_edge=(1, 2),
        distance_from_start=2.0,
        status=AmbulanceStatus.EN_ROUTE_TO_CALL,
        target_edge=(3, 4),
        target_dist=2.0,
        route_edges=[(1, 2), (3, 4)],
    )


class TestAmbulanceRerouteThreshold(unittest.TestCase):
    def test_reroutes_when_new_route_saves_at_least_25_percent(self):
        ambulance = _ambulance()

        changed = ambulance.reroute_if_beneficial(FakeCityMap(new_eta=7.5))

        self.assertTrue(changed)
        self.assertEqual(ambulance.reroute_count, 1)
        self.assertEqual(ambulance.route_edges, [(1, 2), (5, 6), (3, 4)])

    def test_keeps_current_route_when_saving_is_under_25_percent(self):
        ambulance = _ambulance()

        changed = ambulance.reroute_if_beneficial(FakeCityMap(new_eta=7.6))

        self.assertFalse(changed)
        self.assertEqual(ambulance.reroute_count, 0)
        self.assertEqual(ambulance.route_edges, [(1, 2), (3, 4)])


if __name__ == "__main__":
    unittest.main()
