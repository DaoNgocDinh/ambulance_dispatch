"""
Web GUI Server cho Hệ thống Mô phỏng Điều phối Xe Cứu thương Thông minh.
Cung cấp REST API và giao diện Web Dashboard tương tác thời gian thực.
Sử dụng thư viện chuẩn của Python (http.server), không yêu cầu cài đặt thêm dependencies.
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

# Thêm thư mục hiện tại vào sys.path để import các module của simulation
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from map_env import CityMap
from dispatch_env import (
    AmbulanceDispatchEnv,
    DEFAULT_MAX_OVERTIME,
    DEFAULT_MAX_PENDING_CALLS,
    DEFAULT_MAX_TIME,
    DEFAULT_NUM_AMBULANCES,
    DEFAULT_POISSON_LAMBDA,
    DEFAULT_SEED,
)
from dispatch_env.types import AmbulanceStatus, CallStatus


class SimulationManager:
    """Quản lý trạng thái mô phỏng tập trung và đồng bộ luồng."""

    def __init__(
        self,
        num_ambulances: int = DEFAULT_NUM_AMBULANCES,
        max_time: int = DEFAULT_MAX_TIME,
        max_overtime: int = DEFAULT_MAX_OVERTIME,
        poisson_lambda: float = DEFAULT_POISSON_LAMBDA,
        max_pending_calls: int = DEFAULT_MAX_PENDING_CALLS,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.lock = threading.Lock()
        self.config = {
            "num_ambulances": num_ambulances,
            "max_time": max_time,
            "max_overtime": max_overtime,
            "poisson_lambda": poisson_lambda,
            "max_pending_calls": max_pending_calls,
            "seed": seed,
        }
        self.history_logs: List[str] = []
        self._init_env()

    def _init_env(self) -> None:
        cfg = self.config
        city_map = CityMap(seed=cfg["seed"])
        self.env = AmbulanceDispatchEnv(
            city_map=city_map,
            num_ambulances=cfg["num_ambulances"],
            max_time=cfg["max_time"],
            max_overtime=cfg["max_overtime"],
            poisson_lambda=cfg["poisson_lambda"],
            max_pending_calls=cfg["max_pending_calls"],
            seed=cfg["seed"],
        )
        self.done: bool = False
        self.history_logs = [self.env.format_log_step()]

    def reset(self, new_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self.lock:
            if new_config:
                for k, v in new_config.items():
                    if k in self.config and v is not None:
                        if k in ("num_ambulances", "max_time", "max_overtime", "max_pending_calls", "seed"):
                            self.config[k] = int(v)
                        elif k == "poisson_lambda":
                            self.config[k] = float(v)
            self._init_env()
            return self.get_state_locked()

    def step(self, auto_dispatch: bool = True, action: Optional[Any] = None) -> Dict[str, Any]:
        with self.lock:
            if self.done:
                return self.get_state_locked()

            # Xử lý hành động thủ công nếu có
            if action is not None and not auto_dispatch:
                obs, reward, done, info = self.env.step(action=action, auto_dispatch=False)
            else:
                obs, reward, done, info = self.env.step(auto_dispatch=True)

            self.done = done
            log_str = self.env.format_log_step()
            self.history_logs.append(log_str)
            return self.get_state_locked()

    def step_multi(self, count: int = 10, auto_dispatch: bool = True) -> Dict[str, Any]:
        with self.lock:
            for _ in range(max(1, count)):
                if self.done:
                    break
                obs, reward, done, info = self.env.step(auto_dispatch=auto_dispatch)
                self.done = done
                self.history_logs.append(self.env.format_log_step())
            return self.get_state_locked()

    def get_static_map(self) -> Dict[str, Any]:
        with self.lock:
            cm = self.env.city_map
            nodes_data = []
            xs, ys = [], []
            for n_id, node in cm.nodes.items():
                xs.append(node.x)
                ys.append(node.y)
                nodes_data.append({
                    "id": n_id,
                    "x": node.x,
                    "y": node.y,
                    "degree": node.degree,
                    "type": node.intersection_type.value,
                    "name": node.name,
                    "is_central": n_id <= 20,
                    "neighbors": node.neighbors,
                })

            edges_data = []
            for (u, v), edge in cm.edges.items():
                u_node = cm.nodes[u]
                v_node = cm.nodes[v]
                edges_data.append({
                    "edge_id": edge.edge_id,
                    "u": u,
                    "v": v,
                    "length": edge.length,
                    "speed": edge.speed,
                    "congestion": edge.congestion,
                    "is_center": edge.is_center,
                    "x1": u_node.x,
                    "y1": u_node.y,
                    "x2": v_node.x,
                    "y2": v_node.y,
                })

            hospitals_data = []
            for h_id, hosp in cm.hospitals.items():
                hospitals_data.append({
                    "hospital_id": hosp.hospital_id,
                    "name": hosp.name,
                    "from_node": hosp.from_node,
                    "to_node": hosp.to_node,
                    "road_length": hosp.road_length,
                    "distance_from_start": hosp.distance_from_start,
                    "coord": hosp.coord,
                    "handover_time": hosp.handover_time,
                    "edge_id": hosp.edge_id,
                })

            return {
                "nodes": nodes_data,
                "edges": edges_data,
                "hospitals": hospitals_data,
                "bounds": {
                    "min_x": min(xs) if xs else 0.0,
                    "max_x": max(xs) if xs else 35.0,
                    "min_y": min(ys) if ys else 0.0,
                    "max_y": max(ys) if ys else 32.0,
                },
                "config": dict(self.config),
                "num_potential_actions": self.env.num_potential_actions,
            }

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return self.get_state_locked()

    def get_state_locked(self) -> Dict[str, Any]:
        env = self.env
        cm = env.city_map

        # Danh sách xe
        ambulances_data = []
        for aid in range(1, env.num_ambulances + 1):
            amb = env.ambulances[aid]
            coord = amb.get_coord(cm)
            cur_speed = amb.get_current_speed(cm)
            road_speed = amb.get_road_speed(cm)

            # Tính toán tọa độ đường đi (polyline) để vẽ đường dẫn tới mục tiêu
            route_coords = []
            if amb.route_edges:
                # Điểm hiện tại
                route_coords.append(list(coord))
                # Các nút trung gian
                for u_e, v_e in amb.route_edges:
                    node_v = cm.nodes[v_e]
                    route_coords.append([node_v.x, node_v.y])

            ambulances_data.append({
                "id": amb.ambulance_id,
                "status": amb.status.value,
                "status_code": amb.status.name,
                "coord": list(coord),
                "current_edge": list(amb.current_edge),
                "current_edge_id": cm.edge_to_id.get(amb.current_edge, 0),
                "distance_from_start": round(amb.distance_from_start, 2),
                "current_speed": round(cur_speed, 1),
                "road_speed": round(road_speed, 1),
                "current_call_id": amb.current_call_id,
                "target_type": amb.target_type,
                "target_id": amb.target_id,
                "target_name": amb.target_name,
                "assigned_hospital_id": amb.assigned_hospital_id,
                "scene_time_remaining": round(amb.scene_time_remaining, 1),
                "hospital_time_remaining": round(amb.hospital_time_remaining, 1),
                "total_distance_traveled": round(amb.total_distance_traveled, 2),
                "route_edges": [list(e) for e in amb.route_edges],
                "route_coords": route_coords,
            })

        # Danh sách cuộc gọi: active + các cuộc gọi sinh trong bước này
        calls_data = []
        for cid, call in env.calls.items():
            # Chỉ hiển thị các cuộc gọi đang active hoặc vừa mới xong/thất bại gần đây
            is_active = call.status in (
                CallStatus.PENDING,
                CallStatus.WAITING_FOR_AMBULANCE,
                CallStatus.AT_SCENE,
            )
            is_recent_event = (
                call in env.calls_completed_this_step
                or call in env.calls_failed_this_step
                or call in env.calls_generated_this_step
            )

            if is_active or is_recent_event:
                c_edge_id = call.edge_id if call.edge_id != 0 else cm.edge_to_id.get(call.edge, 0)
                calls_data.append({
                    "id": call.call_id,
                    "call_id": call.call_id,
                    "severity": call.severity,
                    "edge": list(call.edge),
                    "edge_id": c_edge_id,
                    "distance_from_start": round(call.distance_from_start, 2),
                    "coord": list(call.coord),
                    "status": call.status.value,
                    "status_code": call.status.name,
                    "created_time": call.created_time,
                    "failure_time": call.failure_time,
                    "processing_time": call.processing_time,
                    "wait_time": round(call.wait_time, 2),
                    "deadline": call.deadline,
                    "time_to_deadline": max(0.0, round(call.deadline - env.current_minute, 1)),
                    "assigned_ambulance_id": call.assigned_ambulance_id,
                    "arrived_at_scene_time": call.arrived_at_scene_time,
                    "is_perfect": call.is_perfect,
                    "failure_reason": call.failure_reason,
                })

        # 3 Slots
        slots_data = []
        for s_idx, c in enumerate(env.get_call_slots()):
            if c is not None:
                c_edge_id = c.edge_id if c.edge_id != 0 else cm.edge_to_id.get(c.edge, 0)
                slots_data.append({
                    "slot_index": s_idx,
                    "call_id": c.call_id,
                    "severity": c.severity,
                    "edge": list(c.edge),
                    "edge_id": c_edge_id,
                    "coord": list(c.coord),
                    "created_time": c.created_time,
                    "deadline": c.deadline,
                    "wait_time": round(c.wait_time, 2),
                    "time_to_deadline": max(0.0, round(c.deadline - env.current_minute, 1)),
                })
            else:
                slots_data.append(None)

        # Trạng thái tắc đường
        edge_congestion = {
            edge.edge_id: edge.congestion
            for edge in cm.edges_by_id.values()
        }

        # Mask và hành động hợp lệ
        valid_actions_summary = []
        for pair_list in env.get_valid_actions():
            if pair_list:
                valid_actions_summary.append(pair_list)

        return {
            "current_minute": env.current_minute,
            "traffic_time_step": env.traffic_time_step,
            "max_time": env.max_time,
            "max_overtime": env.max_overtime,
            "done": self.done,
            "is_truncated": env.is_truncated,
            "stats": {
                "served_calls": env.served_calls_count,
                "failed_calls": env.failed_calls_count,
                "perfect_calls": env.perfect_calls_count,
                "total_calls": len(env.calls),
                "pending_count": len(env.pending_calls),
                "active_count": len(env.active_calls),
                "total_reward": round(env.total_reward, 2),
                "last_step_reward": round(env.last_step_reward, 2),
                "reward_breakdown": {
                    k: round(v, 2) for k, v in env.last_reward_breakdown.items()
                },
                "success_rate": round(
                    (env.served_calls_count / max(1, env.served_calls_count + env.failed_calls_count)) * 100, 1
                ),
            },
            "ambulances": ambulances_data,
            "calls": calls_data,
            "slots": slots_data,
            "edge_congestion": edge_congestion,
            "traffic_transitions": dict(cm.last_traffic_transitions),
            "recent_log": self.history_logs[-1] if self.history_logs else "",
            "last_action_id": env.last_action_id,
            "last_decoded_action": [list(p) for p in env.last_decoded_action],
            "valid_action_count": env.last_valid_action_count,
            "valid_action_samples": valid_actions_summary[:10],
        }

    def get_full_log(self) -> str:
        with self.lock:
            return "\n\n".join(self.history_logs)


# Khởi tạo instance toàn cục của simulation
sim_manager = SimulationManager()


class DispatchRequestHandler(BaseHTTPRequestHandler):
    """Xử lý yêu cầu HTTP phục vụ giao diện và REST API."""

    def log_message(self, format: str, *args: Any) -> None:
        # Giảm bớt log terminal ồn ào khi polling
        pass

    def _set_headers(self, content_type: str = "application/json", status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self._set_headers(status=HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == "/api/map":
            data = sim_manager.get_static_map()
            self._set_headers("application/json")
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
            return

        if path == "/api/state":
            data = sim_manager.get_state()
            self._set_headers("application/json")
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
            return

        if path == "/api/log_export":
            full_log = sim_manager.get_full_log()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Disposition", "attachment; filename=simulation_full.log")
            self.end_headers()
            self.wfile.write(full_log.encode("utf-8"))
            return

        # Phục vụ static files (HTML, CSS, JS) trong thư mục web/
        web_dir = os.path.join(CURRENT_DIR, "web")
        if path == "/" or path == "/index.html":
            file_path = os.path.join(web_dir, "index.html")
        else:
            clean_path = path.lstrip("/")
            file_path = os.path.join(web_dir, clean_path)

        if os.path.exists(file_path) and os.path.isfile(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            content_type = mime_type or "application/octet-stream"
            with open(file_path, "rb") as f:
                content = f.read()
            self._set_headers(content_type)
            self.wfile.write(content)
        else:
            self._set_headers("text/plain", 404)
            self.wfile.write(b"404 Not Found")

    def do_POST(self) -> None:
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            req_data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            req_data = {}

        if path == "/api/step":
            auto_dispatch = req_data.get("auto_dispatch", True)
            action = req_data.get("action", None)
            res = sim_manager.step(auto_dispatch=auto_dispatch, action=action)
            self._set_headers("application/json")
            self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
            return

        if path == "/api/step_multi":
            count = int(req_data.get("count", 10))
            auto_dispatch = req_data.get("auto_dispatch", True)
            res = sim_manager.step_multi(count=count, auto_dispatch=auto_dispatch)
            self._set_headers("application/json")
            self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
            return

        if path == "/api/reset":
            config = req_data.get("config", {})
            res = sim_manager.reset(new_config=config)
            self._set_headers("application/json")
            self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
            return

        if path == "/api/manual_dispatch":
            # Phân công thủ công 1 xe cho 1 cuộc gọi:
            # req_data: {"ambulance_id": 1, "call_id": 10}
            amb_id = int(req_data.get("ambulance_id", 0))
            call_id = int(req_data.get("call_id", 0))
            action = [(amb_id, call_id)] if amb_id and call_id else []
            res = sim_manager.step(auto_dispatch=False, action=action)
            self._set_headers("application/json")
            self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
            return

        self._set_headers("text/plain", 404)
        self.wfile.write(b"404 Not Found")


def run_web_server(port: int = 8000, open_browser: bool = True) -> None:
    """Khởi động Web Server tại cổng chỉ định."""
    server_address = ("", port)
    try:
        httpd = ThreadingHTTPServer(server_address, DispatchRequestHandler)
    except OSError:
        # Nếu cổng 8000 bận, thử 8080 hoặc cổng ngẫu nhiên kế tiếp
        port = 8080
        server_address = ("", port)
        httpd = ThreadingHTTPServer(server_address, DispatchRequestHandler)

    url = f"http://localhost:{port}"
    print("=" * 75)
    print("🚑 HỆ THỐNG GIAO DIỆN ĐIỀU PHỐI XE CỨU THƯƠNG ĐANG CHẠY")
    print(f"👉 Truy cập giao diện tại: {url}")
    print("👉 Nhấn Ctrl+C trong cửa sổ lệnh để dừng máy chủ.")
    print("=" * 75)

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] Đang tắt máy chủ giao diện...")
        httpd.server_close()


if __name__ == "__main__":
    port_arg = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port_arg = int(sys.argv[1])
    run_web_server(port=port_arg, open_browser=True)

