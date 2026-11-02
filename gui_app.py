"""
Ứng dụng Giao diện Desktop Tkinter cho Mô phỏng Điều phối Xe Cứu thương.
Chạy trực tiếp không phụ thuộc trình duyệt, sử dụng thư viện chuẩn Tkinter và Math.
"""

from __future__ import annotations

import math
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Optional, Tuple, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

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


class AmbulanceDispatchDesktopApp(tk.Tk):
    """Cửa sổ ứng dụng desktop điều phối xe cứu thương."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Hệ Thống Điều Phối Cấp Cứu Thông Minh - MDP 80 Nút (Desktop GUI)")
        self.geometry("1400x850")
        self.minsize(1100, 700)
        self.configure(bg="#0f172a")

        # Khởi tạo mô phỏng
        self.seed = DEFAULT_SEED
        self.num_ambulances = DEFAULT_NUM_AMBULANCES
        self.max_time = DEFAULT_MAX_TIME
        self.max_overtime = DEFAULT_MAX_OVERTIME
        self.poisson_lambda = DEFAULT_POISSON_LAMBDA
        self.max_pending_calls = DEFAULT_MAX_PENDING_CALLS

        self.city_map = CityMap(seed=self.seed)
        self.env = AmbulanceDispatchEnv(
            city_map=self.city_map,
            num_ambulances=self.num_ambulances,
            max_time=self.max_time,
            max_overtime=self.max_overtime,
            poisson_lambda=self.poisson_lambda,
            max_pending_calls=self.max_pending_calls,
            seed=self.seed,
        )

        self.is_running = False
        self.speed_ms = 400
        self.done = False

        # Tọa độ và pan/zoom
        self.pan_x = 0
        self.pan_y = 0
        self.zoom = 1.0
        self.drag_start_x = 0
        self.drag_start_y = 0

        self._calculate_bounds()
        self._build_ui()
        self.redraw_canvas()
        self.update_stats_panel()

    def _calculate_bounds(self) -> None:
        xs = [n.x for n in self.city_map.nodes.values()]
        ys = [n.y for n in self.city_map.nodes.values()]
        self.min_x = min(xs)
        self.max_x = max(xs)
        self.min_y = min(ys)
        self.max_y = max(ys)
        self.w_width = self.max_x - self.min_x
        self.w_height = self.max_y - self.min_y

    def _build_ui(self) -> None:
        # 1. Header Toolbar
        header_frame = tk.Frame(self, bg="#1e293b", height=50)
        header_frame.pack(fill=tk.X, side=tk.TOP, padx=0, pady=0)

        title_lbl = tk.Label(
            header_frame,
            text="🚑 ĐIỀU PHỐI CẤP CỨU ĐÔ THỊ (80 NÚT, 4 BỆNH VIỆN)",
            font=("Segoe UI", 12, "bold"),
            fg="#38bdf8",
            bg="#1e293b",
            padx=12,
        )
        title_lbl.pack(side=tk.LEFT)

        self.time_lbl = tk.Label(
            header_frame,
            text="Phút: 0 / 360",
            font=("Segoe UI", 12, "bold"),
            fg="#fbbf24",
            bg="#1e293b",
            padx=16,
        )
        self.time_lbl.pack(side=tk.LEFT)

        self.status_lbl = tk.Label(
            header_frame,
            text="● SẴN SÀNG",
            font=("Segoe UI", 10, "bold"),
            fg="#34d399",
            bg="#1e293b",
            padx=10,
        )
        self.status_lbl.pack(side=tk.LEFT)

        # 2. Main Paned Window
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#0f172a", sashwidth=4)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Left: Map Canvas & Toolbar
        left_frame = tk.Frame(main_pane, bg="#0b0f17")
        main_pane.add(left_frame, minsize=650)

        # Controls Bar
        ctrl_bar = tk.Frame(left_frame, bg="#1e293b", pady=6, padx=8)
        ctrl_bar.pack(fill=tk.X, side=tk.TOP)

        self.play_btn = tk.Button(
            ctrl_bar,
            text="▶ Bắt đầu (Play)",
            font=("Segoe UI", 9, "bold"),
            bg="#2563eb",
            fg="white",
            relief=tk.FLAT,
            padx=10,
            command=self.toggle_play,
        )
        self.play_btn.pack(side=tk.LEFT, padx=4)

        step_btn = tk.Button(
            ctrl_bar,
            text="⏭ +1 Phút",
            font=("Segoe UI", 9),
            bg="#334155",
            fg="white",
            relief=tk.FLAT,
            padx=8,
            command=lambda: self.step_sim(1),
        )
        step_btn.pack(side=tk.LEFT, padx=4)

        step10_btn = tk.Button(
            ctrl_bar,
            text="⏩ +10 Phút",
            font=("Segoe UI", 9),
            bg="#334155",
            fg="white",
            relief=tk.FLAT,
            padx=8,
            command=lambda: self.step_sim(10),
        )
        step10_btn.pack(side=tk.LEFT, padx=4)

        reset_btn = tk.Button(
            ctrl_bar,
            text="🔄 Khởi động lại",
            font=("Segoe UI", 9),
            bg="#dc2626",
            fg="white",
            relief=tk.FLAT,
            padx=8,
            command=self.reset_sim,
        )
        reset_btn.pack(side=tk.LEFT, padx=4)

        # Tốc độ
        speed_lbl = tk.Label(ctrl_bar, text="Tốc độ:", bg="#1e293b", fg="#94a3b8")
        speed_lbl.pack(side=tk.LEFT, padx=(16, 4))

        self.speed_scale = tk.Scale(
            ctrl_bar,
            from_=50,
            to=1000,
            orient=tk.HORIZONTAL,
            showvalue=False,
            bg="#1e293b",
            fg="white",
            highlightthickness=0,
            length=100,
            command=self.on_speed_change,
        )
        self.speed_scale.set(400)
        self.speed_scale.pack(side=tk.LEFT)

        # Zoom buttons
        zoom_in_btn = tk.Button(
            ctrl_bar,
            text="🔍 +",
            font=("Segoe UI", 9),
            bg="#334155",
            fg="white",
            relief=tk.FLAT,
            padx=6,
            command=lambda: self.adjust_zoom(1.2),
        )
        zoom_in_btn.pack(side=tk.RIGHT, padx=2)

        zoom_out_btn = tk.Button(
            ctrl_bar,
            text="🔍 -",
            font=("Segoe UI", 9),
            bg="#334155",
            fg="white",
            relief=tk.FLAT,
            padx=6,
            command=lambda: self.adjust_zoom(0.8),
        )
        zoom_out_btn.pack(side=tk.RIGHT, padx=2)

        zoom_fit_btn = tk.Button(
            ctrl_bar,
            text="⛶ Vừa",
            font=("Segoe UI", 9),
            bg="#334155",
            fg="white",
            relief=tk.FLAT,
            padx=6,
            command=self.reset_zoom,
        )
        zoom_fit_btn.pack(side=tk.RIGHT, padx=2)

        # Map Canvas
        self.canvas = tk.Canvas(left_frame, bg="#06090e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<MouseWheel>", self.on_canvas_wheel)
        self.canvas.bind("<Configure>", lambda e: self.redraw_canvas())

        # Right: Stats, Fleet, Slots, Log (Notebook / Tabs)
        right_frame = tk.Frame(main_pane, bg="#1e293b", width=420)
        main_pane.add(right_frame, minsize=380)

        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Tab 1: Dashboard & Fleet
        tab_dash = tk.Frame(notebook, bg="#1e293b")
        notebook.add(tab_dash, text="🚑 Đội Xe & Hàng Đợi")

        # KPI Panel
        kpi_frame = tk.LabelFrame(
            tab_dash,
            text="CHỈ SỐ THỰC THI (KPIS)",
            bg="#1e293b",
            fg="#38bdf8",
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=6,
        )
        kpi_frame.pack(fill=tk.X, padx=6, pady=4)

        self.kpi_text = tk.Label(
            kpi_frame,
            text="Đang tính toán...",
            font=("Segoe UI", 10),
            bg="#1e293b",
            fg="#f1f5f9",
            justify=tk.LEFT,
        )
        self.kpi_text.pack(fill=tk.X)

        # Slots Panel
        slots_frame = tk.LabelFrame(
            tab_dash,
            text="HÀNG ĐỢI CUỘC GỌI (3 SLOTS CỐ ĐỊNH)",
            bg="#1e293b",
            fg="#38bdf8",
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=6,
        )
        slots_frame.pack(fill=tk.X, padx=6, pady=4)

        self.slots_text = tk.Label(
            slots_frame,
            text="Trống",
            font=("Segoe UI", 9),
            bg="#1e293b",
            fg="#94a3b8",
            justify=tk.LEFT,
        )
        self.slots_text.pack(fill=tk.X)

        # Fleet Treeview
        fleet_frame = tk.LabelFrame(
            tab_dash,
            text="DANH SÁCH XE CỨU THƯƠNG",
            bg="#1e293b",
            fg="#38bdf8",
            font=("Segoe UI", 9, "bold"),
            padx=4,
            pady=4,
        )
        fleet_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        columns = ("id", "status", "edge", "target")
        self.fleet_tree = ttk.Treeview(fleet_frame, columns=columns, show="headings", height=6)
        self.fleet_tree.heading("id", text="Xe")
        self.fleet_tree.heading("status", text="Trạng thái")
        self.fleet_tree.heading("edge", text="Vị trí")
        self.fleet_tree.heading("target", text="Đích đến")

        self.fleet_tree.column("id", width=50, anchor="center")
        self.fleet_tree.column("status", width=110, anchor="w")
        self.fleet_tree.column("edge", width=90, anchor="w")
        self.fleet_tree.column("target", width=120, anchor="w")
        self.fleet_tree.pack(fill=tk.BOTH, expand=True)

        # Tab 2: Logs
        tab_log = tk.Frame(notebook, bg="#1e293b")
        notebook.add(tab_log, text="📜 Nhật Ký Chi Tiết")

        self.log_box = tk.Text(
            tab_log,
            bg="#0b0f17",
            fg="#cbd5e1",
            font=("Consolas", 9),
            wrap=tk.WORD,
            padx=8,
            pady=8,
        )
        self.log_box.pack(fill=tk.BOTH, expand=True)

    def world_to_screen(self, wx: float, wy: float) -> Tuple[float, float]:
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            cw, ch = 800, 600

        padding = 50
        scale = min((cw - padding * 2) / self.w_width, (ch - padding * 2) / self.w_height) * self.zoom

        # Inverted Y
        sx = cw / 2 + (wx - (self.min_x + self.w_width / 2)) * scale + self.pan_x
        sy = ch / 2 - (wy - (self.min_y + self.w_height / 2)) * scale + self.pan_y
        return sx, sy

    def adjust_zoom(self, factor: float) -> None:
        self.zoom = max(0.4, min(4.0, self.zoom * factor))
        self.redraw_canvas()

    def reset_zoom(self) -> None:
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.redraw_canvas()

    def on_canvas_press(self, event: Any) -> None:
        self.drag_start_x = event.x - self.pan_x
        self.drag_start_y = event.y - self.pan_y

    def on_canvas_drag(self, event: Any) -> None:
        self.pan_x = event.x - self.drag_start_x
        self.pan_y = event.y - self.drag_start_y
        self.redraw_canvas()

    def on_canvas_wheel(self, event: Any) -> None:
        factor = 1.15 if event.delta > 0 else 0.85
        self.adjust_zoom(factor)

    def on_speed_change(self, val: str) -> None:
        self.speed_ms = 1050 - int(val)

    def redraw_canvas(self) -> None:
        self.canvas.delete("all")

        # 1. Vẽ các tuyến đường (172 cung đường có hướng)
        for (u, v), edge in self.city_map.edges.items():
            u_node = self.city_map.nodes[u]
            v_node = self.city_map.nodes[v]
            x1, y1 = self.world_to_screen(u_node.x, u_node.y)
            x2, y2 = self.world_to_screen(v_node.x, v_node.y)

            dx = x2 - x1
            dy = y2 - y1
            dist = math.hypot(dx, dy)
            if dist < 1:
                continue

            nx = -dy / dist * 3.0
            ny = dx / dist * 3.0

            color = "#10b981"
            if edge.congestion == 2:
                color = "#f59e0b"
            elif edge.congestion == 3:
                color = "#ef4444"

            self.canvas.create_line(
                x1 + nx,
                y1 + ny,
                x2 + nx,
                y2 + ny,
                fill=color,
                width=2,
                arrow=tk.LAST,
                arrowshape=(8, 10, 3),
            )

        # 2. Vẽ các nút
        for n_id, node in self.city_map.nodes.items():
            sx, sy = self.world_to_screen(node.x, node.y)
            is_central = n_id <= 20
            r = 5 if is_central else 3.5
            fill_col = "#a855f7" if is_central else "#475569"

            self.canvas.create_oval(
                sx - r, sy - r, sx + r, sy + r,
                fill=fill_col, outline="#cbd5e1", width=1
            )
            if self.zoom >= 0.8 or is_central:
                self.canvas.create_text(
                    sx, sy - r - 5,
                    text=str(n_id),
                    fill="#94a3b8",
                    font=("Consolas", 8)
                )

        # 3. Vẽ 4 Bệnh viện
        for hosp in self.city_map.hospitals.values():
            sx, sy = self.world_to_screen(hosp.coord[0], hosp.coord[1])
            self.canvas.create_oval(
                sx - 9, sy - 9, sx + 9, sy + 9,
                fill="#ef4444", outline="white", width=2
            )
            self.canvas.create_text(
                sx, sy,
                text="+",
                fill="white",
                font=("Segoe UI", 11, "bold")
            )
            self.canvas.create_text(
                sx, sy + 13,
                text=f"BV {hosp.hospital_id}",
                fill="#fca5a5",
                font=("Segoe UI", 8, "bold")
            )

        # 4. Vẽ các cuộc gọi đang chờ / xử lý
        for call in self.env.active_calls:
            sx, sy = self.world_to_screen(call.coord[0], call.coord[1])
            sev_col = "#fbbf24" if call.severity == 1 else ("#fb923c" if call.severity == 2 else "#ef4444")
            self.canvas.create_oval(
                sx - 8, sy - 8, sx + 8, sy + 8,
                fill=sev_col, outline="white", width=1
            )
            self.canvas.create_text(
                sx, sy,
                text=f"C{call.call_id}",
                fill="#0f172a",
                font=("Consolas", 7, "bold")
            )

        # 5. Vẽ Xe Cứu Thương
        for amb in self.env.ambulances.values():
            coord = amb.get_coord(self.city_map)
            sx, sy = self.world_to_screen(coord[0], coord[1])

            amb_col = "#06b6d4"  # Idle
            if amb.status == AmbulanceStatus.EN_ROUTE_TO_CALL:
                amb_col = "#f59e0b"
            elif amb.status == AmbulanceStatus.AT_SCENE:
                amb_col = "#a855f7"
            elif amb.status == AmbulanceStatus.EN_ROUTE_TO_HOSPITAL:
                amb_col = "#f43f5e"
            elif amb.status == AmbulanceStatus.AT_HOSPITAL:
                amb_col = "#10b981"

            self.canvas.create_oval(
                sx - 9, sy - 9, sx + 9, sy + 9,
                fill=amb_col, outline="white", width=2
            )
            self.canvas.create_text(
                sx, sy,
                text=str(amb.ambulance_id),
                fill="white",
                font=("Consolas", 8, "bold")
            )

    def update_stats_panel(self) -> None:
        t = self.env.current_minute
        self.time_lbl.config(text=f"Phút: {t} / {self.max_time}")

        if t > self.max_time:
            self.status_lbl.config(text=f"● QUÁ GIỜ (+{t - self.max_time}p)", fg="#f59e0b")
        elif self.done:
            self.status_lbl.config(text="● HOÀN THÀNH" if not self.env.is_truncated else "● DỪNG", fg="#38bdf8")
        elif self.is_running:
            self.status_lbl.config(text="● ĐANG CHẠY", fg="#34d399")
        else:
            self.status_lbl.config(text="● TẠM DỪNG", fg="#94a3b8")

        # KPIs
        served = self.env.served_calls_count
        failed = self.env.failed_calls_count
        perfect = self.env.perfect_calls_count
        reward = self.env.total_reward
        total = served + failed
        rate = (served / total * 100) if total > 0 else 0.0

        kpi_str = (
            f"✔ Hoàn thành: {served} ca (Hoàn hảo: {perfect})\n"
            f"✘ Thất bại: {failed} ca\n"
            f"🎯 Tỉ lệ thành công: {rate:.1f}%\n"
            f"💰 Tổng Reward: {reward:.2f}\n"
            f"⚡ Đang chờ (Pending): {len(self.env.pending_calls)} / {self.max_pending_calls}"
        )
        self.kpi_text.config(text=kpi_str)

        # Slots
        slots = self.env.get_call_slots()
        slot_lines = []
        for i, s in enumerate(slots):
            if s is not None:
                rem = max(0.0, s.deadline - t)
                slot_lines.append(f"Slot {i}: Cuộc gọi #{s.call_id} (Mức {s.severity}, còn {rem:.1f}p)")
            else:
                slot_lines.append(f"Slot {i}: [Trống]")
        self.slots_text.config(text="\n".join(slot_lines))

        # Fleet Treeview
        for item in self.fleet_tree.get_children():
            self.fleet_tree.delete(item)

        for amb in self.env.ambulances.values():
            u, v = amb.current_edge
            pos = f"N{u}->N{v}"
            target = amb.target_name if amb.target_name else "Rảnh"
            self.fleet_tree.insert("", tk.END, values=(f"Xe {amb.ambulance_id}", amb.status.value, pos, target))

        # Log
        log_text = self.env.format_log_step()
        self.log_box.delete("1.0", tk.END)
        self.log_box.insert(tk.END, log_text)

    def step_sim(self, steps: int = 1) -> None:
        if self.done:
            return

        for _ in range(steps):
            if self.done:
                break
            obs, r, done, info = self.env.step(auto_dispatch=True)
            self.done = done

        self.redraw_canvas()
        self.update_stats_panel()

        if self.done and self.is_running:
            self.toggle_play()

    def play_loop(self) -> None:
        if not self.is_running:
            return
        self.step_sim(1)
        if self.is_running and not self.done:
            self.after(self.speed_ms, self.play_loop)

    def toggle_play(self) -> None:
        if self.done:
            self.reset_sim()

        self.is_running = not self.is_running
        if self.is_running:
            self.play_btn.config(text="⏸ Tạm dừng", bg="#10b981")
            self.play_loop()
        else:
            self.play_btn.config(text="▶ Bắt đầu (Play)", bg="#2563eb")
            self.update_stats_panel()

    def reset_sim(self) -> None:
        self.is_running = False
        self.play_btn.config(text="▶ Bắt đầu (Play)", bg="#2563eb")
        self.city_map = CityMap(seed=self.seed)
        self.env = AmbulanceDispatchEnv(
            city_map=self.city_map,
            num_ambulances=self.num_ambulances,
            max_time=self.max_time,
            max_overtime=self.max_overtime,
            poisson_lambda=self.poisson_lambda,
            max_pending_calls=self.max_pending_calls,
            seed=self.seed,
        )
        self.done = False
        self.redraw_canvas()
        self.update_stats_panel()


def run_desktop_app() -> None:
    app = AmbulanceDispatchDesktopApp()
    app.mainloop()


if __name__ == "__main__":
    run_desktop_app()

