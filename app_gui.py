"""
Desktop GUI (Tkinter) for Smart City Ambulance Dispatch MDP Simulation.
Native desktop window with zero external dependencies.
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import numpy as np

from environment import (
    AMB_STATE_IDLE,
    AMB_STATE_NAMES,
    AMB_STATE_ON_SCENE,
    AMB_STATE_RELOCATING,
    AMB_STATE_TO_HOSPITAL,
    AMB_STATE_TO_SCENE,
    EnvConfig,
    PRIO_NAMES,
    PRIO_RED,
    PRIO_YELLOW,
    PRIO_GREEN,
    TRAFFIC_BUSY,
    TRAFFIC_CLEAR,
    TRAFFIC_CONGESTED,
    TRAFFIC_NAMES,
    EmergencyRequest,
    SmartCityAmbulanceEnv,
)


class AmbulanceDispatchDesktopGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("🚑 Smart City Ambulance Dispatch - Điều Phối Xe Cứu Thương MDP")
        self.root.geometry("1260x820")
        self.root.minsize(1050, 700)
        self.root.configure(bg="#0b1120")

        # Khởi tạo mô phỏng
        self.seed = 42
        self.max_steps = 60
        self.env = SmartCityAmbulanceEnv(config=EnvConfig(max_steps=self.max_steps), seed=self.seed)
        self.env.reset(seed=self.seed)

        self.is_playing = False
        self.play_speed_ms = 700
        self.play_job = None
        self.policy_mode = "baseline"  # baseline, random, manual

        # Tọa độ 15 nút trên Canvas (Grid 3x5)
        self.node_coords: Dict[int, Tuple[int, int]] = {
            1:  (90, 80),   2:  (240, 80),  3:  (400, 80),  4:  (560, 80),  5:  (710, 80),
            6:  (90, 240),  7:  (240, 240), 8:  (400, 240), 9:  (560, 240), 10: (710, 240),
            11: (90, 400), 12:  (240, 400), 13: (400, 400), 14: (560, 400), 15: (710, 400),
        }

        self._init_theme()
        self._build_ui()
        self._refresh_ui()

    def _init_theme(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(".", background="#0b1120", foreground="#f8fafc")
        style.configure("TFrame", background="#0b1120")
        style.configure("Card.TFrame", background="#131e36", relief="solid", borderwidth=1)
        style.configure("TLabel", background="#131e36", foreground="#f8fafc", font=("Segoe UI", 9))
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground="#38bdf8")
        style.configure("Title.TLabel", font=("Segoe UI", 13, "bold"), foreground="#ffffff", background="#0b1120")
        style.configure("Sub.TLabel", font=("Segoe UI", 9), foreground="#94a3b8", background="#0b1120")

        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=6)
        style.map("Primary.TButton", background=[("active", "#0284c7"), ("!disabled", "#0369a1")], foreground=[("!disabled", "#ffffff")])
        style.map("Success.TButton", background=[("active", "#059669"), ("!disabled", "#047857")], foreground=[("!disabled", "#ffffff")])
        style.map("Danger.TButton", background=[("active", "#dc2626"), ("!disabled", "#b91c1c")], foreground=[("!disabled", "#ffffff")])
        style.map("Warning.TButton", background=[("active", "#d97706"), ("!disabled", "#b45309")], foreground=[("!disabled", "#ffffff")])

        style.configure("TCombobox", fieldbackground="#1e293b", background="#334155", foreground="#ffffff")

    def _build_ui(self) -> None:
        # Top Header Bar
        top_bar = tk.Frame(self.root, bg="#0f172a", height=60, bd=1, relief="ridge")
        top_bar.pack(side="top", fill="x", padx=0, pady=0)

        title_lbl = tk.Label(
            top_bar,
            text="🚑 ĐIỀU PHỐI XE CỨU THƯƠNG THÀNH PHỐ THÔNG MINH (MDP)",
            font=("Segoe UI", 13, "bold"),
            bg="#0f172a",
            fg="#ffffff",
        )
        title_lbl.pack(side="left", padx=16, pady=6)

        self.lbl_rush_hour = tk.Label(
            top_bar,
            text="🟢 Giờ bình thường",
            font=("Segoe UI", 10, "bold"),
            bg="#064e3b",
            fg="#34d399",
            padx=12,
            pady=4,
            bd=1,
            relief="solid",
        )
        self.lbl_rush_hour.pack(side="right", padx=16, pady=8)

        self.lbl_step = tk.Label(
            top_bar,
            text="⏱ Bước: 0 / 60",
            font=("Segoe UI", 10, "bold"),
            bg="#1e293b",
            fg="#38bdf8",
            padx=12,
            pady=4,
            bd=1,
            relief="solid",
        )
        self.lbl_step.pack(side="right", padx=8, pady=8)

        self.lbl_total_reward = tk.Label(
            top_bar,
            text="⭐ Reward: +0.0",
            font=("Segoe UI", 10, "bold"),
            bg="#1e293b",
            fg="#34d399",
            padx=12,
            pady=4,
            bd=1,
            relief="solid",
        )
        self.lbl_total_reward.pack(side="right", padx=8, pady=8)

        # Main Body Splitter
        body = tk.Frame(self.root, bg="#0b1120")
        body.pack(side="top", fill="both", expand=True, padx=12, pady=10)

        # Left Container: Map & Toolbar
        left_container = tk.Frame(body, bg="#0b1120")
        left_container.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Control Toolbar
        toolbar = tk.Frame(left_container, bg="#131e36", bd=1, relief="solid")
        toolbar.pack(side="top", fill="x", pady=(0, 10), ipady=6, ipadx=8)

        self.btn_play = ttk.Button(toolbar, text="▶ Tự Động Chạy", style="Success.TButton", command=self._toggle_play)
        self.btn_play.pack(side="left", padx=4)

        self.btn_step = ttk.Button(toolbar, text="⏭ Bước Kế (Step)", style="Primary.TButton", command=self._step_once)
        self.btn_step.pack(side="left", padx=4)

        self.btn_reset = ttk.Button(toolbar, text="🔄 Đặt Lại (Reset)", style="Danger.TButton", command=self._reset_env)
        self.btn_reset.pack(side="left", padx=4)

        tk.Label(toolbar, text="Chính sách:", bg="#131e36", fg="#94a3b8", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(12, 4))
        self.combo_policy = ttk.Combobox(
            toolbar,
            values=["Baseline (Gần nhất)", "Ngẫu nhiên (Random)", "Thủ công (Manual)"],
            state="readonly",
            width=18,
        )
        self.combo_policy.current(0)
        self.combo_policy.bind("<<ComboboxSelected>>", self._on_policy_change)
        self.combo_policy.pack(side="left", padx=4)

        tk.Label(toolbar, text="Tốc độ:", bg="#131e36", fg="#94a3b8", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(12, 4))
        self.combo_speed = ttk.Combobox(
            toolbar,
            values=["Chậm (1.2s)", "Vừa (0.7s)", "Nhanh (0.35s)"],
            state="readonly",
            width=13,
        )
        self.combo_speed.current(1)
        self.combo_speed.bind("<<ComboboxSelected>>", self._on_speed_change)
        self.combo_speed.pack(side="left", padx=4)

        # Canvas for Map
        map_frame = tk.Frame(left_container, bg="#131e36", bd=1, relief="solid")
        map_frame.pack(side="top", fill="both", expand=True)

        map_title = tk.Label(
            map_frame,
            text="🗺 SƠ ĐỒ MẠNG LƯỚI ĐÔ THỊ 15 NÚT & 5 TUYẾN ĐƯỜNG HUYẾT MẠCH",
            font=("Segoe UI", 9, "bold"),
            bg="#131e36",
            fg="#94a3b8",
            padx=10,
            pady=6,
        )
        map_title.pack(side="top", anchor="w")

        self.canvas = tk.Canvas(map_frame, bg="#080d1a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Bottom manual dispatch bar
        bottom_dispatch = tk.Frame(left_container, bg="#131e36", bd=1, relief="solid")
        bottom_dispatch.pack(side="bottom", fill="x", pady=(10, 0), ipady=6, ipadx=8)

        tk.Label(bottom_dispatch, text="Gợi ý Baseline:", bg="#131e36", fg="#38bdf8", font=("Segoe UI", 9, "bold")).pack(side="left", padx=6)
        self.lbl_rec_act = tk.Label(bottom_dispatch, text="Đang phân tích...", bg="#131e36", fg="#ffffff", font=("Segoe UI", 9, "bold"))
        self.lbl_rec_act.pack(side="left", padx=4)

        btn_apply_rec = ttk.Button(bottom_dispatch, text="Áp Dụng Gợi Ý", style="Primary.TButton", command=self._exec_rec)
        btn_apply_rec.pack(side="left", padx=8)

        # Right Container: KPIs, Queues, Ambulances, Event Log
        right_container = tk.Frame(body, bg="#0b1120", width=420)
        right_container.pack(side="right", fill="both", padx=(0, 0))
        right_container.pack_propagate(False)

        # KPI Frame
        kpi_frame = tk.Frame(right_container, bg="#131e36", bd=1, relief="solid")
        kpi_frame.pack(side="top", fill="x", pady=(0, 8), padx=2, ipady=6, ipadx=8)

        tk.Label(kpi_frame, text="📈 CHỈ SỐ VẬN HÀNH (KPIS)", font=("Segoe UI", 9, "bold"), bg="#131e36", fg="#38bdf8").pack(anchor="w")

        kpi_sub = tk.Frame(kpi_frame, bg="#131e36")
        kpi_sub.pack(fill="x", pady=4)

        self.lbl_kpi_golden = tk.Label(kpi_sub, text="Tỷ lệ Vàng: 0.0%", bg="#131e36", fg="#34d399", font=("Segoe UI", 9, "bold"))
        self.lbl_kpi_golden.grid(row=0, column=0, sticky="w", padx=6, pady=2)

        self.lbl_kpi_resp = tk.Label(kpi_sub, text="TG Đón TB: 0.0", bg="#131e36", fg="#fbbf24", font=("Segoe UI", 9, "bold"))
        self.lbl_kpi_resp.grid(row=0, column=1, sticky="w", padx=6, pady=2)

        self.lbl_kpi_deliv = tk.Label(kpi_sub, text="Chuyển viện: 0 ca", bg="#131e36", fg="#c084fc", font=("Segoe UI", 9, "bold"))
        self.lbl_kpi_deliv.grid(row=1, column=0, sticky="w", padx=6, pady=2)

        self.lbl_kpi_exp = tk.Label(kpi_sub, text="Quá hạn: 0 ca", bg="#131e36", fg="#f87171", font=("Segoe UI", 9, "bold"))
        self.lbl_kpi_exp.grid(row=1, column=1, sticky="w", padx=6, pady=2)

        # Requests Frame
        req_frame = tk.Frame(right_container, bg="#131e36", bd=1, relief="solid")
        req_frame.pack(side="top", fill="x", pady=(0, 8), padx=2, ipady=4, ipadx=6)

        tk.Label(req_frame, text="🚨 HÀNG CHỜ CẤP CỨU (M=3 SLOTS)", font=("Segoe UI", 9, "bold"), bg="#131e36", fg="#ef4444").pack(anchor="w", padx=4, pady=2)
        self.req_labels: List[tk.Label] = []
        for i in range(3):
            lbl = tk.Label(req_frame, text=f"[Slot {i+1}] Trống", bg="#1e293b", fg="#94a3b8", font=("Segoe UI", 8), anchor="w", padx=8, pady=4, bd=1, relief="groove")
            lbl.pack(fill="x", padx=4, pady=2)
            self.req_labels.append(lbl)

        # Ambulances Frame
        amb_frame = tk.Frame(right_container, bg="#131e36", bd=1, relief="solid")
        amb_frame.pack(side="top", fill="x", pady=(0, 8), padx=2, ipady=4, ipadx=6)

        tk.Label(amb_frame, text="🚑 ĐỘI XE CỨU THƯƠNG (K=3 XE)", font=("Segoe UI", 9, "bold"), bg="#131e36", fg="#38bdf8").pack(anchor="w", padx=4, pady=2)
        self.amb_labels: List[tk.Label] = []
        for i in range(3):
            lbl = tk.Label(amb_frame, text=f"Xe {i+1}: Rảnh tại trạm", bg="#1e293b", fg="#94a3b8", font=("Segoe UI", 8), anchor="w", padx=8, pady=4, bd=1, relief="groove")
            lbl.pack(fill="x", padx=4, pady=2)
            self.amb_labels.append(lbl)

        # Event Log
        log_frame = tk.Frame(right_container, bg="#131e36", bd=1, relief="solid")
        log_frame.pack(side="top", fill="both", expand=True, padx=2)

        tk.Label(log_frame, text="📜 NHẬT KÝ SỰ KIỆN", font=("Segoe UI", 9, "bold"), bg="#131e36", fg="#94a3b8").pack(anchor="w", padx=4, pady=2)
        self.log_text = tk.Text(log_frame, bg="#080d1a", fg="#cbd5e1", font=("Consolas", 8), bd=0, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)

    def _draw_graph(self) -> None:
        self.canvas.delete("all")
        graph = self.env.graph

        # Vẽ các cạnh (Edges)
        for edge in graph.edges:
            u, v = edge.u, edge.v
            if u not in self.node_coords or v not in self.node_coords:
                continue
            x1, y1 = self.node_coords[u]
            x2, y2 = self.node_coords[v]

            if edge.is_arterial:
                cong = graph.arterial_traffic[edge.arterial_idx]
                if cong == 0:
                    color, width = "#10b981", 6  # Clear
                elif cong == 1:
                    color, width = "#f59e0b", 7  # Busy
                else:
                    color, width = "#ef4444", 8  # Congested
                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, capstyle="round")

                # Nhãn HM
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                self.canvas.create_rectangle(mx - 14, my - 9, mx + 14, my + 9, fill="#1e293b", outline="#334155")
                self.canvas.create_text(mx, my, text=f"HM{edge.arterial_idx + 1}", fill="#94a3b8", font=("Segoe UI", 7, "bold"))
            elif edge.base_length > 1.1:
                # Đường nhánh / vành đai nét đứt
                self.canvas.create_line(x1, y1, x2, y2, fill="#334b7a", width=2, dash=(5, 5))
            else:
                # Đường nội đô thông thường
                self.canvas.create_line(x1, y1, x2, y2, fill="#2a3b63", width=3, capstyle="round")

        # Vẽ các nút (15 Nodes)
        for n_id, coord in self.node_coords.items():
            x, y = coord
            node_info = graph.nodes[n_id]
            ztype = node_info.zone_type

            # Màu nút theo phân khu
            if ztype == "station":
                fill_color, outline_color = "#1e3a8a", "#3b82f6"
            elif ztype == "hospital":
                fill_color, outline_color = "#881337", "#f43f5e"
            elif ztype == "commercial":
                fill_color, outline_color = "#78350f", "#f59e0b"
            else:
                fill_color, outline_color = "#064e3b", "#10b981"

            # Vòng tròn nút
            r = 18
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill_color, outline=outline_color, width=3)
            self.canvas.create_text(x, y, text=str(n_id), fill="#ffffff", font=("Segoe UI", 9, "bold"))

            # Tên nút vắn tắt
            short_name = (
                node_info.name.replace("Khu Dân Cư", "KDC")
                .replace("Bệnh Viện Đa Khoa", "BV ĐK")
                .replace("Trung Tâm Thương Mại", "TTTM")
                .replace("Trạm Cứu Hộ", "Trạm")
                .replace("Trạm Trung Tâm Thành Phố", "Trạm TT")
            )
            self.canvas.create_text(x, y + 26, text=short_name, fill="#94a3b8", font=("Segoe UI", 7))

        # Vẽ chỉ báo cuộc gọi cấp cứu (Distress Beacon)
        for req in self.env.pending_requests:
            if req is not None and req.loc in self.node_coords:
                rx, ry = self.node_coords[req.loc]
                prio_col = "#ef4444" if req.prio == 1 else ("#f59e0b" if req.prio == 2 else "#10b981")
                # Vòng sáng cảnh báo
                self.canvas.create_oval(rx - 25, ry - 25, rx + 25, ry + 25, outline=prio_col, width=2, dash=(3, 3))
                # Icon còi
                self.canvas.create_text(rx + 16, ry - 14, text="🚨", font=("Segoe UI", 11))

        # Vẽ các xe cứu thương (Ambulances)
        for amb in self.env.ambulances:
            pos = self.node_coords.get(amb.node)
            if not pos:
                continue
            ax, ay = pos

            # Offset nếu nhiều xe cùng 1 nút
            offset = (amb.amb_id - 1) * 14
            cur_x = ax + offset
            cur_y = ay - 24

            if amb.status == AMB_STATE_IDLE:
                amb_col = "#059669"
            elif amb.status == AMB_STATE_ON_SCENE:
                amb_col = "#d97706"
            elif amb.status == AMB_STATE_TO_HOSPITAL:
                amb_col = "#7c3aed"
            elif amb.status == AMB_STATE_RELOCATING:
                amb_col = "#0891b2"
            else:
                amb_col = "#0284c7"

            # Hộp xe
            self.canvas.create_rectangle(cur_x - 14, cur_y - 9, cur_x + 14, cur_y + 9, fill=amb_col, outline="#ffffff", width=1.5)
            self.canvas.create_text(cur_x, cur_y, text=f"A{amb.amb_id + 1}", fill="#ffffff", font=("Segoe UI", 7, "bold"))

    def _refresh_ui(self) -> None:
        env = self.env
        m = env.get_metrics()

        # Update Header
        self.lbl_step.config(text=f"⏱ Bước: {env.step_count} / {env.max_steps}")
        sign = "+" if env.total_reward >= 0 else ""
        self.lbl_total_reward.config(text=f"⭐ Reward: {sign}{env.total_reward:.1f}", fg="#34d399" if env.total_reward >= 0 else "#f87171")

        is_rush = bool(20 <= (env.step_count % 50) <= 38)
        if is_rush:
            self.lbl_rush_hour.config(text="⚠️ GIỜ CAO ĐIỂM (Rush Hour)", bg="#78350f", fg="#fbbf24")
        else:
            self.lbl_rush_hour.config(text="🟢 Giờ bình thường", bg="#064e3b", fg="#34d399")

        # Update KPIs
        self.lbl_kpi_golden.config(text=f"Tỷ lệ Vàng: {m['golden_time_rate_pct']:.1f}%")
        self.lbl_kpi_resp.config(text=f"TG Đón TB: {m['avg_response_time']:.2f}")
        self.lbl_kpi_deliv.config(text=f"Chuyển viện: {m['completed_hospital_deliveries']} ca")
        self.lbl_kpi_exp.config(text=f"Quá hạn: {m['expired_requests']} ca")

        # Update Requests Slots
        for idx, lbl in enumerate(self.req_labels):
            req = env.pending_requests[idx]
            if req is None:
                lbl.config(text=f"[Slot {idx+1}] Trống (Sẵn sàng)", bg="#1e293b", fg="#64748b")
            else:
                p_col = "#f87171" if req.prio == 1 else ("#fbbf24" if req.prio == 2 else "#34d399")
                lbl.config(
                    text=f"[Slot {idx+1}] Ca #{req.req_id} ({req.prio_name}) tại Nút {req.loc}\n  Đã chờ: {req.wait_time}/{req.max_wait_time} bước (Vàng: {req.golden_time})",
                    bg="#0f172a",
                    fg=p_col,
                )

        # Update Ambulances Status
        for idx, lbl in enumerate(self.amb_labels):
            amb = env.ambulances[idx]
            detail = f"Nút {amb.node} | {amb.status_name}"
            if amb.status == AMB_STATE_TO_SCENE and amb.active_request:
                detail += f" -> Đón Ca #{amb.active_request.req_id} (Nút {amb.target_node})"
            elif amb.status == AMB_STATE_TO_HOSPITAL:
                detail += f" -> BV Nút {amb.target_node}"

            lbl.config(text=f"Xe {idx+1}: {detail}", bg="#0f172a", fg="#e2e8f0")

        # Update Recommended Action
        rec_act = env.recommend_baseline_action()
        dec = env.decode_action(rec_act)
        self.lbl_rec_act.config(text=f"[{rec_act}] {dec['label']}")

        # Re-render Graph
        self._draw_graph()

    def _step_once(self) -> None:
        if self.env.step_count >= self.env.max_steps:
            if self.is_playing:
                self._toggle_play()
            messagebox.showinfo("Hoàn tất", f"Episode kết thúc!\nTổng reward: {self.env.total_reward:.1f}")
            return

        if self.policy_mode == "baseline":
            action = self.env.recommend_baseline_action()
        elif self.policy_mode == "random":
            v_acts = self.env.get_valid_actions()
            action = int(self.env.rng.choice(v_acts))
        else:
            action = self.env.recommend_baseline_action()

        dec = self.env.decode_action(action)
        _, r, done, info = self.env.step(action)

        # Log events
        for ev in info.get("events", []):
            self.log_text.insert("end", f"[t={self.env.step_count}] {ev}\n")
        self.log_text.see("end")

        self._refresh_ui()

    def _toggle_play(self) -> None:
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.config(text="⏸ Tạm Dừng", style="Warning.TButton")
            self._schedule_next_step()
        else:
            self.btn_play.config(text="▶ Tự Động Chạy", style="Success.TButton")
            if self.play_job:
                self.root.after_cancel(self.play_job)
                self.play_job = None

    def _schedule_next_step(self) -> None:
        if self.is_playing and self.env.step_count < self.env.max_steps:
            self._step_once()
            self.play_job = self.root.after(self.play_speed_ms, self._schedule_next_step)
        elif self.is_playing:
            self._toggle_play()

    def _reset_env(self) -> None:
        if self.is_playing:
            self._toggle_play()
        self.seed = int(np.random.randint(1, 9999))
        self.env = SmartCityAmbulanceEnv(config=EnvConfig(max_steps=self.max_steps), seed=self.seed)
        self.env.reset(seed=self.seed)
        self.log_text.delete("1.0", "end")
        self.log_text.insert("end", f"Đã đặt lại môi trường với seed = {self.seed}\n")
        self._refresh_ui()

    def _exec_rec(self) -> None:
        rec_act = self.env.recommend_baseline_action()
        _, r, done, info = self.env.step(rec_act)
        for ev in info.get("events", []):
            self.log_text.insert("end", f"[t={self.env.step_count}] {ev}\n")
        self.log_text.see("end")
        self._refresh_ui()

    def _on_policy_change(self, event=None) -> None:
        val = self.combo_policy.get()
        if "Baseline" in val:
            self.policy_mode = "baseline"
        elif "Random" in val or "Ngẫu nhiên" in val:
            self.policy_mode = "random"
        else:
            self.policy_mode = "manual"

    def _on_speed_change(self, event=None) -> None:
        val = self.combo_speed.get()
        if "Chậm" in val:
            self.play_speed_ms = 1200
        elif "Nhanh" in val:
            self.play_speed_ms = 350
        else:
            self.play_speed_ms = 700


def main() -> None:
    root = tk.Tk()
    app = AmbulanceDispatchDesktopGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
