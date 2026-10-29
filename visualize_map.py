"""
Script trực quan hóa Bản đồ Thành phố Thông minh 80 NÚT dạng cây phân nhánh.
Các cải tiến đồ họa đáp ứng yêu cầu:
1. MÔ HÌNH 80 NÚT GIAO THÔNG (đánh số 1..80).
2. TĂNG ĐỘ DÀY CỦA ĐƯỜNG ĐI (Thick road lanes).
3. HIỂN THỊ RÕ HƯỚNG ĐI DƯỚI DẠNG MŨI TÊN (Xuất phát từ u -> v bằng mũi tên rõ ràng).
4. HIỂN THỊ ĐA DẠNG ĐỘ DÀI ĐOẠN ĐƯỜNG (Từ 1.00km đến 6.00km+).
5. 4 BỆNH VIỆN:
   - 2 Bệnh viện GẦN TRUNG TÂM (mạng lưới đường dày đặc): BV 1 và BV 2.
   - 2 Bệnh viện Ở XA TRUNG TÂM (khu vực ngoại ô vành đai): BV 3 và BV 4.
"""

from __future__ import annotations

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

from map_env import (
    CityMap,
    IntersectionType,
    CONGESTION_CLEAR,
    CONGESTION_MODERATE,
    CONGESTION_HEAVY,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def plot_city_map(city: CityMap, save_path: str = "city_map_80nodes.png") -> None:
    fig, ax = plt.subplots(figsize=(28, 24), dpi=220)

    # Bảng màu cho độ tắc đường
    congestion_colors = {
        CONGESTION_CLEAR: "#27ae60",     # Xanh lá đậm: Thông thoáng (1)
        CONGESTION_MODERATE: "#f39c12",  # Vàng cam: Đông đúc (2)
        CONGESTION_HEAVY: "#e74c3c",     # Đỏ tươi: Ùn tắc (3)
    }

    # Bảng màu và marker cho kiểu giao lộ
    type_styles = {
        IntersectionType.FOUR_WAY: {"color": "#8e44ad", "marker": "s", "size": 320, "label": "Ngã 4 (Crossroads)"},
        IntersectionType.THREE_WAY: {"color": "#2980b9", "marker": "D", "size": 260, "label": "Ngã 3 (T-junction)"},
        IntersectionType.TURN: {"color": "#16a085", "marker": "o", "size": 220, "label": "Đường rẽ / Nối tiếp (Turn)"},
        IntersectionType.DEAD_END: {"color": "#d35400", "marker": "^", "size": 240, "label": "Đường cụt (Dead-end)"},
        IntersectionType.COMPLEX: {"color": "#c0392b", "marker": "h", "size": 350, "label": "Nút phức hợp (Bậc ≥ 5)"},
    }

    # ------------------------------------------------------------------------
    # 1. Vẽ các tuyến đường (ĐƯỜNG DÀY & MŨI TÊN RÕ RÀNG THEO HƯỚNG XUẤT PHÁT)
    # ------------------------------------------------------------------------
    for u, v in city.road_pairs:
        nu = city.nodes[u]
        nv = city.nodes[v]

        dx = nv.x - nu.x
        dy = nv.y - nu.y
        dist = np.hypot(dx, dy)
        if dist == 0:
            continue

        # Độ lệch giữa 2 làn song song
        offset_dist = 0.14  # km
        nx_perp = -dy / dist * offset_dist
        ny_perp = dx / dist * offset_dist

        # Làn 1: Xuất phát từ u -> v
        edge_uv = city.edges[(u, v)]
        color_uv = congestion_colors[edge_uv.congestion]

        arrow_uv = FancyArrowPatch(
            (nu.x + nx_perp, nu.y + ny_perp),
            (nv.x + nx_perp, nv.y + ny_perp),
            arrowstyle="-|>",
            mutation_scale=16,     # Mũi tên rõ ràng
            linewidth=3.2,         # Độ dày đường đi
            color=color_uv,
            shrinkA=14,
            shrinkB=14,
            zorder=2,
        )
        ax.add_patch(arrow_uv)

        # Làn 2: Xuất phát từ v -> u
        edge_vu = city.edges[(v, u)]
        color_vu = congestion_colors[edge_vu.congestion]

        arrow_vu = FancyArrowPatch(
            (nv.x - nx_perp, nv.y - ny_perp),
            (nu.x - nx_perp, nu.y - ny_perp),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=3.2,
            color=color_vu,
            shrinkA=14,
            shrinkB=14,
            zorder=2,
        )
        ax.add_patch(arrow_vu)

        # Nhãn hiển thị độ dài đoạn đường tại trung điểm
        mid_x = (nu.x + nv.x) / 2.0
        mid_y = (nu.y + nv.y) / 2.0
        ax.text(
            mid_x,
            mid_y,
            f"{edge_uv.length:.1f}k",
            fontsize=6.5,
            fontweight="bold",
            color="#2c3e50",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="#ffffff", edgecolor="#bdc3c7", alpha=0.85, lw=0.6),
            zorder=4,
        )

    # ------------------------------------------------------------------------
    # 2. Vẽ 4 BỆNH VIỆN (2 GẦN TRUNG TÂM, 2 Ở XA TRUNG TÂM)
    # ------------------------------------------------------------------------
    hosp_callout_offsets = {
        1: (0.6, 0.8),    # BV 1: Lên trên bên phải
        2: (-4.5, -1.2),  # BV 2: Xuống dưới bên trái
        3: (0.6, 0.9),    # BV 3: Lên trên bên phải
        4: (-4.5, -1.2),  # BV 4: Xuống dưới bên trái
    }

    for hid, hosp in city.hospitals.items():
        hx, hy = hosp.coord

        # Biểu tượng dấu thập y tế
        ax.scatter(
            hx,
            hy,
            s=580,
            marker="P",
            color="#c0392b",
            edgecolors="#ffffff",
            linewidths=2.5,
            zorder=7,
        )

        zone_label = "[GẦN TRUNG TÂM]" if hid in (1, 2) else "[XA TRUNG TÂM]"
        callout_text = (
            f"★ {hosp.name} {zone_label}\n"
            f"• Nằm trên đường: N{hosp.from_node} ➔ N{hosp.to_node}\n"
            f"• Cách điểm xuất phát (N{hosp.from_node}): {hosp.distance_from_start:.2f} km\n"
            f"• Còn cách điểm đến (N{hosp.to_node}): {hosp.distance_to_end:.2f} km"
        )
        ox, oy = hosp_callout_offsets.get(hid, (0.6, 0.8))
        ax.annotate(
            callout_text,
            xy=(hx, hy),
            xytext=(hx + ox, hy + oy),
            fontsize=9.5,
            fontweight="bold",
            color="#78281f",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fcedeb", edgecolor="#c0392b", linewidth=2.0),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2.0),
            zorder=8,
        )

    # ------------------------------------------------------------------------
    # 3. Vẽ 80 nút giao thông
    # ------------------------------------------------------------------------
    drawn_types = set()
    for node_id in range(1, city.num_nodes + 1):
        node = city.nodes[node_id]
        style = type_styles[node.intersection_type]
        lbl = style["label"] if node.intersection_type not in drawn_types else None
        drawn_types.add(node.intersection_type)

        ax.scatter(
            node.x,
            node.y,
            s=style["size"],
            marker=style["marker"],
            color=style["color"],
            edgecolors="#ffffff",
            linewidths=1.8,
            label=lbl,
            zorder=5,
        )

        # Số thứ tự nút 1..80
        ax.text(
            node.x,
            node.y,
            str(node.node_id),
            color="white",
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )

        # Nhãn tên nút giao
        ax.text(
            node.x,
            node.y - 0.38,
            f"N{node.node_id}",
            fontsize=7.5,
            ha="center",
            va="top",
            color="#1a252f",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.08", facecolor="#ecf0f1", edgecolor="none", alpha=0.65),
            zorder=6,
        )

    # Vùng highlight bao quanh khu Trung tâm (Lõi dày đặc)
    rect = plt.Rectangle((12.0, 11.0), 12.0, 10.0, fill=True, facecolor="#3498db", alpha=0.06, linestyle=":", edgecolor="#2980b9", lw=1.8, zorder=1)
    ax.add_patch(rect)
    ax.text(12.3, 20.6, "[VÙNG LÕI TRUNG TÂM NỘI ĐÔ - TUYẾN ĐƯỜNG DÀY ĐẶC]", fontsize=11, fontweight="bold", color="#1f618d")

    # ------------------------------------------------------------------------
    # 4. Chú thích & Trang trí bản đồ
    # ------------------------------------------------------------------------
    traffic_legend_elements = [
        Line2D([0], [0], color=congestion_colors[CONGESTION_CLEAR], lw=4.0, label="Mức 1: Thông thoáng (~50 km/h)"),
        Line2D([0], [0], color=congestion_colors[CONGESTION_MODERATE], lw=4.0, label="Mức 2: Đông đúc (~30 km/h)"),
        Line2D([0], [0], color=congestion_colors[CONGESTION_HEAVY], lw=4.0, label="Mức 3: Ùn tắc (~15 km/h)"),
        Line2D([0], [0], marker="P", color="w", markerfacecolor="#c0392b", markersize=14, label="Bệnh viện (4 BV: 2 gần trung tâm, 2 xa trung tâm)"),
    ]

    leg1 = ax.legend(
        handles=traffic_legend_elements,
        loc="upper right",
        title="Độ tắc đường (Mỗi hướng có mũi tên riêng)",
        framealpha=0.95,
        fontsize=10.5,
        title_fontsize=11.5,
    )
    ax.add_artist(leg1)

    ax.legend(
        loc="upper left",
        title="Phân loại 80 nút giao cắt",
        framealpha=0.95,
        fontsize=10.5,
        title_fontsize=11.5,
    )

    ax.set_title(
        "BẢN ĐỒ THÀNH PHỐ THÔNG MINH 80 NÚT (DẠNG CÂY PHÂN NHÁNH)\n"
        "Mạng lưới trung tâm dày đặc, đường dày, mũi tên hướng đi & 4 Bệnh viện (2 Gần Trung Tâm, 2 Xa Trung Tâm)",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("Tọa độ trục X (km)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Tọa độ trục Y (km)", fontsize=11, fontweight="bold")
    ax.set_xlim(0.0, 35.0)
    ax.set_ylim(0.0, 32.5)
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Đã tạo và lưu thành công bản đồ trực quan tại: {save_path}")


if __name__ == "__main__":
    city = CityMap(seed=100)
    plot_city_map(city)
