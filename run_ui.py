"""
Script Khởi Chạy Giao Diện Điều Phối Xe Cứu Thương Thông Minh.
Hỗ trợ cả Giao diện Web Dashboard thời gian thực (mặc định) và Giao diện Desktop Tkinter.

Cách dùng:
  python run_ui.py             # Mở Web Dashboard trên trình duyệt (Khuyên dùng)
  python run_ui.py --desktop   # Mở Desktop Tkinter GUI
  python run_ui.py --port 8080 # Chỉ định cổng Web Server
"""

from __future__ import annotations

import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Khởi chạy Giao diện Mô phỏng Điều phối Xe Cứu thương 80 Nút (MDP)"
    )
    parser.add_argument(
        "--desktop",
        action="store_true",
        help="Khởi chạy giao diện Desktop Tkinter độc lập",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Cổng chạy Web GUI Server (mặc định: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Không tự động mở trình duyệt web khi chạy Web GUI",
    )

    args = parser.parse_args()

    if args.desktop:
        print("[+] Đang khởi động Giao diện Desktop Tkinter...")
        from gui_app import run_desktop_app
        run_desktop_app()
    else:
        from web_gui import run_web_server
        run_web_server(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()

