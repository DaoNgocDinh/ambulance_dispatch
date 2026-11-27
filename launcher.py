"""
Trình khởi chạy tiện ích cho Hệ thống Điều phối Xe cứu thương (Smart City Ambulance Dispatch).
Cho phép chọn giữa Giao diện Web (khuyến nghị) hoặc Giao diện Desktop (Tkinter).
"""

import argparse
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Khởi chạy Giao diện Điều phối Xe Cứu Thương Đô Thị Thông Minh (MDP)"
    )
    parser.add_argument(
        "--mode",
        choices=["web", "desktop", "demo"],
        default="web",
        help="Chế độ giao diện: 'web' (mặc định), 'desktop' (Tkinter), hoặc 'demo' (Console trace)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Cổng mạng cho Giao diện Web (mặc định: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Không tự động mở trình duyệt khi chạy Giao diện Web",
    )

    args = parser.parse_args()

    if args.mode == "web":
        from web_gui import run_web_server
        run_web_server(port=args.port, open_browser=not args.no_browser)
    elif args.mode == "desktop":
        from app_gui import main as run_desktop
        run_desktop()
    elif args.mode == "demo":
        from demo_env import run_step_by_step_demo, evaluate_policy
        run_step_by_step_demo(num_demo_steps=6)
        print("\nSo sánh thử nghiệm 30 episodes...")
        for pol in ["random", "baseline"]:
            res = evaluate_policy(pol, num_episodes=15)
            print(f"Policy: {pol} -> Reward TB: {res['total_reward']:.2f}")


if __name__ == "__main__":
    main()
