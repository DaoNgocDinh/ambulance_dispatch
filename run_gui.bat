@echo off
chcp 65001 > nul
title Trung Tâm Điều Phối Xe Cứu Thương - Smart City Ambulance Dispatch (MDP)
color 0B

echo ===============================================================================
echo     TRUNG TÂM ĐIỀU PHỐI XE CỨU THƯƠNG THÀNH PHỐ THÔNG MINH (MDP)
echo ===============================================================================
echo.
echo  Chọn giao diện mô phỏng bạn muốn khởi chạy:
echo.
echo   [1] Giao diện Web Hiện Đại (Khuyến nghị - Mở trên trình duyệt)
echo       - Đồ họa SVG động, giám sát giao thông 5 tuyến huyết mạch
echo       - Điều phối xe thời gian thực, hàng chờ 3 ca cấp cứu
echo       - Đánh giá KPI & công cụ so sánh Benchmark trực quan
echo.
echo   [2] Giao diện Cửa Sổ Desktop (Tkinter Native)
echo       - Cửa sổ ứng dụng máy tính nhẹ, đồ họa Canvas 15 nút
echo       - Chạy tự động hoặc từng bước, nhật ký sự kiện thời gian thực
echo.
echo   [3] Chạy demo terminal cơ bản (demo_env.py)
echo.
echo   [0] Thoát
echo.
echo ===============================================================================
set /p choice="Nhập lựa chọn của bạn (1, 2, 3 hoặc 0): "

if "%choice%"=="1" (
    echo.
    echo Đang khởi động Giao diện Web tại http://localhost:8000 ...
    python run_ui.py
    pause
    exit
)

if "%choice%"=="2" (
    echo.
    echo Đang khởi động Giao diện Desktop Tkinter (80 Nút, Đa thuật toán)...
    python run_ui.py --desktop
    pause
    exit
)

if "%choice%"=="3" (
    echo.
    echo Đang chạy mô phỏng mẫu...
    python run_simulation.py
    pause
    exit
)

echo Tạm biệt!
pause
