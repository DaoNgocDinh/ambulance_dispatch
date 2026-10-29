# Môi Trường Bản Đồ Đô Thị 80 Nút Cho Điều Phối Xe Cứu Thương (Tree-like Urban Road Network)

Môi trường bản đồ thử nghiệm đã được nâng cấp lên quy mô **80 NÚT GIAO THÔNG** theo kiến trúc mạng lưới **dạng cây phân nhánh kết hợp mạng lưới trung tâm dày đặc (Dense Core Spider-web & Multi-branch Suburban Corridors)** và **4 BỆNH VIỆN** (2 bệnh viện gần trung tâm, 2 bệnh viện ở xa trung tâm), phục vụ cho bài toán điều phối xe cứu thương trong đô thị thông minh.

---

## 1. Các Đặc Điểm Cốt Lõi Của Bản Đồ 80 Nút

1. **Quy mô 80 nút giao cắt ($1 \to 80$):**
   - Đánh số liên tục từ `1` đến `80`.
   - Mỗi nút có tọa độ thực $(x, y)$ trên không gian đô thị rộng lớn $35\text{ km} \times 32\text{ km}$.
   - Tượng trưng cho đầy đủ các kiểu điểm giao cắt thực tế:
     - **Ngã 4 (Crossroads - Bậc 4):** 16 nút (Nút 1, 2, 3, 4, 5, 10, 11, 12, 13, 21, 22, 23, 41, 42, 61, 62) – các đại lộ trung tâm nội đô và đầu mối giao thông liên vùng.
     - **Ngã 3 (T-junction - Bậc 3):** 15 nút (Nút 6, 7, 8, 9, 24, 25, 27, 43, 44, 45, 48, 63, 64, 65, 68) – các điểm phân luồng liên khu và cửa ngõ vành đai.
     - **Đường rẽ / Nối tiếp (Turn - Bậc 2):** 14 nút (Nút 14, 15, 16, 17, 18, 19, 20, 26, 46, 47, 49, 66, 67, 69) – các đoạn rẽ/vành đai chuyển tiếp kết nối mạch lạc.
     - **Đường cụt (Dead-end - Bậc 1):** 35 nút (Nút 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 50..60, 70..80) – các khu dân cư nội bộ, làng ngoại ô, cảng hàng hóa, khu bảo tồn sinh thái.

2. **Vùng lõi trung tâm nội đô dày đặc (Dense Inner-City Core):**
   - Khu vực trung tâm (Nút 1 đến 20) trong phạm vi $X \in [12, 24], Y \in [11, 21]$ có mật độ tuyến đường cực kỳ dày đặc.
   - Các nút được liên kết đan xen đa tầng (Spider-web + Grid) với khoảng cách ngắn ($1.0 \to 2.5\text{ km}$), phản ánh đúng thực tế giao thông giờ cao điểm trong khu vực nội đô.

3. **Tăng độ dày và hiển thị rõ hướng đi bằng mũi tên:**
   - Các tuyến đường được vẽ với độ dày lớn (`linewidth = 3.2`), thể hiện rõ ràng 2 làn đường riêng biệt song song.
   - Mỗi hướng đi xuất phát từ nút $u$ đến nút $v$ đều có **mũi tên to, rõ ràng** hiển thị chính xác chiều lưu thông ($u \to v$ và $v \to u$).
   - Màu sắc mỗi mũi tên thể hiện độ tắc của riêng chiều đó:
     - **Màu Xanh lá:** Mức 1 - Thông thoáng ($\approx 50\text{ km/h}$)
     - **Màu Vàng cam:** Mức 2 - Đông đúc ($\approx 30\text{ km/h}$)
     - **Màu Đỏ tươi:** Mức 3 - Ùn tắc ($\approx 15\text{ km/h}$)

4. **Tăng độ đa dạng của độ dài đoạn đường:**
   - 86 tuyến vô hướng tạo thành **172 cung đường có hướng**.
   - Chiều dài được tính theo khoảng cách Euclid giữa tọa độ hai nút, trải dài đa dạng từ **$1.00\text{ km}$** đến **$7.00\text{ km}$** (trung bình $3.13\text{ km}$):
     - **Đoạn ngắn nội đô ($1.00 \to 1.58\text{ km}$):** Nút 1-18 ($1.00\text{ km}$), Nút 1-19 ($1.00\text{ km}$), Nút 1-20 ($1.00\text{ km}$), Nút 4-20 ($1.00\text{ km}$), Nút 3-19 ($1.50\text{ km}$)...
     - **Đoạn trung bình kết nối ($2.00 \to 4.00\text{ km}$):** Nút 2-10 ($2.50\text{ km}$), Nút 4-9 ($2.50\text{ km}$), Nút 1-5 ($3.00\text{ km}$), Nút 21-24 ($3.50\text{ km}$), Nút 62-65 ($3.61\text{ km}$)...
     - **Đoạn dài trục vành đai và cao tốc ($5.00 \to 7.00\text{ km}$):** Nút 23-38 ($6.71\text{ km}$), Nút 27-32 ($7.00\text{ km}$)...

5. **4 Bệnh viện (2 Gần Trung Tâm, 2 Xa Trung Tâm):**
   - **Bệnh viện chỉ ở 1 đường duy nhất**: Mỗi bệnh viện được neo trên **DUY NHẤT 1 cung đường có hướng** ($u \to v$). Tuyệt đối **không** nằm trên chiều ngược lại ($v \to u$) và không nằm trên các nút.
   - **Vị trí tính chính xác bằng độ dài từ điểm xuất phát (from_node):**
     - **Bệnh viện 1 (BV Đa khoa Trung tâm Thành phố - GẦN TRUNG TÂM):**
       - Nằm trên cung đường: **$19 \to 3$** (tổng chiều dài cung $1.50\text{ km}$)
       - Điểm xuất phát: **Nút 19**
       - Khoảng cách từ điểm xuất phát: **$0.80\text{ km}$** tính từ Nút 19
       - Khoảng cách còn lại đến đích: **$0.70\text{ km}$** đến Nút 3
       - Tọa độ địa lý: $(20.30, 16.00)\text{ km}$
     - **Bệnh viện 2 (BV Cấp cứu Tim mạch Nội đô - GẦN TRUNG TÂM):**
       - Nằm trên cung đường: **$4 \to 9$** (tổng chiều dài cung $2.50\text{ km}$)
       - Điểm xuất phát: **Nút 4**
       - Khoảng cách từ điểm xuất phát: **$1.00\text{ km}$** tính từ Nút 4
       - Khoảng cách còn lại đến đích: **$1.50\text{ km}$** đến Nút 9
       - Tọa độ địa lý: $(17.00, 14.00)\text{ km}$
     - **Bệnh viện 3 (BV Đa khoa Khu vực Bắc Đô thị - XA TRUNG TÂM):**
       - Nằm trên cung đường: **$21 \to 24$** (tổng chiều dài cung $3.50\text{ km}$)
       - Điểm xuất phát: **Nút 21**
       - Khoảng cách từ điểm xuất phát: **$1.80\text{ km}$** tính từ Nút 21
       - Khoảng cách còn lại đến đích: **$1.70\text{ km}$** đến Nút 24
       - Tọa độ địa lý: $(18.00, 25.30)\text{ km}$
     - **Bệnh viện 4 (BV Cấp cứu Ngoại ô Tây Nam - XA TRUNG TÂM):**
       - Nằm trên cung đường: **$62 \to 65$** (tổng chiều dài cung $3.61\text{ km}$)
       - Điểm xuất phát: **Nút 62**
       - Khoảng cách từ điểm xuất phát: **$1.80\text{ km}$** tính từ Nút 62
       - Khoảng cách còn lại đến đích: **$1.81\text{ km}$** đến Nút 65
       - Tọa độ địa lý: $(13.00, 8.00)\text{ km}$

---

## 2. Cấu Trúc Thư Mục Dự Án

```text
ambulance_dispatch/
├── map_env/                    # Package mã nguồn môi trường bản đồ
│   ├── __init__.py             # Export MapNode, DirectedEdge, HospitalOnRoad, CityMap
│   ├── node.py                 # Định nghĩa MapNode và phân loại kiểu giao lộ (Ngã 4, 3, Rẽ, Cụt, Phức hợp)
│   ├── edge.py                 # Định nghĩa DirectedEdge, độ dài, biến độ tắc in {1,2,3}, cập nhật Markov
│   ├── hospital.py             # Định nghĩa HospitalOnRoad (bệnh viện trên 1 cung đường, đo từ điểm đi)
│   └── city_map.py             # Trình quản lý CityMap 80 nút, 172 cung có hướng, 4 bệnh viện
├── tests/
│   └── test_map.py             # Bộ 8 unit tests kiểm tra toàn diện quy mô 80 nút và 4 bệnh viện
├── demo_map.py                 # Script chạy demo chi tiết, in phân tích 80 nút, mũi tên 2 chiều, định tuyến 4 BV
├── visualize_map.py            # Script trực quan hóa đồ thị và lưu ảnh city_map_80nodes.png
├── city_map_80nodes.png        # Ảnh sơ đồ trực quan mạng lưới 80 nút (đường dày, mũi tên hướng đi, 4 BV)
├── requirements.txt            # Danh sách thư viện cần thiết (numpy, matplotlib)
└── README.md                   # Tài liệu hướng dẫn chi tiết
```

---

## 3. Phân Khái Quát 4 Vùng Mạng Lưới 80 Nút

| Khu vực | Phạm vi Nút | Đặc điểm mạng lưới | Kiểu giao lộ chính | Bệnh viện trực thuộc |
|:---|:---:|:---|:---|:---|
| **Vùng Lõi Nội Đô (Trung tâm)** | Nút 1 đến Nút 20 | Mạng lưới mạng nhện dày đặc (Spider-web + Grid), cự ly ngắn ($1.0 \to 3.0\text{ km}$), mật độ giao cắt cao. | Ngã 4 (5 nút), Ngã 3 (4 nút), Đường rẽ/nối tiếp (7 nút), Đường cụt (4 nút) | **BV 1** ($19 \to 3$)<br>**BV 2** ($4 \to 9$) |
| **Hành lang Phía Bắc (Ngoại ô Bắc)** | Nút 21 đến Nút 40 | Nhánh cây phân tỏa về vùng hồ và bảo tồn sinh thái Tây Bắc, Đông Bắc. Cự ly $1.5 \to 7.0\text{ km}$. | Ngã 4 (4 nút), Ngã 3 (3 nút), Đường rẽ (2 nút), Đường cụt (11 nút) | **BV 3** ($21 \to 24$) |
| **Hành lang Phía Đông (Cảng Biển & KCN)** | Nút 41 đến Nút 60 | Nhánh cây công nghiệp hướng ra cửa biển và cụm logistics phía Đông. Cự ly $1.5 \to 5.5\text{ km}$. | Ngã 4 (2 nút), Ngã 3 (4 nút), Đường rẽ (3 nút), Đường cụt (11 nút) | Nằm giữa hành lang Đông và Trung tâm |
| **Hành lang Phía Nam & Tây Nam** | Nút 61 đến Nút 80 | Trục cao tốc đô thị kết nối khu đô thị vệ tinh và KCN công nghệ cao Tây Nam. | Ngã 4 (2 nút), Ngã 3 (4 nút), Đường rẽ (2 nút), Đường cụt (12 nút) | **BV 4** ($62 \to 65$) |

---

## 4. Chi Tiết Vị Trí 4 Bệnh Viện Trên Cung Đường Có Hướng

```text
[BỆNH VIỆN 1 - GẦN TRUNG TÂM]
Nút 19 (19.5, 16.0) ======= [BV 1] ===================> Nút 3 (21.0, 16.0)
                     0.80 km             0.70 km
        (Chiều 19 -> 3 duy nhất! Chiều 3 -> 1 KHÔNG có bệnh viện)
        Tọa độ BV 1: (20.30, 16.00) km | Tổng chiều dài: 1.50 km

[BỆNH VIỆN 2 - GẦN TRUNG TÂM]
Nút 4 (18.0, 14.0) ======== [BV 2] =====================> Nút 9 (15.5, 14.0)
                     1.00 km             1.50 km
        (Chiều 4 -> 9 duy nhất! Chiều 9 -> 4 KHÔNG có bệnh viện)
        Tọa độ BV 2: (17.00, 14.00) km | Tổng chiều dài: 2.50 km

[BỆNH VIỆN 3 - XA TRUNG TÂM (BẮC ĐÔ THỊ)]
Nút 21 (18.0, 23.5) ======= [BV 3] ====================> Nút 24 (18.0, 27.0)
                     1.80 km             1.70 km
        (Chiều 21 -> 24 duy nhất! Chiều 24 -> 21 KHÔNG có bệnh viện)
        Tọa độ BV 3: (18.00, 25.30) km | Tổng chiều dài: 3.50 km

[BỆNH VIỆN 4 - XA TRUNG TÂM (TÂY NAM)]
Nút 62 (15.5, 9.5) ======== [BV 4] =====================> Nút 65 (11.0, 6.5)
                     1.80 km             1.81 km
        (Chiều 62 -> 65 duy nhất! Chiều 65 -> 62 KHÔNG có bệnh viện)
        Tọa độ BV 4: (13.00, 8.00) km  | Tổng chiều dài: 3.61 km
```

- **Quy tắc tiếp cận bệnh viện:**
  - Để đưa bệnh nhân đến **Bệnh viện $k$**: Xe cứu thương di chuyển qua các nút giao đến **Nút xuất phát ($u$)** của cung đường chứa bệnh viện đó, sau đó đi tiếp đúng đoạn $d_{\text{from\_start}}$ theo chiều $u \to v$ để đến cổng bệnh viện.
  - Xe cứu thương đi theo chiều ngược lại ($v \to u$) sẽ không thể vào bệnh viện do đặc thù đường phân cách có dải phân cách cứng chống quay đầu tùy tiện.

---

## 5. Hướng Dẫn Sử Dụng & Chạy Thử Nghiệm

### 5.1. Chạy Demo Bản Đồ và Biến Thiên Giao Thông
```bash
python demo_map.py
```
*Script in tổng quan 80 nút, bảng các cung đường 2 chiều với mũi tên hướng đi rõ ràng, thống kê độ dài đa dạng ($1.00 \to 7.00\text{ km}$), mô phỏng 4 bước biến thiên giao thông giờ cao điểm, và tìm đường Dijkstra từ nút ngoại ô xa nhất (Nút 32) đến cả 4 bệnh viện.*

### 5.2. Chạy Bộ Kiểm Thử Tự Động (Unit Tests)
```bash
python -m unittest discover -s tests -v
```
*Tất cả 8 unit tests kiểm tra: đủ 80 nút liên tục, đủ các kiểu giao lộ, cự ly đa dạng (tỷ lệ $> 4\times$), tính độc lập và biến thiên của độ tắc đường $\{1, 2, 3\}$, và định vị chính xác 4 bệnh viện (2 gần trung tâm, 2 xa trung tâm).*

### 5.3. Tạo Ảnh Sơ Đồ Trực Quan Hóa Bản Đồ
```bash
python visualize_map.py
```
*Xuất ra file [`city_map_80nodes.png`](city_map_80nodes.png) kích thước siêu nét ($28 \times 24$ inch, 220 DPI) với:*
- Đường đi vẽ dày nổi bật.
- Mũi tên định hướng đi riêng biệt cho từng chiều.
- Khung tô sáng vùng lõi trung tâm nội đô dày đặc.
- Bảng chú giải trực quan 4 bệnh viện và các phân loại giao lộ.
