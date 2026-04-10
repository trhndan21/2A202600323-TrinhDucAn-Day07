# Hiền - Kết quả Benchmark

## Chiến lược
- Thành viên: Hiền
- Chunking: FixedSizeChunker
- chunk_size: 256
- overlap: 20% (khoảng 51 ký tự)
- top_k: 3
- Embedding: all-MiniLM-L6-v2

## Kết quả truy vấn

| # | Query (rút gọn) | Điểm Top-1 | Có chunk liên quan trong top-3 | Đạt/Không đạt |
|---|---|---:|---|---|
| 1 | Lý dị + bệnh tim của Thúy Hằng | 0.742 | Có | PASS |
| 2 | Nhân vật "tôi" gặp người con trai qua phương tiện nào | 0.671 | Có | PASS |
| 3 | Ở bên nhau bao lâu trước chia tay sân bay | 0.684 | Có | PASS |
| 4 | Vì sao Mẫn Huy bỏ nhà ra đi | 0.709 | Có | PASS |
| 5 | Vì sao nhân vật quyết vượt biển | 0.543 | Không | FAIL |

## Tổng kết
- Tổng số query: 5
- PASS: 4
- FAIL: 1
- Tỉ lệ đạt: 80%

## Nhận xét
- FixedSize 256 với overlap 20% cho kết quả ổn với các query trực tiếp về nhân vật - sự kiện.
- Query 5 khó hơn do liên quan nghĩa ẩn dụ ("vượt biển" theo nghĩa "vượt cạn"), nên chưa truy hồi tốt trong top-3.
