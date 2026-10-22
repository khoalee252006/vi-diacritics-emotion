# Thiết kế CHỐT TRƯỚC: phương pháp A (huấn luyện hai góc nhìn) và B (gating bằng bộ phát hiện đã kiểm định), 2026-10-14

Cơ sở (đã có số liệu): model không dấu (PS) bền nhưng mất 10–15 điểm câu sạch; D2 cho thấy phơi nhiễm dạng không dấu
có tác dụng đặc hiệu theo từ và không hại câu sạch; gating theo tỉ lệ dấu thất bại trên dữ liệu thật vì bộ phát hiện
(13/70); bộ phát hiện mới đạt precision 88.1% trên ViLexNorm.

## A — Huấn luyện hai góc nhìn (một model, không cần bộ phát hiện)
- **A-dual:** nhân đôi train: bản gốc + bản bỏ HẾT dấu (cùng nhãn). Cùng công thức RQ1 (lr 2e-5, batch 16, 20 epoch).
- **A-kl:** mỗi mẫu đưa vào cả hai bản trong cùng bước; loss = ½[CE(gốc) + CE(không dấu)] + λ·KL đối xứng giữa hai
  phân phối dự đoán; λ = 1 (chốt trước, không dò).
- Model: XLM-R, PhoBERT, ViSoBERT × seed 42–44 × 2 biến thể = 18 run (cùng phạm vi với D2; CafeBERT không làm).
- **Chọn checkpoint:** valid sạch (như mọi thí nghiệm khác).
- **Chọn biến thể chính (A-dual hay A-kl) trên VALID, không xem test:** MF1 trung bình đều trên valid sạch + valid
  bỏ dấu 25/50/75/100% (`results/uncertainty/valid_noisy/`), trung bình 3 model × 3 seed. Biến thể còn lại là phụ.
- **Tiêu chí (biến thể chính, test chấm một lần), bootstrap hai tầng 10,000 vòng:**
  - A1 câu sạch: A không thua model thường (RQ1 P0, 5 seed) có ý nghĩa — từng model (3 phép so).
  - A2 bỏ dấu 100%: A không thua model không dấu PS (5 seed) có ý nghĩa — từng model (3).
  - A3 bỏ dấu 75% và 100%: A hơn augment hỗn hợp (rq3_aug) có ý nghĩa — ≥ 2/3 model ở mỗi mức (6).
  - A4 mất dấu tự nhiên (N = 94, C = 282, `results/natural/`): khoảng cách acc(C) − acc(N) của A nhỏ hơn của model
    thường — ≥ 2/3 model (3).
  - Holm chung 15 phép so. "Không thua" = không có khác biệt âm có ý nghĩa sau Holm.
  - **THÀNH CÔNG** khi A1 và A2 đạt ở cả 3 model VÀ A3 đạt. A4 báo cáo như kiểm chứng ngoài.

## B — Gating bằng bộ phát hiện đã kiểm định
- Trọng số: với câu x, bộ phát hiện (`src/natural.py: detector`, đã chuẩn hóa NFC) cho miss (âm tiết lẽ ra có dấu bị
  mất) và acc (âm tiết có dấu). Nếu miss + acc < 3 → w = 1 (giữ model thường). Ngược lại w = acc / (acc + miss).
  p = w·p_thường + (1 − w)·p_PS. Không có tham số dò.
- Model: 4 PLM (CafeBERT, XLM-R, PhoBERT, ViSoBERT) trên các tập đã có dự đoán PS; với tập mất dấu tự nhiên cần PS
  checkpoint → train lại PS cho XLM-R, PhoBERT, ViSoBERT × seed 42–44 có lưu checkpoint (9 run).
- **Tiêu chí, bootstrap hai tầng, so B với gating tỉ lệ dấu cũ (G0) và model thường (STD):**
  - B1 câu sạch: B không thua STD có ý nghĩa (4 PLM).
  - B2 bỏ dấu 75%, 100%: B không thua G0 có ý nghĩa (4 PLM × 2 = 8).
  - B3 ViGoEmotions test toàn tập (1,367, lenient): B không thua STD có ý nghĩa (4) — đây là chỗ G0 thất bại.
  - B4 mất dấu tự nhiên: khoảng cách C − N của B nhỏ hơn của STD (3 model có PS checkpoint).
  - Holm chung 19 phép so. **THÀNH CÔNG** khi B1, B2, B3 đạt ở mọi model đo được.
- Mô tả thêm: trên 70 câu G0 gắn cờ ở ViGo test, B có làm hại như G0 không.

## Chạy
vast RTX 4090 bằng `setup_vast.sh` (torch 2.9.1 cu128): 18 run A + 9 run PS; chấm valid nhiễu và tập tự nhiên trên máy
(ViGoEmotions chỉ xử lý trên máy thuê, không phân phối). Phân tích trên laptop.

## Nhật ký thay đổi
(trống)
