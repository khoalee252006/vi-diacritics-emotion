# Thiết kế CHỐT TRƯỚC: giả thuyết "dạng lạ" (2026-10-14)

**Nguồn gốc (trung thực):** giả thuyết phát sinh SAU khi xem H4 (review/08): các từ cảm xúc gây hại nhiều nhất gồm cả
từ không mơ hồ (khóc, buồn). Vì vậy H4 KHÔNG được dùng làm bằng chứng cho giả thuyết này; bằng chứng phải đến từ các
phép thử dưới đây, chốt trước khi chạy.

**Giả thuyết G:** bỏ dấu làm hại chủ yếu vì dạng không dấu là dạng **lạ** với model đã fine-tune trên văn bản có dấu
(hiếm gặp trong dữ liệu train; bị tokenizer cắt khác đi), chứ không chủ yếu vì mơ hồ từ vựng.

## D1 — Quan sát (dữ liệu sẵn có: 7,074 biến thể một âm tiết của Phần 1)
- Biến kết quả: Δp(nhãn đúng) (điểm %, p̄ theo seed), như Phần 1.
- Độ lạ, tính TRƯỚC khi nhìn Δp:
  - U1 = log(1 + số lần dạng không dấu k xuất hiện như một âm tiết trong VSMEC train) (độ quen thuộc; càng lớn càng quen);
  - U2 = (số subword của k) − (số subword của dạng có dấu w) theo tokenizer của TỪNG model (PhoBERT: tokenizer BPE
    trên âm tiết); U2 > 0 = bị cắt vụn hơn.
- Mô hình: `dp ~ U1 + U2 + confus + emo + n_syl + rel_pos + log_freq(w)`, sai số chuẩn cụm theo câu, 4 PLM.
- Dự đoán của G: hệ số U1 < 0 và U2 > 0.
- Kiểm định: 4 PLM × 2 hệ số (U1, U2) = 8 phép so, Holm 8. Phụ (mô tả): |t| của U1/U2 so với confus.
- G được ủng hộ ở D1 khi: U1 < 0 có ý nghĩa ở ≥ 3/4 PLM.

## D2 — Can thiệp bằng huấn luyện (bằng chứng nhân quả; cũng là ứng viên phương pháp)
- Ý tưởng: làm cho dạng không dấu của MỘT tập từ W trở nên "quen" bằng cách cho model thấy chúng khi fine-tune,
  không động tới các từ khác.
- Điều kiện huấn luyện (cùng công thức RQ1: lr 2e-5, batch 16, 20 epoch, chọn checkpoint theo valid sạch):
  - **T-emo:** mỗi câu train được nhân đôi; bản sao bỏ dấu mọi lần xuất hiện của các từ thuộc W_emo
    (300 từ cảm xúc của 6 lớp, Phần 1). Câu không chứa từ nào của W_emo thì bản sao giữ nguyên.
  - **T-rand (đối chứng):** như trên nhưng W_rand = tập âm tiết có dấu KHÔNG phải từ cảm xúc, ghép cặp với W_emo
    theo tần suất trong train (cùng bin log-tần suất), cùng kích thước → cùng lượng token bị bỏ dấu (±10%, kiểm tra trước khi train).
  - Mốc có sẵn: model thường (RQ1 P0), PS, augment hỗn hợp (rq3_aug).
- Model: XLM-R, PhoBERT, ViSoBERT × 3 seed (42–44) × 2 điều kiện = 18 run (CafeBERT không train được trên GPU 8 GB).
- Đánh giá (test chấm một lần): sạch; bỏ dấu 25/50/75/100%; 7,074 biến thể một âm tiết (Δp tách theo từ ∈ W_emo / ∉);
  ViGoEmotions mới (lenient).
- Dự đoán của G:
  (P1) T-emo giảm Δp trên biến thể bỏ dấu từ cảm xúc nhiều hơn T-rand;
  (P2) T-emo tăng MF1 ở bỏ dấu 100% hơn T-rand;
  (P3) T-emo không làm giảm MF1 trên câu sạch so với model thường.
- Kiểm định: bootstrap hai tầng; P1: 3 model; P2: 3 model; P3: 3 model (T-emo vs thường); Holm chung 9.
- Kết luận nhân quả "có" khi P1 và P2 có ý nghĩa ở ≥ 2/3 model.
- Ứng viên đóng góp phương pháp (chỉ nêu nếu đạt): "phơi nhiễm dấu có mục tiêu" — chỉ cần bỏ dấu ~300 từ cảm xúc lúc
  train, giữ độ chính xác câu sạch, một model duy nhất, không cần bộ phát hiện. So thêm (mô tả) với augment hỗn hợp
  và gating về MF1 bỏ dấu 100% và câu sạch.

## Rủi ro đã biết
- W_emo lấy từ hệ số LR (Phần 1) → lựa chọn từ phụ thuộc dữ liệu train, nhưng không dùng test → không rò rỉ.
- Nhân đôi dữ liệu làm số bước train gấp đôi ở cả T-emo và T-rand → so sánh công bằng giữa hai điều kiện; so với
  model thường (ít bước hơn) chỉ là mô tả.

## Nhật ký thay đổi
- 2026-10-14, kết quả D1: KHÔNG ủng hộ G (U1 có ý nghĩa 0/4, U2 0/4; `results/unfamiliar/d1_regression.csv`).
- 2026-10-14, TRƯỚC khi chạy D2 (chưa có kết quả D2): danh sách "từ cảm xúc" của Phần 1 (top-50 hệ số LR mỗi lớp) có
  120/300 từ thuộc 200 âm tiết phổ biến nhất (cho, vào, thì, làm, được…): đó là **từ đặc trưng cho lớp**, không hẳn là
  từ chỉ cảm xúc. Nếu giữ nguyên, W_emo phủ 5,262/5,548 câu train và không ghép được W_rand (lệch −27% token).
  → Định nghĩa lại cho D2: W_emo = từ của 6 lớp cảm xúc, CÓ dấu, có trong train, KHÔNG thuộc 200 âm tiết phổ biến nhất
  (138 từ, 3,928 token, 2,507 câu); W_rand = 138 âm tiết có dấu không thuộc danh sách từ cảm xúc và không thuộc top-200,
  ghép gần nhất theo log-tần suất (3,927 token, 2,450 câu; lệch 0.0%). Tệp: `results/unfamiliar/W_{emo,rand}.txt`.
- D2 chạy trên vast RTX 4090 bằng `setup_vast.sh` (torch 2.9.1 cu128, cùng bộ phần mềm với RQ1). Bootstrap P1 theo cụm
  câu 2,000 vòng; P2, P3 bootstrap hai tầng 10,000 vòng.
- Trong bài phải gọi đúng tên: "từ đặc trưng cho lớp cảm xúc" (class-indicative words), kể cả ở Phần 1.
