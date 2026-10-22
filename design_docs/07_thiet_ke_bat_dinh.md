# Thiết kế CHỐT TRƯỚC: đo độ bất định khi mất dấu (2026-10-14)

Mục đích: bổ sung chiều "độ tin cậy / bất định" cho câu hỏi của thầy *"Why missing diacritics break Vietnamese emotion
recognition"*. Mọi giả thuyết, chỉ số, phép kiểm định và quy tắc kết luận dưới đây được CHỐT TRƯỚC khi chạy; thay đổi
sau khi có kết quả phải ghi lại ở mục 8 kèm lý do.

## 0. Dữ liệu và ràng buộc chung
- **Dự đoán sẵn có (không cần chạy lại):** `results/rq1/<m>_P0_none_s<seed>/pred_{test,valid,diac25..100}.csv`,
  `results/strip/<m>_PS_none_s<seed>/pred_*.csv`, `results/ambiguity/intervention/<run>/pred_{single,single_clean,multi_*}.csv`.
  Mỗi file có xác suất 7 nhãn `s_<nhãn>`.
- **Model:** 4 PLM (CafeBERT 3 seed; XLM-R, PhoBERT, ViSoBERT 5 seed) là đối tượng chính. LR có xác suất (đưa vào phụ).
  SVM chỉ có điểm, không phải xác suất → **loại khỏi mọi phân tích hiệu chỉnh**; chỉ dùng ở chỗ cần thứ hạng.
- **Ghép theo VỊ TRÍ dòng** (id test VSMEC bị trùng: 693 dòng/608 id).
- **Hạn chế đã biết của dữ liệu:** xác suất lưu với 5 chữ số thập phân → entropy có sàn nhỏ (không ảnh hưởng so sánh);
  model fine-tune rất tự tin (p tối đa thường > 0.999) nên ECE thô có thể lớn ngay trên câu sạch.
- **LLM:** chỉ có nhãn, không có xác suất → KHÔNG nằm trong thiết kế này (xem mục 7).
- **Họ phép so Holm:** mỗi giả thuyết H1–H5 là một họ riêng; ghi rõ số phép so trong bảng kết quả.
- **Bootstrap:** hai tầng (câu dùng chung + seed), 10,000 vòng, seed ngẫu nhiên 0, như `src/stats.py`.

## 1. H1 — Hiệu chỉnh: model có "tự tin sai" khi mất dấu không?
- **Giả thuyết:** sai lệch hiệu chỉnh tăng theo mức bỏ dấu (0 → 25 → 50 → 75 → 100%).
- **Chỉ số:** ECE (15 bin bằng nhau theo độ tin cậy), Brier đa lớp, NLL; độ tin cậy trung bình trên câu SAI.
- **Hai phiên bản:** (a) xác suất thô; (b) sau **temperature scaling** fit trên `pred_valid` (sạch) của từng run
  — để biết lỗi hiệu chỉnh do mất dấu có còn sau khi đã hiệu chỉnh trên dữ liệu sạch không (đây là câu hỏi chính).
- **Kiểm định:** ΔECE(bỏ dấu 100% − sạch) cho từng PLM, bootstrap hai tầng; Holm trên 4 PLM × {thô, đã hiệu chỉnh} = 8 phép so.
- **Kết luận "có" khi:** ΔECE > 0 có ý nghĩa ở ≥ 3/4 PLM ở phiên bản (b).
- **Sản phẩm:** `results/uncertainty/calibration.csv`, biểu đồ reliability (sạch vs bỏ dấu 100%) cho 4 PLM.

## 2. H2 — Dự đoán có chọn lọc: bất định có lọc được lỗi do mất dấu không?
- **Tín hiệu bất định (trên mỗi câu):**
  - MSP: 1 − max p (của trung bình các seed);
  - Entropy dự đoán H[p̄] (tổng bất định);
  - Độ bất đồng giữa seed = mutual information = H[p̄] − mean_s H[p_s] (bất định của model, ensemble theo seed).
- **Chỉ số:** AURC (diện tích dưới đường risk–coverage; thấp = tốt), và AUROC phân biệt câu sai/đúng.
- **Đối chứng:** xếp hạng ngẫu nhiên (AURC kỳ vọng = tỉ lệ lỗi).
- **Kiểm định:** trên bỏ dấu 100%, so từng cặp tín hiệu (MI vs entropy, MI vs MSP, entropy vs MSP) cho 4 PLM =
  12 phép so, bootstrap theo câu (seed đã gộp vào tín hiệu); Holm 12.
- **Câu hỏi phụ (mô tả, không kiểm định):** AUROC trên sạch vs bỏ dấu — tín hiệu có kém đi khi mất dấu không.

## 3. H3 — Bất định có đến từ độ mơ hồ của từ không? (nối Phần 1)
Dùng lại đúng các tập can thiệp đã có (Phần 1), chỉ đổi biến kết quả.
- **Cấp âm tiết** (7,074 biến thể, mỗi biến thể bỏ dấu đúng 1 âm tiết):
  biến kết quả ΔH = H(biến thể) − H(câu sạch) (entropy của p̄ theo seed) và ΔMI.
  Hồi quy `ΔH ~ emo * amb + collision + n_syl + rel_pos + log_freq`, sai số chuẩn cụm theo câu — giống hệt mô hình Phần 1.
- **Giả thuyết:** hệ số tương tác `emo:amb` > 0 (bỏ dấu từ cảm xúc mơ hồ làm model phân vân hơn ba loại từ còn lại).
- **Cấp câu** (416 câu): ΔH khi bỏ dấu k từ cảm xúc mơ hồ so với bỏ dấu k âm tiết ngẫu nhiên / k âm tiết mơ hồ không cảm xúc.
- **Kiểm định:** cấp âm tiết: `emo:amb` cho 4 PLM (+LR phụ); cấp câu: 4 PLM × 2 đối chứng; Holm chung 12 phép so.
- **Kết luận "có" khi:** `emo:amb` > 0 có ý nghĩa ở ≥ 3/4 PLM.

## 4. H4 — Bất định của TỪNG TỪ (hai loại, tách rõ)
### 4a. Bất định của chính từ (không cần model phân loại)
- **(I-0) Không ngữ cảnh:** đã có trong `lexicon.csv` (entropy H(dạng có dấu | dạng không dấu), confus).
- **(I-1) Ngữ cảnh rẻ:** entropy có điều kiện theo âm tiết trước và sau, đếm trên cùng corpus train
  (VSMEC train + ViGo new train), làm trơn add-k (k = 0.1; chốt trước, không dò).
- **(I-2) Ngữ cảnh mạnh:** xác suất từng dạng có dấu do model khôi phục dấu `peterhung/vietnamese-accent-marker-xlm-roberta`
  cho ra tại vị trí đó trong câu test → entropy theo ngữ cảnh. Chạy GPU laptop, chỉ suy luận.
  (Chỉ làm I-2 nếu I-1 cho thấy ngữ cảnh làm giảm mơ hồ ≥ 20% so với I-0 trên từ cảm xúc — chốt trước.)
### 4b. Bất định mà từ gây ra cho model (II)
- Trên 7,074 biến thể một âm tiết: ΔH, Δp(nhãn đúng), ΔMI; gộp theo loại từ → bảng xếp hạng các từ gây bất định
  nhiều nhất cho từng model và lớp cảm xúc (chỉ từ xuất hiện ≥ 5 lần).
### 4c. Câu hỏi chính
- **Giả thuyết:** (I) theo ngữ cảnh giải thích (II) tốt hơn (I) không ngữ cảnh.
- **Kiểm định:** so R² (hoặc tương quan Spearman) của mô hình `ΔH ~ I-0` với `ΔH ~ I-1` (và `~ I-2` nếu chạy), trên cùng
  biến thể, bootstrap theo câu; Holm trên 4 PLM × số cặp so.
- **Sản phẩm:** `results/uncertainty/word_uncertainty.csv` (danh sách từ + I-0/I-1/I-2 + ΔH/Δp/ΔMI theo model) — có thể
  công bố như tài nguyên (chỉ từ và số, không chứa câu ViGoEmotions).

## 5. H5 — Gating theo bất định (phương pháp, sửa chỗ gating theo tỉ lệ dấu thất bại)
- **Bối cảnh:** gating theo tỉ lệ dấu thắng trên nhiễu tổng hợp nhưng hại trên dữ liệu thật (13/70 câu bị gắn cờ là mất dấu thật).
- **Phương pháp:** `p = (1−g)·p_std + g·p_free`, với g = σ((u − τ)/s), u = bất định của model thường (tín hiệu tốt nhất
  theo H2, chọn trên VALID, không xem test). Hai tham số τ, s chọn trên tập valid.
- **Cần làm trước:** tạo tập **valid nhiễu** (bỏ dấu 25/50/75/100%, cùng bộ sinh nhiễu với test, seed khác) và chấm
  bằng checkpoint P0 và PS sẵn có trên GPU laptop (chỉ suy luận; CafeBERT suy luận được trên 8 GB).
  Dữ liệu chọn τ, s = valid sạch + valid nhiễu, trộn đều các mức (chốt trước).
- **So sánh (test, chấm một lần):** model thường; PS; gating tỉ lệ dấu; gating bất định; trung bình đều (avg).
  Tập: sạch, bỏ dấu 25/50/75/100%, và ViGoEmotions mới (1,367 câu; toàn tập + 70 câu bị gắn cờ).
- **Kiểm định:** gating bất định vs gating tỉ lệ dấu và vs model thường, 4 PLM × 6 tập test × 2 = 48 phép so, Holm 48.
- **Tiêu chí thành công (chốt trước):** (1) không thua model thường có ý nghĩa trên câu sạch ở PLM nào;
  (2) không thua gating tỉ lệ dấu có ý nghĩa ở bỏ dấu 75–100%;
  (3) trên ViGoEmotions: không làm giảm MF1 nhóm 70 câu bị gắn cờ như gating cũ (PhoBERT −9.6), và toàn tập ≥ model thường.
  Đạt (1)+(2)+(3) mới gọi là cải thiện; đạt một phần thì báo cáo đúng phần đạt.
- **Rủi ro biết trước:** fine-tune rất tự tin → u có thể kém phân biệt; nếu AUROC (H2) trên bỏ dấu 100% < 0.6 thì
  dừng H5 và báo cáo là kết quả âm tính.
- **Giấy phép:** ViGoEmotions chỉ chạy cục bộ, không gửi lên API.

## 6. Thứ tự chạy và chi phí
| Bước | Việc | Phần cứng | Ước tính |
|---|---|---|---|
| 1 | H1, H2 trên dự đoán sẵn có | CPU | vài giờ |
| 2 | H3, H4b trên tập can thiệp sẵn có | CPU | vài giờ |
| 3 | H4a I-1 (đếm ngữ cảnh) → quyết định I-2 → H4c | CPU (+ GPU laptop nếu I-2) | ~nửa ngày |
| 4 | Tạo valid nhiễu + suy luận P0/PS | GPU laptop | ~1–2 giờ |
| 5 | H5: chọn τ, s trên valid → chấm test + ViGo một lần | CPU | vài giờ |
Không cần thuê vast. Không gọi API.

## 7. Ngoài phạm vi (ghi ở hạn chế)
- Bất định của LLM: cần logprob (vLLM có; Gemini qua endpoint OpenAI chưa xác minh) hoặc lấy mẫu nhiều lần → để sau.
- MC-dropout: không làm (đã có ensemble theo seed, rẻ và chuẩn hơn).
- Danh sách từ cảm xúc lấy từ hệ số LR → kết quả của LR có tính vòng lặp; bằng chứng chính là 4 PLM.

## 8. Nhật ký thay đổi sau khi chốt
- 2026-10-14, H4 (lỗi dữ liệu, không phải đổi thiết kế): `lexicon.csv` có từ "nan"; pandas đọc thành NaN → 7 dòng
  (nạn, nản, nần, năn, nặn, nàn, nan) bị mất khóa. Đã sửa bằng `keep_default_na=False` trong `src/uncertainty.py`,
  `src/ambiguity.py`, `src/intervention.py`. Ảnh hưởng tới Phần 1: 3/7,074 biến thể (giá trị collision) — chưa chạy lại Phần 1.
- H4a I-1: âm tiết lân cận lấy ĐÚNG như trong câu biến thể (vẫn có dấu, vì chỉ một âm tiết bị bỏ dấu).
- H4a I-2 (xấp xỉ, chốt trước khi xem kết quả I-2): p(c) ∝ Σ_subword Σ_{tag t: áp t lên dạng không dấu ra c} p(t);
  chuẩn hóa trên các ứng viên trong lexicon.
- H4c: bootstrap theo cụm câu 2,000 vòng (thay vì 10,000) vì mỗi vòng tính lại hệ số Spearman trên 7,074 dòng.
- H5: PS bỏ hết dấu ở đầu vào nên dự đoán không phụ thuộc mức bỏ dấu (đã kiểm: `pred_diac*` trùng `pred_test` với
  chênh xác suất 0) → valid nhiễu dùng `pred_valid.csv` của PS; chỉ chấm lại model thường (P0) trên valid nhiễu.
  Gating áp theo từng seed (P0 seed s ghép PS seed s, như gating cũ); tín hiệu u chỉ còn MSP và entropy của seed đó
  (MI cần nhiều seed nên không dùng được trong gating theo seed). Lưới chọn: τ = 19 phân vị (5%–95%) của u trên valid,
  s ∈ {0.01, 0.03, 0.1, 0.3} × khoảng giá trị của u; mục tiêu = MF1 trung bình đều 5 tập valid.
- H5 trên ViGo (chấm lenient): bootstrap câu + seed 2,000 vòng, gold hiệu dụng tính riêng cho từng hệ.
