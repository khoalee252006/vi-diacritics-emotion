# Thiết kế CHỐT TRƯỚC: thí nghiệm bổ sung theo nhận xét của thầy (2026-10-17)

Nguồn: nhận xét của thầy (phân tích ở `review/16_phan_tich_nhan_xet_thay.md`). User đồng ý: thuê GPU, dùng GPU laptop.

**Công khai những gì ĐÃ XEM trước khi viết thiết kế này (kiểm tra nhanh, thăm dò, 2026-10-17):**
- KL vs nhân đôi thường với 3 seed: 0/12 có ý nghĩa sau Holm, chiều dương 9/12 (100%: XLM-R +2.9, PhoBERT +3.1, ViSoBERT +1.2).
- ECE của dual-view (KL) thấp hơn model thường (100%: 56.7→28.0, 54.1→29.1, 47.6→32.1).
- Ô "class-indicative không mơ hồ" ở cấp âm tiết (đã có từ Phần 1): 0.9–2.5 điểm so với 3.5–6.6.
Các kiểm định dưới đây vì vậy là xác nhận trên dữ liệu MỚI (seed mới, CafeBERT, điều kiện mới), không phải trên chính
các số đã xem; phần dùng lại 3 seed cũ được ghi rõ.

## E1 — Thêm seed và CafeBERT cho dual-view (GPU thuê)
- Train: `dual` (nhân đôi, bản sao bỏ hết dấu) và `dualkl` (λ = 1) cho XLM-R, PhoBERT, ViSoBERT seed 45, 46 (→ 5 seed,
  bằng model thường); CafeBERT seed 42–44 (bằng CafeBERT thường). Cùng công thức, chọn checkpoint trên valid sạch.
- **NI (non-inferiority), biên δ = −2 MF1** (lấy theo ví dụ của thầy, chốt TRƯỚC khi có seed mới):
  - NI1 câu sạch: dualkl − thường; NI2 bỏ dấu 100%: dualkl − PS. 4 model × 2.
  - Đạt khi cận dưới CI 95% (bootstrap hai tầng, 10,000 vòng) > −2. Báo cả CI; không Holm (mỗi model một khẳng định riêng),
    nhưng báo thêm CI 98.75% (Bonferroni 4) để thận trọng.
- **KL1 (KL có giúp không):** dualkl − dual tại sạch, 75%, 100%; 4 model × 3 = 12, Holm 12.
  KL "có tác dụng về độ chính xác" khi có ý nghĩa dương ở 75% hoặc 100% với ≥ 2/4 model.
- **CAL (hiệu chỉnh):** ECE thô (15 bin) tại bỏ dấu 100%: dualkl − thường và dualkl − dual; 4 model × 2 = 8, Holm 8.
  Bootstrap: lấy mẫu lại câu (chung) và seed (riêng từng hệ). "Dual-view giảm silent failure" khi dualkl < thường có
  ý nghĩa ≥ 3/4. "KL là nguồn cải thiện hiệu chỉnh" khi dualkl < dual có ý nghĩa ≥ 2/4.
- Baseline augment cân sức = `dual` (bản sao bỏ hết dấu của mọi câu). Augment hỗn hợp giữ nguyên làm baseline phụ, mô tả
  đúng cấu hình: THAY mỗi câu bằng 1 bản nhiễu, loại nhiễu chọn đều trong 4 loại (25% câu bị bỏ dấu, 6.4% bỏ hết dấu).

## E2 — Độ nhạy λ (GPU thuê)
- `dualkl` với λ ∈ {0.5, 2}, XLM-R/PhoBERT/ViSoBERT, seed 42–44. Chỉ MÔ TẢ (valid + test, sạch và 25–100%).
- Phương pháp chính GIỮ λ = 1 dù λ khác tốt hơn (không chọn lại sau khi xem test).

## E3 — LLM mở lớn (GPU thuê, ≥ 48 GB VRAM)
- Qwen2.5-72B-Instruct-AWQ qua vLLM, đúng giao thức cũ (prompt, nhãn, temperature 0, max_tokens 8, 0/7-shot; sạch,
  bỏ dấu 50%, 100%). Không nạp được thì dùng Qwen2.5-32B-Instruct-AWQ (ghi rõ).
- Kiểm định như `src/llm_extra.py`: vs model thường (sạch, 100%), vs PS (100%); Holm trong họ mới.
- Diễn giải chốt trước: "bền" khi drop(sạch→100%) ≤ 6 điểm (mức của họ Google). Nếu 72B bền → sửa kết luận thành
  "độ bền tăng theo cỡ model", không còn "không phải đặc tính chung của LLM".

## E4 — Few-shot với ví dụ không dấu (API Gemini)
- Gemini 3.5 Flash và Flash-Lite, 7-shot, ví dụ bỏ hết dấu, chỉ trên test bỏ dấu 100%. So với 7-shot ví dụ có dấu:
  bootstrap câu (một lần chạy), Holm 2. Không chạy Gemma (gói free).

## E5 — Điều kiện "class-indicative KHÔNG mơ hồ" ở cấp câu (GPU laptop, checkpoint RQ1, không train)
- 434 câu test có ≥ 1 từ class-indicative mơ hồ VÀ ≥ 1 từ class-indicative không mơ hồ (confus < 0.2, có dấu).
  Mỗi câu: k = min(số mơ hồ, số không mơ hồ); amb_k = bỏ dấu k từ mơ hồ, clear_k = bỏ dấu k từ không mơ hồ (chọn ngẫu
  nhiên khi thừa, 5 lần rút mỗi điều kiện, seed 0).
- Kiểm định: MF1(amb_k) − MF1(clear_k), bootstrap hai tầng, 6 hệ, Holm 6.
  "Độ mơ hồ gây hại thêm ngoài tính class-indicative" khi âm có ý nghĩa ở ≥ 3/4 PLM; nếu không → viết lại cơ chế thành
  "dạng không dấu của từ khóa" (thầy đề xuất), mơ hồ chỉ là yếu tố phụ.
- Phụ: clear_k vs ngẫu nhiên cùng số âm tiết trên 513 câu có ≥ 1 từ không mơ hồ (Holm 6, mô tả nếu không chốt được).

## E6 — Chọn từ theo từng model bằng occlusion (GPU laptop)
- Mỗi PLM, checkpoint seed 42: trên TRAIN (không dùng test), độ quan trọng của một lần xuất hiện âm tiết có dấu =
  p(gold | câu) − p(gold | câu bỏ âm tiết đó). Gộp theo từ trong các câu thuộc lớp c, chỉ từ xuất hiện ≥ 5 lần trong lớp;
  top-50 mỗi lớp cảm xúc (6 lớp) → danh sách riêng của model. Mơ hồ = confus ≥ 0.2 (như cũ).
- Lặp lại can thiệp cấp câu (từ mơ hồ của danh sách riêng vs ngẫu nhiên vs mơ hồ khác, 5 lần rút) trên test, mọi seed
  của model đó. 4 PLM × 2 = 8, Holm 8. Đạt khi vs ngẫu nhiên có ý nghĩa ≥ 3/4 PLM.

## E7 — Phân tích CPU (không train)
- Khôi phục dấu có gắn cờ: p = w·p_thường + (1 − w)·p_khôi_phục, w từ bộ phát hiện (như detector gate). 4 PLM; so với
  thường (sạch) và khôi phục toàn bộ (sạch, 100%); Holm 12. Viết "with the restorer we use".
- ViGo strict (bỏ câu đa nhóm) cho bảng chính, bên cạnh lenient.
- Dual-view vs LLM (sạch, 100%): 3–4 model dual-view × 6 LLM-cấu hình Google × 2 mức, Holm trong họ.
- Bảng theo lớp (class_drop.csv): mức giảm F1 theo lớp khi bỏ dấu 100%.
- Tập tự nhiên: acc(N) là chỉ số chính, gap là phụ.

## Thứ tự chạy
1. Viết code + gói cho vast (E1, E2, E3) → user thuê máy → chạy nền.
2. Song song trên laptop: E5, E6. 3. CPU: E7, E4 (API). 4. Phân tích theo đúng tiêu chí trên, rồi sửa bài.

## Nhật ký thay đổi
(trống)

## Kết quả E5, E6 (2026-10-17, GPU laptop RTX 5070, checkpoint RQ1; `results/ambiguity/intervention_rev/`)
- **E5 ĐẠT (4/4 PLM, cả 6/6 hệ):** trên 434 câu, bỏ dấu k từ class-indicative MƠ HỒ làm MF1 thấp hơn bỏ dấu k từ
  class-indicative KHÔNG mơ hồ: CafeBERT −9.2 [−13.3, −5.0], XLM-R −7.6, PhoBERT −6.9, ViSoBERT −7.0, LR −10.9,
  SVM −10.0 (p_holm ≤ 0.002). Phụ (513 câu): bỏ dấu từ class-indicative không mơ hồ ≈ bỏ dấu cùng số âm tiết ngẫu
  nhiên (−1.2..+4.3, 0/6 có ý nghĩa) → "dạng không dấu lạ của từ khóa" KHÔNG giải thích được; độ mơ hồ là yếu tố cần.
- **E6 ĐẠT (4/4):** danh sách từ chọn theo occlusion của TỪNG model (235–254 từ, 65–68% mơ hồ): bỏ dấu từ mơ hồ của
  danh sách riêng hại hơn ngẫu nhiên −7.0 đến −11.6 và hơn mơ hồ khác −7.3 đến −11.2 (p_holm ≤ 0.014) → kết quả Phần 1
  không do cách chọn từ bằng LR.
- **E4 (2026-10-17, API Gemini):** 7-shot với ví dụ KHÔNG dấu trên test bỏ dấu 100%: Flash 56.49 vs 54.53 (ví dụ có dấu),
  +2.0 [−1.2, 5.2], p_holm 0.44; Flash-Lite 52.51 vs 51.83, +0.7 [−2.6, 4.0], p_holm 0.69 → 0/2 có ý nghĩa: cách chọn
  ví dụ (có/không dấu) không thay đổi kết luận về LLM. `results/llm/rev_fewshot_strip.csv`, tag `*_stripex`.
- **E7 khôi phục có gắn cờ:** sạch ≈ thường (|Δ| ≤ 0.17, 0/4), 100% không khác khôi phục toàn bộ (−1.0..−1.8, 0/4);
  MF1 100%: 49.1 / 48.4 / 44.1 / 45.3 (CafeBERT/XLM-R/PhoBERT/ViSoBERT). `results/ab/rev_restore_*.csv`.

## Nhật ký chạy E1–E3 (2026-10-17, vast 2× RTX 4090 48GB, CUDA 13.2, torch 2.9.1+cu128; gói md5 5d837725..., kết quả md5 eb14cd4e...)
- 36/36 run train xong (results/dual: +18; results/dual_lam: 18), mỗi run 25 file dự đoán; Qwen2.5-72B-Instruct-AWQ 4158/4158 lượt.
- **Sự cố (ghi để minh bạch):** lệnh mở 2 hàng đợi phụ mà user TỪ CHỐI đã kịp chạy trên máy trước khi bị ngắt
  (lỗi quy trình của Claude: lẽ ra phải đề xuất và chờ đồng ý trước). Hàng phụ C (ViSoBERT λ) lỗi cú pháp, không chạy
  gì. Hàng phụ D chạy CafeBERT dual + dualkl seed 44 trên card 1 (08:18–09:28 UTC), song song với hàng B; hàng A sau đó
  bỏ qua 2 run này ([skip] đã xong). Mỗi run train đúng MỘT lần, cùng seed/công thức → không ảnh hưởng kết quả; chỉ khác
  card và chạy chung card (chậm hơn). Log: results/llm/vast_logs/rev_extra_d.log.

## Kết quả E1–E3 theo tiêu chí chốt trước (2026-10-17, `src/rev_analyze.py`, log `results/ab/rev_analyze.log`)
- MF1 5 seed (CafeBERT 3), sạch / bỏ dấu 100%: dualkl XLM-R 60.38/51.31, PhoBERT 61.54/48.98, ViSoBERT 57.97/50.95,
  CafeBERT 62.72/54.71; nhân đôi thường (dual) 61.14/49.45, 60.34/44.99, 55.73/48.82, 64.78/50.62.
- **NI1 (sạch, biên −2): 0/4 ĐẠT.** Δ −2.0 [−5.2, 1.0], +1.1 [−2.1, 4.3], −0.8 [−3.7, 2.2], −0.8 [−4.7, 3.1] → không
  loại trừ được mức giảm 2–5 điểm; KHÔNG được viết "keeps clean accuracy" như tương đương.
- **NI2 (100% vs PS, biên −2): 3/4 ĐẠT** (XLM-R +3.3, PhoBERT +4.2, CafeBERT +6.8 [3.4, 10.3]); ViSoBERT −0.6 [−3.7, 2.4] không đạt.
- **KL1: 1/4 → KHÔNG ĐẠT** (chỉ PhoBERT 100% +4.0, p_holm 0.041; CafeBERT 100% +4.1 p_holm 0.079). Chiều dương 9/12.
- **CAL: ĐẠT cả hai.** ECE 100%: dualkl 26.6–29.2 vs thường 47.6–56.7 (4/4, p_holm ≤ 0.016); dualkl vs nhân đôi thường
  −11.0 (XLM-R), −12.6 (PhoBERT) có ý nghĩa, ViSoBERT −6.6, CafeBERT −8.4 (p_holm 0.076) → 2/4 → vai trò của KL là HIỆU CHỈNH.
- **E2 λ:** valid gần như phẳng (50.9–52.7); λ = 2 nhỉnh hơn ở 100% cho XLM-R/PhoBERT (+1.4/+0.6) → không nhạy với λ.
- **E3 Qwen2.5-72B-AWQ:** sạch 54.1/55.4, 100% 45.1/46.5 (0/7-shot) → mức giảm ~9 điểm > 6 → KHÔNG "bền" theo tiêu chí;
  100% hơn mọi model thường 8/8 (p_holm ≤ 0.003); vs hệ không dấu: không hơn hệ nào, THUA LR-PS (p_holm 0.006 / 0.050).
- **E7 dual-view vs LLM (2026-10-17, `src/rev_cpu.py dvllm`, Holm 64):** gần như không khác: dual-view HƠN 4/64
  (CafeBERT/PhoBERT vs Flash-Lite 0-shot câu sạch; CafeBERT vs Qwen-72B 0-shot ở sạch và 100%), THUA 1/64 (PhoBERT vs
  Gemma 7-shot ở 100%, −9.2). `results/llm/rev_dualview_vs_llm.csv`.
- Bảng/hình bản B: `python -m src.paper_v2` (5 seed, CafeBERT, ECE, ablation, E5/E6, strict, theo lớp); main_fix.csv.
  Mô tả thêm (không chốt trước): CafeBERT dual-view trên tập tự nhiên N: 39.0 vs thường 26.6.
