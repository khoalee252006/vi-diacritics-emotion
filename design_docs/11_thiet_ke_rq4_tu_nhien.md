# Thiết kế CHỐT TRƯỚC: RQ4 — mất dấu tự nhiên (2026-10-14)

## Dữ liệu
- **N (mất dấu tự nhiên):** các câu trong `review/rq4_natural_candidates.xlsx` có nhãn cuối `mat_dau_that`
  (nhãn Claude lần 2, nhóm sửa đè ở cột `nhom_kiem_tra`): 94/106 câu. Bỏ câu không có nhóm nhãn sau ánh xạ a.
- **C (đối chứng có dấu):** câu thuộc phần mới ViGoEmotions (mọi split), bộ lọc cho miss = 0 và acc ≥ 3 (đủ dấu),
  có nhóm nhãn sau ánh xạ a, không thuộc N. Với mỗi câu của N chọn 3 câu C: cùng nhóm nhãn đại diện (canonical),
  độ dài (số âm tiết bộ lọc đếm) lệch ≤ 3; thiếu thì nới ≤ 6; không lặp lại; seed 0.
- **C_strip:** C bị bỏ hết dấu (nhiễu tổng hợp trên cùng câu) — để so nhiễu tự nhiên với nhiễu tổng hợp.
- Ghi chú: mọi model train trên VSMEC → khoảng cách miền VSMEC→ViGo tác động đều lên N và C (đã kiểm soát bằng ghép cặp).

## Hệ được chấm (không train thêm)
Model thường P0: CafeBERT (3 seed), XLM-R, PhoBERT, ViSoBERT (5 seed), LR, SVM.
Model không dấu PS: LR, SVM (có `model.joblib`; checkpoint PS của PLM không được lưu).
Gating tỉ lệ dấu (LR, SVM: P0 + PS).

## Chỉ số
- Chính: **độ chính xác lenient** (dự đoán thuộc nhóm nhãn vàng) — vì N nhỏ, MF1 7 lớp không ổn định.
- Phụ: MF1 lenient.

## Giả thuyết và kiểm định
- **N1:** với model thường, độ chính xác trên N thấp hơn trên C. 6 model (4 PLM + LR + SVM).
  Thống kê: acc(N) − acc(C); bootstrap lấy mẫu lại câu của N, câu của C độc lập, và seed; 10,000 vòng. Holm 6.
- **N2:** bỏ dấu tổng hợp làm giảm nhiều hơn mất dấu tự nhiên? Mô tả: [acc(C) − acc(C_strip)] so với [acc(C) − acc(N)]
  cho 6 model (không kiểm định; mất dấu tự nhiên thường chỉ một phần nên dự kiến nhẹ hơn).
- **N3:** model không dấu (PS) thu hẹp khoảng cách: khoảng cách (C − N) của PS nhỏ hơn của P0, cho LR và SVM;
  bootstrap như N1; Holm 2.
- Kết luận "mất dấu tự nhiên gây hại" khi N1 có ý nghĩa ở ≥ 4/6 model.

## Hạn chế biết trước
N nhỏ (94); nhãn mất dấu do LLM gán, nhóm kiểm tra (không mù); nhãn cảm xúc ViGoEmotions có gợi ý từ LLM;
bộ lọc ưu tiên độ chính xác nên N lệch về câu mất dấu rõ, ngắn hơn trung bình.

## Nhật ký thay đổi
(trống)
- 2026-10-14: phát hiện lỗi Unicode NFD trong bộ lọc (dấu tách rời bị đọc như không dấu). Sửa bằng chuẩn hóa NFC.
  Ảnh hưởng: 3/106 ứng viên ViGo bị loại khi chạy lại — đúng 3 câu đã được gán `khong_phai: da_co_dau`; tập N giữ nguyên 94 câu.
- 2026-10-14: kiểm định bộ lọc (ngưỡng chốt trước) trên toàn bộ ViLexNorm (đáp án tự động từ cặp câu gốc/chuẩn hóa do người
  của nhóm ViLexNorm tạo): precision 88.1% [75.0, 94.8], recall 74.0% [60.4, 84.1] (Wilson 95%).
