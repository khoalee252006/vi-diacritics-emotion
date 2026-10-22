"""Chú thích tay lỗi khôi phục dấu (theo cách phân tích lỗi của Náplava et al., 2021, PBML 116, tr.1–2).

Tạo file:  python -m src.restore_annotation make     -> review/restore_errors_sample.xlsx
  Lấy ngẫu nhiên N lỗi (seed cố định) trong các từ khôi phục sai khi bỏ hết dấu rồi khôi phục (test_diac100 so với
  câu gốc của test). Điền sẵn ngữ cảnh, loại lỗi tự động và gợi ý; người chú thích điền 2 cột có danh sách thả xuống.
Phân tích: python -m src.restore_annotation analyze  -> in tỉ lệ từng nhãn (khoảng tin cậy Wilson 95%) và độ chính
  xác mức từ đã hiệu chỉnh (coi "biến thể hợp lý" và "sửa lỗi gốc" là đúng).
"""
import argparse
import math
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .noise import strip_diacritics
from .preprocess import load_resources
from .utils import load_config

LETTER = re.compile(r"[^\W\d_]")
D = Path("dataset/derived/restored_test")
OUT = Path("review/restore_errors_sample.xlsx")
LABELS = ["loi_that", "bien_the_hop_ly", "sua_loi_goc"]
YESNO = ["co", "khong"]


def error_tokens():
    rst = pd.read_csv(D / "test_diac100.csv", dtype={"id": str})
    gold = pd.read_csv(D / "test_test.csv", dtype={"id": str})
    assert (rst["id"] == gold["id"]).all()
    rows, n_alpha = [], 0
    for sid, a, b in zip(gold["id"], rst["text"].fillna(""), gold["text_in"].fillna("")):
        wa, wb = a.split(), b.split()
        if len(wa) != len(wb):
            continue
        for i, (x, y) in enumerate(zip(wa, wb)):
            if not LETTER.search(y):
                continue
            n_alpha += 1
            if x != y:
                rows.append(dict(id=sid, pos=i, tu_goc=y, tu_khoi_phuc=x,
                                 cau_goc=" ".join(wb[:i] + [f"«{y}»"] + wb[i + 1:]),
                                 cau_khoi_phuc=" ".join(wa[:i] + [f"«{x}»"] + wa[i + 1:])))
    return rows, n_alpha


def auto_type(gold, pred):
    if strip_diacritics(gold) == gold:
        return "them_dau_vao_tu_khong_dau"
    if strip_diacritics(gold) != strip_diacritics(pred):
        return "khac_chu_cai"
    if gold.replace("đ", "d") == pred.replace("đ", "d"):
        return "thieu_hoac_thua_đ"
    return "sai_thanh_hoac_dau_mu"


def make(cfg, n, seed):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    rows, n_alpha = error_tokens()
    sample = random.Random(seed).sample(rows, min(n, len(rows)))
    wm = load_resources(cfg).word_map
    freq = Counter((r["tu_goc"], r["tu_khoi_phuc"]) for r in rows)

    wb = Workbook()
    ws = wb.active
    ws.title = "chu_thich"
    cols = ["stt", "id_cau", "tu_goc", "tu_khoi_phuc", "cau_goc", "cau_khoi_phuc", "loai_tu_dong", "goi_y",
            "so_lan_cap_nay", "LOAI (dien)", "DOI_NGHIA_CAM_XUC (dien)", "ghi_chu"]
    ws.append(cols)
    for k, r in enumerate(sample, 1):
        hint = "tu goc khong chuan (co trong tu dien teencode/NSW) -> co the la sua_loi_goc" \
            if r["tu_goc"].lower() in wm else ""
        ws.append([k, r["id"], r["tu_goc"], r["tu_khoi_phuc"], r["cau_goc"], r["cau_khoi_phuc"],
                   auto_type(r["tu_goc"], r["tu_khoi_phuc"]), hint, freq[(r["tu_goc"], r["tu_khoi_phuc"])], "", "", ""])
    last = ws.max_row
    dv1 = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True)
    dv2 = DataValidation(type="list", formula1='"' + ",".join(YESNO) + '"', allow_blank=True)
    ws.add_data_validation(dv1)
    ws.add_data_validation(dv2)
    dv1.add(f"J2:J{last}")
    dv2.add(f"K2:K{last}")
    fill = PatternFill("solid", fgColor="FFF2CC")
    for c in ws[1]:
        c.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2, min_col=10, max_col=11):
        for c in row:
            c.fill = fill
    for col, w in zip("ABCDEFGHIJKL", [5, 8, 12, 12, 60, 60, 24, 30, 8, 18, 22, 25]):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2, min_col=5, max_col=6):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "C2"

    g = wb.create_sheet("huong_dan")
    for line in [
        ["Mỗi dòng là 1 từ bị khôi phục dấu sai (từ trong «»). Câu gốc = câu thật trong VSMEC test; câu khôi phục = bỏ hết dấu rồi khôi phục bằng peterhung/vietnamese-accent-marker-xlm-roberta."],
        [f"Mẫu ngẫu nhiên {len(sample)} / {len(rows)} lỗi (seed {seed}); tổng {n_alpha} từ có chữ cái; độ chính xác thô = {100 * (1 - len(rows) / n_alpha):.2f}%."],
        [""],
        ["Cột LOAI (chọn 1):"],
        ["  loi_that        : khôi phục sai thật, câu đọc sai nghĩa hoặc sai chính tả (vd. tao -> tạo, buồn -> buôn)"],
        ["  bien_the_hop_ly : cả hai cách viết đều chấp nhận được trong ngữ cảnh (vd. hoà/hòa, hai nghĩa đều hợp lý)"],
        ["  sua_loi_goc     : từ GỐC viết sai/không chuẩn, model lại cho ra dạng đúng (vd. viêc -> việc, mói -> mới)"],
        [""],
        ["Cột DOI_NGHIA_CAM_XUC: 'co' nếu lỗi làm đổi cảm xúc người đọc cảm nhận từ câu (vd. buồn -> buôn, vui -> vùi), 'khong' nếu không ảnh hưởng (vd. đó -> do)."],
        ["Chỉ điền 2 cột tô vàng. Cột loai_tu_dong, goi_y là gợi ý tự động, có thể sai."],
    ]:
        g.append(line)
    g.column_dimensions["A"].width = 140
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"[ann] đã ghi {OUT} ({len(sample)} dòng; tổng lỗi {len(rows)}, tổng từ {n_alpha})")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def analyze():
    rows, n_alpha = error_tokens()
    df = pd.read_excel(OUT, sheet_name="chu_thich")
    lab = df["LOAI (dien)"].astype(str).str.strip()
    done = df[lab.isin(LABELS)]
    n = len(done)
    print(f"Đã chú thích {n}/{len(df)} dòng")
    for L in LABELS:
        p, lo, hi = wilson(int((done["LOAI (dien)"] == L).sum()), n)
        print(f"  {L:16s} {p:5.1f}%  [{lo:5.1f}, {hi:5.1f}]")
    real = (done["LOAI (dien)"] == "loi_that").mean()
    raw = 1 - len(rows) / n_alpha
    print(f"Độ chính xác thô: {100 * raw:.2f}% | đã hiệu chỉnh (chỉ tính lỗi thật): {100 * (1 - (1 - raw) * real):.2f}%")
    yn = df["DOI_NGHIA_CAM_XUC (dien)"].astype(str).str.strip()
    m = yn.isin(YESNO)
    p, lo, hi = wilson(int((yn[m] == "co").sum()), int(m.sum()))
    print(f"Lỗi làm đổi nghĩa cảm xúc: {p:.1f}% [{lo:.1f}, {hi:.1f}] (trên {int(m.sum())} dòng)")
    print(df[lab.isin(LABELS)].groupby(["loai_tu_dong", "LOAI (dien)"]).size().unstack(fill_value=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "analyze"])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    if a.cmd == "make":
        make(load_config(a.config), a.n, a.seed)
    else:
        analyze()


if __name__ == "__main__":
    main()
