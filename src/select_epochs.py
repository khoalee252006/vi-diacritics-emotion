"""Chọn 20 hay 30 epoch làm kết quả chính, theo đúng quy tắc chốt ngày 2026-10-10 (trước khi có kết quả 30 epoch):

  - cùng một số epoch cho mọi model;
  - so valid MF1 tại checkpoint được chọn (= max valid MF1 qua các epoch), trung bình trên các run PLM của RQ1
    có mặt ở CẢ HAI bộ (cùng model, prep, seed);
  - chọn bộ B (30 epoch) CHỈ KHI valid trung bình của B cao hơn A ít nhất `--margin` (mặc định 0.5 MF1);
    ngược lại giữ bộ A (20 epoch).
Sửa ngày 2026-10-10 (vẫn trước khi có kết quả 30 epoch): thêm ngưỡng 0.5 vì bộ nhiều epoch hơn có nhiều checkpoint
để chọn max hơn nên valid lợi sẵn (~0.13 MF1 đo ở epoch 16-20 trên 69 run 20 epoch); 0.5 ≈ mức dao động valid giữa
các epoch cuối (std 0.35-0.72).

Script KHÔNG đọc điểm test.

Chạy: python -m src.select_epochs --a configs/base.yaml --b configs/e30.yaml
"""
import argparse
import json

import pandas as pd

from .utils import enumerate_runs, is_done, load_config, p, run_dir, save_table


def valid_scores(cfg, exp):
    out = {}
    for spec in enumerate_runs(cfg, exp, kind="plm"):
        d = run_dir(cfg, spec)
        if is_done(d):
            with open(d / "metrics.json", encoding="utf-8") as f:
                h = json.load(f)["history"]
            out[(spec.model, spec.prep, spec.seed)] = max(x["valid_sel_mf1"] for x in h)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="configs/base.yaml")
    ap.add_argument("--b", default="configs/e30.yaml")
    ap.add_argument("--exp", default="rq1")
    ap.add_argument("--margin", type=float, default=0.5)
    args = ap.parse_args()
    ca, cb = load_config(args.a), load_config(args.b)
    ea, eb = ca["train"]["epochs"], cb["train"]["epochs"]
    A, B = f"valid_A_{ea}ep", f"valid_B_{eb}ep"
    va, vb = valid_scores(ca, args.exp), valid_scores(cb, args.exp)
    keys = sorted(set(va) & set(vb))
    if not keys:
        print("Chưa có run nào xong ở cả hai bộ.")
        return
    df = pd.DataFrame([{"model": k[0], "prep": k[1], "seed": k[2], A: va[k], B: vb[k]}
                       for k in keys])
    by = df.groupby(["model", "prep"])[[A, B]].mean().reset_index()
    by["chenh_lech"] = by[B] - by[A]
    ma, mb = df[A].mean(), df[B].mean()
    choice = eb if mb - ma >= args.margin else ea
    print(by.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nSố run khớp ở cả hai bộ: {len(keys)} (thiếu ở A: {len(set(vb) - set(va))}, thiếu ở B: {len(set(va) - set(vb))})")
    print(f"Valid MF1 trung bình: {ea} epoch = {ma:.3f} | {eb} epoch = {mb:.3f} | chênh = {mb - ma:+.3f}")
    print(f">>> KẾT QUẢ CHÍNH: {choice} epoch (quy tắc: chọn {eb} chỉ khi valid cao hơn >= {args.margin}; ngược lại {ea})")
    by.loc[len(by)] = {"model": "TRUNG BÌNH", "prep": "", A: ma, B: mb, "chenh_lech": mb - ma}
    save_table(by, p(ca, "results_dir") / "tables" / "epoch_selection",
               f"Chọn số epoch theo valid MF1 (RQ1, PLM). Kết quả chính: {choice} epoch.")


if __name__ == "__main__":
    main()
