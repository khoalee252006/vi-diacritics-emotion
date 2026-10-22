"""Gating theo tỉ lệ dấu trên dữ liệu nhiễu THẬT: phần mới của ViGoEmotions (vigo_new_test), chấm lenient với ánh xạ a
như RQ4. Model thường = rq1/<m>_P0, model không dấu = strip/<m>_PS (cả hai đều train trên VSMEC, đã chấm sẵn
pred_vigo_new_test.csv). Báo cáo trên toàn tập và trên nhóm câu bị gắn cờ mất dấu (w < 0.5).

Kết quả là KẾT QUẢ ÂM TÍNH: trên dữ liệu thật, tỉ lệ từ có dấu thấp phần lớn do tên riêng, tag, teencode, tiếng Anh,
không phải do mất dấu (xem review/gate_vigo_flagged_labels.csv), nên gating không giúp.

Chạy: python -m src.gate_real --config configs/base.yaml   -> results/tables/gate_vigo.csv, gate_vigo_flagged.csv
"""
import argparse

import numpy as np
import pandas as pd

from .data import add_groups, load_vigo, load_vsmec
from .gate_fusion import _probs, diac_ratio, gate_weights
from .utils import compute_metrics, labels, lenient_targets, load_config, mean_std, p, save_table

SEEDS = {"lr": [42], "svm": [42], "visobert": [42, 43, 44, 45, 46], "phobert": [42, 43, 44, 45, 46],
         "xlmr": [42, 43, 44, 45, 46], "cafebert": [42, 43, 44]}   # TF-IDF tất định: 1 seed là đủ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    labs = labels(cfg)
    R, out = p(cfg, "results_dir"), p(cfg, "results_dir") / "tables"
    r_train = np.nanmean([diac_ratio(t) for t in load_vsmec(cfg, "train")["text"]])
    vt = add_groups(load_vigo(cfg, "new", "test"), cfg, "a")
    keep = (vt["groups"].map(len) > 0).values
    w = gate_weights(vt["text"].tolist(), r_train)
    low = keep & (w < 0.5)

    def scores(exp, run, model):
        df = pd.read_csv(R / exp / run / "pred_vigo_new_test.csv", dtype={"id": str})
        assert df["id"].tolist() == vt["id"].tolist()
        return _probs(df[[f"s_{l}" for l in labs]].to_numpy(float), model)

    def mf1(S, mask):
        pred = S.argmax(1)[mask]
        sub = vt[mask]
        eff = lenient_targets([labs[i] for i in pred], sub["groups"].map(set).tolist(), sub["votes"].tolist(), labs)
        return compute_metrics([labs.index(g) for g in eff], pred, len(labs))["mf1"]

    rows = []
    for m, seeds in SEEDS.items():
        v = {k: [] for k in ["all_std", "all_gate", "low_std", "low_gate"]}
        for s in seeds:
            A, B = scores("rq1", f"{m}_P0_none_s{s}", m), scores("strip", f"{m}_PS_none_s{s}", m)
            G = w[:, None] * A + (1 - w[:, None]) * B
            for k, S, mask in [("all_std", A, keep), ("all_gate", G, keep), ("low_std", A, low), ("low_gate", G, low)]:
                v[k].append(mf1(S, mask))
        rows.append({"model": m, "n_seeds": len(seeds), **{k: mean_std(x) for k, x in v.items()},
                     "d_all": round(np.mean(v["all_gate"]) - np.mean(v["all_std"]), 2),
                     "d_low": round(np.mean(v["low_gate"]) - np.mean(v["low_std"]), 2)})
    t = pd.DataFrame(rows)
    save_table(t, out / "gate_vigo", f"Gating trên vigo_new_test (dữ liệu thật): toàn tập {int(keep.sum())} câu, "
               f"nhóm w<0.5: {int(low.sum())} câu. MF1 lenient (ánh xạ a).")
    vt.loc[low, ["id", "text"]].assign(r=[diac_ratio(x) for x in vt.loc[low, "text"]], w=w[low]) \
        .to_csv(out / "gate_vigo_flagged.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
