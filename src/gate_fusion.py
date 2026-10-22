"""Kết hợp model thường (P0) với model không dấu (PS) theo tỉ lệ dấu của từng câu (phương pháp đề xuất, tên tạm
"Diacritic-Aware Gating"; CHƯA kiểm tra tính mới trong tài liệu).

Với câu x:
  r(x) = (số từ có dấu / số từ có chữ cái) của x, sau base_clean
  w(x) = clip(r(x) / r_train, 0, 1), r_train = trung bình r trên tập train VSMEC  (không có tham số cần dò)
  p(x) = w(x) · p_P0(x) + (1 − w(x)) · p_PS(x)
Câu không có chữ cái: w = 1. Câu sạch: w ≈ 1 (dùng model thường); câu mất hết dấu: w = 0 (dùng model không dấu).

Các hệ được so (cùng model, cùng seed, chấm trên cùng các câu):
  normal   : rq1/<m>_P0_none_s<seed>                 (model thường)
  dictnorm : như normal, chấm trên bản chuẩn hóa từ điển (__dn, cách bù 1 của RQ3)
  aug      : rq3_aug/<m>_P0_mix_s<seed>              (augment hỗn hợp nhiễu, cách bù 2 của RQ3)
  restore  : khôi phục dấu (restore_diacritics.py) rồi chấm bằng model thường (pred_<tập>__rst, restore_eval.py)
  strip    : strip/<m>_PS_none_s<seed>               (chỉ model không dấu)
  avg      : 0.5·normal + 0.5·strip                  (ablation: kết hợp KHÔNG theo tỉ lệ dấu)
  ens2     : 0.5·normal(seed i) + 0.5·normal(seed i+1) (baseline: ensemble 2 model thường, cùng chi phí)
  gate     : w·normal + (1−w)·strip                  (phương pháp đề xuất)
  gate_aug : w·aug + (1−w)·strip                     (biến thể: nhánh thường là model augment)
  gate_rst : w·normal + (1−w)·restore                (biến thể: nhánh không dấu là khôi phục dấu, không cần model PS)
Điểm SVM là decision_function nên được đổi sang xác suất bằng softmax trước khi kết hợp.
Kiểm định: paired bootstrap (stats.paired_bootstrap) và bootstrap hai tầng (stats.hier_bootstrap, lấy mẫu lại cả
seed lẫn câu) + hiệu chỉnh Holm trên toàn bộ các phép so.

Chạy: python -m src.gate_fusion --config configs/base.yaml      (hoặc configs/e30.yaml)
Kết quả: <results_dir>/tables/gate_{long,main,weights,tests}.csv/.tex
"""
import argparse
import re

import numpy as np
import pandas as pd

from .data import load_noise_test, load_vsmec, noise_test_sets
from .noise import strip_diacritics
from .preprocess import base_clean
from .stats import hier_bootstrap, holm, paired_bootstrap
from .utils import compute_metrics, labels, load_config, mean_std, p, save_table

LETTER = re.compile(r"[^\W\d_]")
MODELS = ["lr", "svm", "visobert", "phobert", "xlmr", "cafebert"]
METHODS = ["normal", "dictnorm", "aug", "restore", "strip", "avg", "ens2", "gate", "gate_aug", "gate_rst"]
TESTS = [("gate", "normal"), ("gate", "aug"), ("gate", "restore"), ("gate", "ens2"), ("gate", "avg"), ("gate", "dictnorm"),
         ("gate_aug", "aug"), ("gate_rst", "restore"), ("gate_rst", "normal"), ("gate", "gate_rst")]
TEST_SETS = ["test", "diac25", "diac50", "diac75", "diac100"]


def diac_ratio(text):
    toks = [t for t in base_clean(text).split() if LETTER.search(t)]
    if not toks:
        return np.nan
    return sum(strip_diacritics(t) != t for t in toks) / len(toks)


def gate_weights(texts, r_train):
    r = np.array([diac_ratio(t) for t in texts], dtype=float)
    return np.where(np.isnan(r), 1.0, np.clip(r / r_train, 0.0, 1.0))


def _probs(S, model):
    if model == "svm" or not np.allclose(S.sum(1), 1.0, atol=1e-2):
        e = np.exp(S - S.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    return S


def load_scores(cfg, exp, run_id, set_name, model):
    f = p(cfg, "results_dir") / exp / run_id / f"pred_{set_name}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, dtype={"id": str})
    return df["id"].tolist(), df["gold"].tolist(), _probs(df[[f"s_{l}" for l in labels(cfg)]].to_numpy(float), model)


def raw_sets(cfg):
    """Tên tập -> DataFrame có id, text gốc (chưa tiền xử lý)."""
    out = {"test": load_vsmec(cfg, "test")}
    for name in noise_test_sets(cfg):
        out[name] = load_noise_test(cfg, name)
    return out


def system_scores(cfg, model, seed, seeds, set_name):
    """Trả về (ids, gold, dict method -> ma trận xác suất N×K) cho một model, một seed, một tập."""
    rid = lambda prep, noise, s: f"{model}_{prep}_{noise}_s{s}"
    base = load_scores(cfg, "rq1", rid("P0", "none", seed), set_name, model)
    if base is None:
        return None
    ids, gold, normal = base
    sc = {"normal": normal}
    for key, args in [("dictnorm", ("rq1", rid("P0", "none", seed), f"{set_name}__dn")),
                      ("aug", ("rq3_aug", rid("P0", "mix", seed), set_name)),
                      ("restore", ("rq1", rid("P0", "none", seed), f"{set_name}__rst")),
                      ("strip", ("strip", rid("PS", "none", seed), set_name))]:
        r = load_scores(cfg, *args, model)
        if r is not None:
            if r[0] != ids:
                raise ValueError(f"Thứ tự câu khác nhau: {args}")
            sc[key] = r[2]
    i = seeds.index(seed)
    if len(seeds) > 1:
        r = load_scores(cfg, "rq1", rid("P0", "none", seeds[(i + 1) % len(seeds)]), set_name, model)
        if r is not None:
            sc["ens2"] = 0.5 * normal + 0.5 * r[2]
    return ids, gold, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--no-tests", action="store_true", help="bỏ qua bootstrap (nhanh hơn)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    labs = labels(cfg)
    K = len(labs)
    out = p(cfg, "results_dir") / "tables"

    r_train = np.nanmean([diac_ratio(t) for t in load_vsmec(cfg, "train")["text"]])
    print(f"[gate] tỉ lệ từ có dấu trung bình trên train: r_train = {r_train:.4f}")
    sets = raw_sets(cfg)

    long_rows, weight_rows, preds = [], [], {}
    for model in MODELS:
        seeds = [s for s in cfg["seeds"]
                 if (p(cfg, "results_dir") / "rq1" / f"{model}_P0_none_s{s}" / "DONE").exists()]
        for set_name, raw in sets.items():
            w = gate_weights(raw["text"].tolist(), r_train)
            weight_rows.append({"set": set_name, "w_mean": float(w.mean()), "w_eq0": float((w == 0).mean()),
                                "w_eq1": float((w == 1).mean())})
            for seed in seeds:
                r = system_scores(cfg, model, seed, seeds, set_name)
                if r is None:
                    continue
                ids, gold, sc = r
                if ids != raw["id"].astype(str).tolist():
                    raise ValueError(f"Thứ tự câu của {model} s{seed} {set_name} khác dữ liệu gốc")
                if "strip" in sc:
                    sc["avg"] = 0.5 * sc["normal"] + 0.5 * sc["strip"]
                    sc["gate"] = w[:, None] * sc["normal"] + (1 - w[:, None]) * sc["strip"]
                    if "aug" in sc:
                        sc["gate_aug"] = w[:, None] * sc["aug"] + (1 - w[:, None]) * sc["strip"]
                if "restore" in sc:
                    sc["gate_rst"] = w[:, None] * sc["normal"] + (1 - w[:, None]) * sc["restore"]
                g = np.array([labs.index(x) for x in gold])
                for meth, S in sc.items():
                    pr = S.argmax(1)
                    m = compute_metrics(g, pr, K)
                    long_rows.append({"model": model, "method": meth, "set": set_name, "seed": seed,
                                      "acc": m["acc"], "wf1": m["wf1"], "mf1": m["mf1"]})
                    preds.setdefault((model, meth, set_name), {"gold": g, "P": []})["P"].append(pr)

    if not long_rows:
        print("[gate] chưa có dữ liệu (cần rq1 và strip đã chạy xong)")
        return
    df = pd.DataFrame(long_rows)
    save_table(df, out / "gate_long", "Gating theo tỉ lệ dấu: MF1 từng seed.")
    wt = pd.DataFrame(weight_rows).drop_duplicates("set")
    save_table(wt, out / "gate_weights", "Trọng số w của nhánh thường: trung bình và tỉ lệ w=0, w=1 theo tập.")

    noisy = [s for s in sets if s != "test"]
    avg = (df[df["set"].isin(noisy)].groupby(["model", "method", "seed"])["mf1"].mean()
           .rename("mf1").reset_index().assign(set="avg_noisy"))
    full = pd.concat([df[["model", "method", "set", "seed", "mf1"]], avg])
    cols = ["test", "diac25", "diac50", "diac75", "diac100", "teen30", "typo30", "emoji_remove", "avg_noisy"]
    rows = []
    for model in MODELS:
        for meth in METHODS:
            sub = full[(full["model"] == model) & (full["method"] == meth)]
            if sub.empty:
                continue
            row = {"model": model, "method": meth, "n_seeds": sub["seed"].nunique()}
            for c in cols:
                v = sub[sub["set"] == c]["mf1"].tolist()
                row[c] = mean_std(v) if v else ""
            rows.append(row)
    save_table(pd.DataFrame(rows), out / "gate_main", "MF1 (mean ± std qua seed) của các hệ trên test sạch và test nhiễu.")
    print(pd.DataFrame(rows).to_string(index=False))

    if a.no_tests:
        return
    sc = cfg["stats"]
    tests = []
    for model in MODELS:
        for set_name in TEST_SETS + ["teen30", "typo30"]:
            for x, y in TESTS:
                A, B = preds.get((model, x, set_name)), preds.get((model, y, set_name))
                if not A or not B:
                    continue
                PA, PB = np.stack(A["P"]), np.stack(B["P"])
                d, pv, lo, hi = paired_bootstrap(A["gold"], PA, PB, K, sc["n_boot"], sc["seed"])
                _, hpv, hlo, hhi = hier_bootstrap(A["gold"], PA, PB, K, sc["n_boot"], sc["seed"])
                tests.append({"model": model, "set": set_name, "A": x, "B": y, "delta_MF1": d, "p": pv,
                              "ci95_lo": lo, "ci95_hi": hi, "hp": hpv, "hci95_lo": hlo, "hci95_hi": hhi})
    if tests:
        t = pd.DataFrame(tests)
        t["p_holm"] = holm(t["p"].to_numpy())
        t["sig_holm"] = t["p_holm"] < sc["alpha"]
        t["hp_holm"] = holm(t["hp"].to_numpy())
        t["hsig_holm"] = t["hp_holm"] < sc["alpha"]
        save_table(t, out / "gate_tests", "Paired bootstrap (câu) và bootstrap hai tầng (seed + câu) cho MF1, "
                   "hiệu chỉnh Holm trên toàn bộ phép so.")
        print(f"[gate] {len(t)} phép so | có ý nghĩa sau Holm: bootstrap câu {int(t['sig_holm'].sum())}, "
              f"bootstrap hai tầng {int(t['hsig_holm'].sum())}")


if __name__ == "__main__":
    main()
