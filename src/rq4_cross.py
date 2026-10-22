"""RQ4: chuyển giao chéo UIT-VSMEC -> ViGoEmotions (phần mới) và đồng thuận nhãn trên câu trùng.

Ánh xạ 28 nhãn -> 7 nhóm VSMEC theo project_knowledge.md §3 (config `rq4.mapping`):
  variant a (chính): disapproval nằm ngoài phép ánh xạ; câu chỉ có nhãn disapproval bị loại.
  variant b (độ nhạy): disapproval -> Anger. Chỉ đổi nhãn vàng lúc chấm, không train lại model VSMEC.
    Mốc trần (train trên vigo_new_train) được train với `rq4.train_mapping` (mặc định a); điểm của nó theo
    variant b vì thế lệch định nghĩa nhãn giữa train và test, cần ghi chú khi báo cáo.

Chấm điểm:
  lenient (chính): đúng nếu nhãn dự đoán thuộc tập nhóm vàng. Để tính được WF1/MF1, gold hiệu dụng của mỗi câu
    = nhãn dự đoán nếu khớp, ngược lại = nhóm đại diện (nhóm có nhiều nhãn mịn nhất, hòa thì theo thứ tự nhãn).
    Accuracy khi đó đúng bằng lenient accuracy. Đây là quy ước của pipeline, cần nêu rõ trong bài.
  strict (phụ): bỏ các câu có từ 2 nhóm trở lên, chấm đơn nhãn bình thường.

Lệnh:
  train   train mốc trần (rq4_ceiling) trên vigo_new_train.
  report  chấm mọi run có pred_vigo_new_test.csv (rq1 và rq4_ceiling), tính Cohen's κ trên vigo_vsmec_*.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from .data import add_groups, load_vigo
from .utils import (canonical_group, compute_metrics, enumerate_runs, is_done, labels, lenient_targets, load_config,
                    mean_std, p, run_dir, save_table)

VARIANTS = ["a", "b"]


def cmd_train(cfg):
    from .baseline_tfidf import run_group
    from .train_plm import run_one
    exp = "rq4_ceiling"
    groups = {}
    for spec in enumerate_runs(cfg, exp, kind="classic"):
        groups.setdefault((spec.model, spec.prep, spec.train_noise), []).append(spec)
    for specs in groups.values():
        run_group(cfg, specs)
    for spec in enumerate_runs(cfg, exp, kind="plm"):
        run_one(cfg, spec)


def score(pred_labels, gold_df, labs, scoring):
    """pred_labels: list nhãn str, gold_df: DataFrame có groups, votes (cùng thứ tự)."""
    gsets = gold_df["groups"].map(set).tolist()
    keep = [len(g) > 0 for g in gsets]
    if scoring == "strict":
        keep = [k and len(g) == 1 for k, g in zip(keep, gsets)]
    pl = [x for x, k in zip(pred_labels, keep) if k]
    gs = [g for g, k in zip(gsets, keep) if k]
    vt = [v for v, k in zip(gold_df["votes"].tolist(), keep) if k]
    if scoring == "strict":
        eff = [next(iter(g)) for g in gs]
    else:
        eff = lenient_targets(pl, gs, vt, labs)
    m = compute_metrics([labs.index(x) for x in eff], [labs.index(x) for x in pl], len(labs))
    m["n_excluded"] = int(len(keep) - sum(keep))
    return m


def report_transfer(cfg, out):
    labs = labels(cfg)
    test = load_vigo(cfg, "new", "test")
    gold = {v: add_groups(test, cfg, v).set_index("id") for v in VARIANTS}
    rows = []
    for exp in ["rq1", "rq4_ceiling"]:
        if exp not in cfg["experiments"]:
            continue
        for spec in enumerate_runs(cfg, exp):
            f = run_dir(cfg, spec) / "pred_vigo_new_test.csv"
            if not (is_done(run_dir(cfg, spec)) and f.exists()):
                continue
            pr = pd.read_csv(f, dtype={"id": str})
            for v in VARIANTS:
                g = gold[v].loc[pr["id"]]
                for sc in ["lenient", "strict"]:
                    m = score(pr["pred"].tolist(), g, labs, sc)
                    rows.append({"setting": "in-domain ceiling" if exp == "rq4_ceiling" else "VSMEC->ViGo",
                                 **spec.__dict__, "mapping": v, "scoring": sc, **m})
    if not rows:
        print("[rq4] chưa có prediction trên vigo_new_test")
        return
    df = pd.DataFrame(rows)
    df.to_csv(out / "rq4_long.csv", index=False)
    agg = []
    for k, g in df.groupby(["setting", "model", "prep", "mapping", "scoring"], sort=True):
        agg.append({**dict(zip(["setting", "model", "prep", "mapping", "scoring"], k)), "n_seeds": g["seed"].nunique(),
                    "n_eval": int(g["n"].iloc[0]), "n_excluded": int(g["n_excluded"].iloc[0]),
                    "Acc": mean_std(g["acc"]), "WF1": mean_std(g["wf1"]), "MF1": mean_std(g["mf1"])})
    agg = pd.DataFrame(agg)
    save_table(agg[(agg["mapping"] == "a") & (agg["scoring"] == "lenient")], out / "rq4_main",
               "RQ4: VSMEC -> ViGoEmotions (phần mới), ánh xạ a, chấm lenient.")
    save_table(agg, out / "rq4_all", "RQ4: mọi tổ hợp ánh xạ (a/b) và cách chấm (lenient/strict).")


def report_label_stats(cfg, out):
    """Thống kê tập test sau ánh xạ: số câu 0/1/≥2 nhóm."""
    rows = []
    test = load_vigo(cfg, "new", "test")
    for v in VARIANTS:
        g = add_groups(test, cfg, v)["groups"].map(len)
        rows.append({"mapping": v, "n": len(g), "no_group(excluded)": int((g == 0).sum()),
                     "single_group": int((g == 1).sum()), "multi_group": int((g >= 2).sum())})
    save_table(pd.DataFrame(rows), out / "rq4_test_label_stats", "vigo_new_test sau ánh xạ.")


def report_kappa(cfg, out):
    labs = labels(cfg)
    df = pd.concat([load_vigo(cfg, "vsmec", s) for s in ["train", "valid", "test"]], ignore_index=True)
    df = df[df["vsmec_label"].notna()]
    rows = []
    for v in VARIANTS:
        g = add_groups(df, cfg, v)
        g = g[g["groups"].map(len) > 0]
        gsets = g["groups"].map(set).tolist()
        vs = g["vsmec_label"].tolist()
        member = np.mean([a in s for a, s in zip(vs, gsets)])
        single = [len(s) == 1 for s in gsets]
        a_s = [a for a, k in zip(vs, single) if k]
        b_s = [next(iter(s)) for s, k in zip(gsets, single) if k]
        b_len = [a if a in s else canonical_group(s, vt, labs) for a, s, vt in zip(vs, gsets, g["votes"])]
        rows.append({"mapping": v, "n_rows": int(len(df)), "n_used": int(len(g)),
                     "excluded_no_group": int(len(df) - len(g)),
                     "agree_rate_vsmec_in_groups": 100 * member,
                     "kappa_strict(single-group rows)": cohen_kappa_score(a_s, b_s, labels=labs),
                     "n_strict": int(sum(single)),
                     "kappa_lenient": cohen_kappa_score(vs, b_len, labels=labs)})
        cm = confusion_matrix(a_s, b_s, labels=labs)
        pd.DataFrame(cm, index=[f"vsmec_{x}" for x in labs], columns=[f"vigo_{x}" for x in labs]) \
            .to_csv(out / f"rq4_kappa_confusion_strict_{v}.csv")
    save_table(pd.DataFrame(rows), out / "rq4_kappa",
               "Đồng thuận nhãn VSMEC gốc và ViGoEmotions sau ánh xạ trên các câu trùng.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "report"])
    ap.add_argument("--config", default="configs/base.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.cmd == "train":
        cmd_train(cfg)
    else:
        out = p(cfg, "results_dir") / "tables"
        out.mkdir(parents=True, exist_ok=True)
        report_label_stats(cfg, out)
        report_kappa(cfg, out)
        report_transfer(cfg, out)


if __name__ == "__main__":
    main()
