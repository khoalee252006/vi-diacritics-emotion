"""RQ3: độ bền khi test trên văn bản nhiễu.

Lệnh:
  train   chạy các thí nghiệm augment (rq3_aug: hỗn hợp nhiễu, theo Belinkov & Bisk, 2018, tr.7–8) và
          ma trận train-nhiễu-X / test-nhiễu-Y (rq3_matrix). Bỏ qua run đã xong.
  eval    chấm lại mọi checkpoint/joblib của các run đã xong (rq1, rq3_aug, rq3_matrix) trên các tập nhiễu
          còn thiếu prediction (vd. khi sinh thêm tập nhiễu sau khi train). Bình thường không cần: train_plm
          và baseline_tfidf đã chấm mọi tập nhiễu ngay sau khi train.
  report  tổng hợp bảng: mức sụt MF1 (tuyệt đối, tương đối) so với test sạch, hiệu quả cách bù 1
          (chuẩn hóa từ điển lúc test, hậu tố __dn) và cách bù 2 (augment), và ma trận X/Y.

Cách bù 1 chỉ báo cáo cho model P0 vì P1 đã gồm bước chuẩn hóa này.
"""
import argparse

import numpy as np
import pandas as pd

from .utils import (collect_metrics, enumerate_runs, get_device, is_done, labels, load_config, load_json, p,
                    run_dir, save_json, save_table)

EXPS = ["rq1", "rq3_aug", "rq3_matrix"]


def cmd_train(cfg):
    from .baseline_tfidf import run_group
    from .train_plm import run_one
    for exp in ["rq3_aug", "rq3_matrix"]:
        if exp not in cfg["experiments"]:
            continue
        groups = {}
        for spec in enumerate_runs(cfg, exp, kind="classic"):
            groups.setdefault((spec.model, spec.prep, spec.train_noise), []).append(spec)
        for specs in groups.values():
            run_group(cfg, specs)
        for spec in enumerate_runs(cfg, exp, kind="plm"):
            run_one(cfg, spec)


def cmd_eval(cfg):
    from .data import build_eval_sets, eval_gold
    from .preprocess import prepare
    from .utils import compute_metrics, save_predictions
    labs = labels(cfg)
    for exp in EXPS:
        if exp not in cfg["experiments"]:
            continue
        exp_cfg = cfg["experiments"][exp]
        for spec in enumerate_runs(cfg, exp):
            d = run_dir(cfg, spec)
            if not is_done(d):
                continue
            sets = build_eval_sets(cfg, exp_cfg, spec.prep)
            missing = {k: v for k, v in sets.items() if not (d / f"pred_{k}.csv").exists()}
            if not missing:
                continue
            m = load_json(d / "metrics.json")
            if spec.kind == "classic":
                from .baseline_tfidf import load_model
                model = load_model(d / "model.joblib")
                for name, es in missing.items():
                    sc = model.scores(prepare(es["df"]["text"].tolist(), spec.prep, cfg, dictnorm=es["dictnorm"]))
                    pred = sc.argmax(1)
                    gs, gi = eval_gold(es, cfg)
                    save_predictions(d / f"pred_{name}.csv", es["df"]["id"].tolist(), gs, pred, sc, labs)
                    if gi is not None:
                        m["metrics"][name] = compute_metrics(gi, pred, len(labs))
            else:
                ck = d / "checkpoint"
                if not ck.exists():
                    print(f"[eval] {exp}/{spec.run_id}: thiếu {len(missing)} tập nhưng không có checkpoint, bỏ qua")
                    continue
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                from .train_plm import evaluate_and_save
                device = get_device(cfg)
                tok = AutoTokenizer.from_pretrained(ck)
                model = AutoModelForSequenceClassification.from_pretrained(ck, torch_dtype=torch.float32).to(device)
                mcfg = cfg["models"][spec.model]
                ccfg = {**cfg, "train": {**cfg["train"], **mcfg.get("train", {})}}
                met, _ = evaluate_and_save(model, tok, missing, spec.prep, ccfg, device,
                                           model.config.problem_type == "multi_label_classification",
                                           mcfg.get("segment", False), d)
                m["metrics"].update(met)
                del model
            save_json(m, d / "metrics.json")
            print(f"[eval] {exp}/{spec.run_id}: thêm {len(missing)} tập")


def _paired_drop(df, ref_set):
    """Mức sụt MF1 theo từng seed so với ref_set rồi lấy mean/std."""
    key = ["exp", "model", "prep", "train_noise", "seed"]
    ref = df[df["set"] == ref_set][key + ["mf1"]].rename(columns={"mf1": "mf1_ref"})
    x = df.merge(ref, on=key, how="left")
    x["drop_abs"] = x["mf1"] - x["mf1_ref"]
    x["drop_rel"] = 100 * x["drop_abs"] / x["mf1_ref"]
    return x


def _agg(x, by, cols):
    from .utils import mean_std
    rows = []
    for k, g in x.groupby(by, sort=False, observed=True):
        r = dict(zip(by, k if isinstance(k, tuple) else (k,)))
        r["n_seeds"] = g["seed"].nunique()
        for c in cols:
            r[c] = mean_std(g[c].tolist())
            r[f"{c}_mean"] = g[c].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def cmd_report(cfg):
    out = p(cfg, "results_dir") / "tables"
    df = collect_metrics(cfg, EXPS)
    if df.empty:
        print("[report] chưa có run nào xong")
        return
    df = df[df["set"] != "valid"]
    df["base_set"] = df["set"].str.replace("__dn", "", regex=False)
    df["comp"] = np.where(df["set"].str.endswith("__dn"), "dictnorm", "none")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rq3_long.csv", index=False)

    x = _paired_drop(df, "test")
    order = ["test"] + [s for s in x["base_set"].unique() if s != "test"]
    x["base_set"] = pd.Categorical(x["base_set"], order, ordered=True)

    # Bảng 1: model train sạch (rq1), mọi tập nhiễu, có/không chuẩn hóa từ điển
    t1 = _agg(x[x["exp"] == "rq1"].sort_values(["model", "prep", "base_set", "comp"]),
              ["model", "prep", "base_set", "comp"], ["mf1", "drop_abs", "drop_rel"])
    save_table(t1[["model", "prep", "base_set", "comp", "n_seeds", "mf1", "drop_abs", "drop_rel"]],
               out / "rq3_clean_trained", "RQ3: MF1 và mức sụt so với test sạch (train sạch).")

    # Bảng 2: so ba cách: không bù / bù 1 (dictnorm) / bù 2 (augment hỗn hợp), chỉ P0
    sub = x[(x["prep"] == "P0") & (((x["exp"] == "rq1") & (x["train_noise"] == "none")) |
                                   ((x["exp"] == "rq3_aug") & (x["comp"] == "none")))].copy()
    sub["method"] = np.select([(sub["exp"] == "rq1") & (sub["comp"] == "none"),
                               (sub["exp"] == "rq1") & (sub["comp"] == "dictnorm"),
                               sub["exp"] == "rq3_aug"], ["none", "dictnorm_test", "augment_mix"], "other")
    t2 = _agg(sub.sort_values(["model", "base_set", "method"]), ["model", "base_set", "method"], ["mf1"])
    if not t2.empty:
        t2["base_set"] = pd.Categorical(t2["base_set"], order, ordered=True)
        wide = t2.pivot_table(index=["model", "base_set"], columns="method", values="mf1", aggfunc="first",
                              observed=True).reset_index()
        save_table(wide, out / "rq3_compensation", "RQ3: MF1 (mean ± std) theo cách bù, model P0.")

    # Bảng 3: ma trận train-nhiễu-X / test-nhiễu-Y (MF1 trung bình qua các mức và seed)
    mx = df[(df["prep"] == "P0") & (df["comp"] == "none") &
            (((df["exp"] == "rq1") & (df["train_noise"] == "none")) | df["exp"].isin(["rq3_aug", "rq3_matrix"]))].copy()
    mx_models = set(cfg["experiments"].get("rq3_matrix", {}).get("plm_models", []) or []) | \
        set(cfg["experiments"].get("rq3_matrix", {}).get("classic_models", []) or [])
    mx = mx[mx["model"].isin(mx_models)]
    if not mx.empty:
        mx["test_type"] = mx["base_set"].str.extract(r"^(test|diac|teen|typo|emoji)")[0].replace({"test": "clean"})
        cell = mx.groupby(["model", "train_noise", "test_type", "seed"])["mf1"].mean().reset_index()
        t3 = cell.groupby(["model", "train_noise", "test_type"])["mf1"].mean().unstack("test_type").reset_index()
        cols = [c for c in ["clean", "diac", "teen", "typo", "emoji"] if c in t3.columns]
        t3["avg_noisy"] = t3[[c for c in cols if c != "clean"]].mean(axis=1)
        t3[cols + ["avg_noisy"]] = t3[cols + ["avg_noisy"]].round(2)
        rows_order = ["none", "diac", "teen", "typo", "emoji", "mix"]
        t3["train_noise"] = pd.Categorical(t3["train_noise"], rows_order, ordered=True)
        save_table(t3.sort_values(["model", "train_noise"])[["model", "train_noise"] + cols + ["avg_noisy"]],
                   out / "rq3_matrix", "RQ3: train trên nhiễu X (hàng), test trên nhiễu Y (cột), MF1 trung bình.")

    ns = p(cfg, "noise_dir") / "noise_stats.csv"
    if ns.exists():
        save_table(pd.read_csv(ns), out / "rq3_noise_stats", "Thống kê tập test nhiễu (tỉ lệ thực tế).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "report"])
    ap.add_argument("--config", default="configs/base.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"train": cmd_train, "eval": cmd_eval, "report": cmd_report}[a.cmd](cfg)


if __name__ == "__main__":
    main()
