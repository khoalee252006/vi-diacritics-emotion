"""Baseline "khôi phục dấu rồi mới phân loại", bước 2: chấm các tập đã khôi phục dấu (do restore_diacritics.py tạo)
bằng model RQ1 P0 đã train (checkpoint PLM / model.joblib của TF-IDF). Không train thêm.

Ghi pred_<tên>__rst.csv vào thư mục từng run và thêm metrics "<tên>__rst" vào metrics.json.
Không đụng manifest tập nhiễu nên không ảnh hưởng các bảng RQ3.

Chạy: python -m src.restore_eval --config configs/base.yaml --restored results/noise/restored
"""
import argparse
from pathlib import Path

import pandas as pd

from .data import eval_gold
from .preprocess import prepare
from .utils import (compute_metrics, enumerate_runs, get_device, is_done, labels, load_config, load_json, run_dir,
                    save_json, save_predictions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--restored", default="results/noise/restored")
    ap.add_argument("--exp", default="rq1")
    a = ap.parse_args()
    cfg = load_config(a.config)
    labs = labels(cfg)
    rd = Path(a.restored)
    if not rd.is_absolute():
        rd = Path(cfg["_root_"]) / rd
    sets = {f.stem[len("test_"):] + "__rst": dict(df=pd.read_csv(f, dtype={"id": str}).assign(
                text=lambda d: d["text"].fillna("").astype(str)), dictnorm=False, kind="single")
            for f in sorted(rd.glob("test_*.csv"))}
    if not sets:
        raise SystemExit(f"Không thấy test_*.csv trong {rd}")
    print(f"[rst] {len(sets)} tập đã khôi phục dấu: {', '.join(sets)}")

    for spec in enumerate_runs(cfg, a.exp):
        if spec.prep != "P0":
            continue
        d = run_dir(cfg, spec)
        if not is_done(d):
            continue
        missing = {k: v for k, v in sets.items() if not (d / f"pred_{k}.csv").exists()}
        if not missing:
            continue
        m = load_json(d / "metrics.json")
        if spec.kind == "classic":
            from .baseline_tfidf import load_model
            model = load_model(d / "model.joblib")
            for name, es in missing.items():
                sc = model.scores(prepare(es["df"]["text"].tolist(), "P0", cfg))
                pred = sc.argmax(1)
                gs, gi = eval_gold(es, cfg)
                save_predictions(d / f"pred_{name}.csv", es["df"]["id"].tolist(), gs, pred, sc, labs)
                m["metrics"][name] = compute_metrics(gi, pred, len(labs))
        else:
            ck = d / "checkpoint"
            if not ck.exists():
                print(f"[rst] {spec.run_id}: không có checkpoint, bỏ qua")
                continue
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from .train_plm import evaluate_and_save
            device = get_device(cfg)
            tok = AutoTokenizer.from_pretrained(ck)
            model = AutoModelForSequenceClassification.from_pretrained(ck, torch_dtype=torch.float32).to(device)
            mcfg = cfg["models"][spec.model]
            ccfg = {**cfg, "train": {**cfg["train"], **mcfg.get("train", {})}}
            met, _ = evaluate_and_save(model, tok, missing, "P0", ccfg, device, False, mcfg.get("segment", False), d)
            m["metrics"].update(met)
            del model
        save_json(m, d / "metrics.json")
        r = m["metrics"].get("diac100__rst", {}).get("mf1")
        print(f"[rst] {a.exp}/{spec.run_id}: +{len(missing)} tập" + (f" | diac100__rst MF1={r:.2f}" if r else ""))


if __name__ == "__main__":
    main()
