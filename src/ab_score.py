"""Chấm thêm cho phương pháp A/B (review/12_thiet_ke_phuong_phap_AB.md), chạy trên máy có checkpoint:
  - valid bỏ dấu 25/50/75/100% (results/uncertainty/valid_noisy/valid_diac*.csv) -> chọn biến thể A trên valid;
  - tập mất dấu tự nhiên N, C, C_strip (results/natural/set_*.pkl).
Ghi pred_<tập>.csv vào thư mục từng run của exp `dual` và `strip_ck`.
Chạy: python -m src.ab_score
"""
import pandas as pd

from .utils import enumerate_runs, get_device, load_config, p, run_dir


def eval_sets(cfg):
    sets = {}
    vd = p(cfg, "results_dir") / "uncertainty" / "valid_noisy"
    for lv in [25, 50, 75, 100]:
        df = pd.read_csv(vd / f"valid_diac{lv}.csv", dtype={"id": str}, keep_default_na=False)
        sets[f"valid_diac{lv}"] = {"df": df, "dictnorm": False, "kind": "single"}
    nd = p(cfg, "results_dir") / "natural"
    for k in ["N", "C", "C_strip"]:
        sets[f"nat_{k}"] = {"df": pd.read_pickle(nd / f"set_{k}.pkl").reset_index(drop=True), "dictnorm": False, "kind": "vigo"}
    return sets


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .train_plm import evaluate_and_save
    cfg = load_config("configs/base.yaml")
    sets = eval_sets(cfg)
    device = get_device(cfg)
    for exp in ["dual", "dual_lam", "strip_ck"]:
        for spec in enumerate_runs(cfg, exp, kind="plm"):
            d = run_dir(cfg, spec)
            todo = {k: v for k, v in sets.items() if not (d / f"pred_{k}.csv").exists()}
            if not todo or not (d / "checkpoint").exists():
                continue
            tok = AutoTokenizer.from_pretrained(d / "checkpoint")
            model = AutoModelForSequenceClassification.from_pretrained(d / "checkpoint", dtype=torch.float32).to(device)
            mcfg = cfg["models"][spec.model]
            ccfg = {**cfg, "train": {**cfg["train"], **mcfg.get("train", {})}}
            evaluate_and_save(model, tok, todo, spec.prep, ccfg, device, False, mcfg.get("segment", False), d)
            del model
            torch.cuda.empty_cache()
            print(f"[ab_score] {exp}/{spec.run_id}: {len(todo)} tập")


if __name__ == "__main__":
    main()
