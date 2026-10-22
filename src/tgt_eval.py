"""D2 (review/09_thiet_ke_dang_la.md): chấm 7,074 biến thể một âm tiết (+ câu sạch tương ứng) bằng checkpoint của
thí nghiệm `tgt`, rồi phân tích P1–P3.

  score   : (chạy trên máy có checkpoint) -> results/tgt/<run>/pred_single.csv, pred_single_clean.csv
  analyze : P1 Δp(nhãn đúng) trên biến thể bỏ dấu từ thuộc W_emo: T-emo < T-rand;
            P2 MF1 bỏ dấu 100%: T-emo > T-rand;  P3 MF1 sạch: T-emo không thua model thường (rq1 P0).
            Bootstrap hai tầng; Holm chung 9 (3 model x 3).
Chạy: python -m src.tgt_eval score | analyze
"""
import argparse

import numpy as np
import pandas as pd

from .data import load_vsmec
from .preprocess import base_clean
from .stats import hier_bootstrap, holm, macro_f1_batch
from .uncertainty import EPS, N_BOOT, iv_dir
from .utils import get_device, labels, load_config, p, save_predictions

MODELS = ["xlmr", "phobert", "visobert"]
SEEDS = [42, 43, 44]
COND = ["tgt_emo", "tgt_rand"]


def run_path(cfg, m, c, s):
    return p(cfg, "results_dir") / "tgt" / f"{m}_P0_{c}_s{s}"


def score(cfg):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .train_plm import evaluate_and_save
    inst = pd.read_csv(iv_dir(cfg) / "single_instances.csv", dtype={"id": str, "sent_id": str}, keep_default_na=False)
    t = load_vsmec(cfg, "test")
    sets = {"single": {"df": inst[["id", "text", "label", "y"]], "dictnorm": False, "kind": "single"},
            "single_clean": {"df": t.assign(id=t["id"].astype(str), text=t["text"].map(base_clean))[["id", "text", "label", "y"]],
                             "dictnorm": False, "kind": "single"}}
    device = get_device(cfg)
    for m in MODELS:
        for c in COND:
            for s in SEEDS:
                d = run_path(cfg, m, c, s)
                todo = {k: v for k, v in sets.items() if not (d / f"pred_{k}.csv").exists()}
                if not todo or not (d / "checkpoint").exists():
                    continue
                tok = AutoTokenizer.from_pretrained(d / "checkpoint")
                model = AutoModelForSequenceClassification.from_pretrained(d / "checkpoint", dtype=torch.float32).to(device)
                mcfg = cfg["models"][m]
                ccfg = {**cfg, "train": {**cfg["train"], **mcfg.get("train", {})}}
                evaluate_and_save(model, tok, todo, "P0", ccfg, device, False, mcfg.get("segment", False), d)
                del model
                torch.cuda.empty_cache()
                print(f"[score] {d.name}: {len(todo)} tập")


def _probs(f, labs, ids=None):
    x = pd.read_csv(f, dtype={"id": str})
    if ids is not None:
        assert x["id"].tolist() == ids, f
    q = np.clip(x[[f"s_{l}" for l in labs]].to_numpy(float), EPS, None)
    return q / q.sum(1, keepdims=True), x


def analyze(cfg):
    labs = labels(cfg)
    K = len(labs)
    from .unfamiliar import load_targets
    We, Wr = set(load_targets(cfg, "emo")), set(load_targets(cfg, "rand"))
    inst = pd.read_csv(iv_dir(cfg) / "single_instances.csv", dtype={"id": str, "sent_id": str}, keep_default_na=False)
    gi = np.array([labs.index(g) for g in inst["label"]])
    ix = np.arange(len(inst))
    sp = inst["sent_pos"].to_numpy()
    in_e, in_r = inst["word"].isin(We).to_numpy(), inst["word"].isin(Wr).to_numpy()
    print(f"biến thể có từ thuộc W_emo: {in_e.sum()}, W_rand: {in_r.sum()}")
    gt = np.array(load_vsmec(cfg, "test")["y"])
    test_ids = load_vsmec(cfg, "test")["id"].astype(str).tolist()
    R = p(cfg, "results_dir")
    rows, tests, rng = [], [], np.random.default_rng(0)
    for m in MODELS:
        DP, PRED = {}, {}
        for c in COND:
            dps = []
            for s in SEEDS:
                d = run_path(cfg, m, c, s)
                Pc, _ = _probs(d / "pred_single_clean.csv", labs)
                Pv, _ = _probs(d / "pred_single.csv", labs, inst["id"].tolist())
                dps.append(100 * (Pc[sp][ix, gi] - Pv[ix, gi]))
            DP[c] = np.stack(dps)
            for sn in ["test", "diac100"]:
                PRED[(c, sn)] = np.stack([_probs(run_path(cfg, m, c, s) / f"pred_{sn}.csv", labs, test_ids)[0].argmax(1) for s in SEEDS])
        PRED[("standard", "test")] = np.stack([_probs(R / "rq1" / f"{m}_P0_none_s{s}" / "pred_test.csv", labs, test_ids)[0].argmax(1)
                                               for s in [42, 43, 44, 45, 46]])
        PRED[("standard", "diac100")] = np.stack([_probs(R / "rq1" / f"{m}_P0_none_s{s}" / "pred_diac100.csv", labs, test_ids)[0].argmax(1)
                                                  for s in [42, 43, 44, 45, 46]])
        mf = lambda A: macro_f1_batch(np.broadcast_to(gt, A.shape), A, K).mean()
        rec = {"model": m}
        for k, v in PRED.items():
            rec[f"MF1_{k[0]}_{k[1]}"] = mf(v)
        for c in COND:
            rec[f"dp_Wemo_{c}"] = DP[c][:, in_e].mean()
            rec[f"dp_Wrand_{c}"] = DP[c][:, in_r].mean()
            rec[f"dp_other_{c}"] = DP[c][:, ~(in_e | in_r)].mean()
        rows.append(rec)
        # P1: Δp trên biến thể W_emo, T-emo − T-rand (âm = T-emo giảm tác hại nhiều hơn); bootstrap câu (cụm) + seed
        sel = np.where(in_e)[0]
        cl = sp[sel]
        groups = {u: sel[cl == u] for u in np.unique(cl)}
        keys = np.array(list(groups))
        A, B = DP["tgt_emo"][:, sel], DP["tgt_rand"][:, sel]
        obs = A.mean() - B.mean()
        bs = []
        for _ in range(N_BOOT // 5):
            kk = rng.choice(keys, len(keys))
            jj = np.concatenate([np.searchsorted(sel, groups[u]) for u in kk])
            sa, sb = rng.integers(0, 3, 3), rng.integers(0, 3, 3)
            bs.append(A[sa][:, jj].mean() - B[sb][:, jj].mean())
        bs = np.array(bs)
        tests.append({"model": m, "test": "P1_dp_Wemo_emo_minus_rand", "delta": obs, "ci_lo": np.percentile(bs, 2.5),
                      "ci_hi": np.percentile(bs, 97.5), "p": (1 + np.sum(np.abs(bs - obs) >= abs(obs))) / (len(bs) + 1)})
        for name, a, b in [("P2_diac100_emo_minus_rand", PRED[("tgt_emo", "diac100")], PRED[("tgt_rand", "diac100")]),
                           ("P3_clean_emo_minus_standard", PRED[("tgt_emo", "test")], PRED[("standard", "test")])]:
            dlt, pv, lo, hi = hier_bootstrap(gt, a, b, K, N_BOOT, 0)
            tests.append({"model": m, "test": name, "delta": dlt, "ci_lo": lo, "ci_hi": hi, "p": pv})
    tt = pd.DataFrame(tests)
    tt["p_holm"] = holm(tt["p"].to_numpy())
    tt["sig"] = tt["p_holm"] < 0.05
    d = p(cfg, "results_dir") / "unfamiliar"
    pd.DataFrame(rows).round(3).to_csv(d / "d2_summary.csv", index=False)
    tt.round(4).to_csv(d / "d2_tests.csv", index=False)
    print(pd.DataFrame(rows).round(2).T.to_string())
    print("\nD2 (Holm 9):\n", tt.round(3).to_string(index=False))
    p1 = int(((tt.test.str.startswith("P1")) & tt.sig & (tt.delta < 0)).sum())
    p2 = int(((tt.test.str.startswith("P2")) & tt.sig & (tt.delta > 0)).sum())
    print(f"\nKết luận nhân quả (P1 và P2 có ý nghĩa ở >= 2/3 model): P1 {p1}/3, P2 {p2}/3 -> "
          f"{'CÓ' if p1 >= 2 and p2 >= 2 else 'KHÔNG'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["score", "analyze"])
    a = ap.parse_args()
    globals()[a.step](load_config("configs/base.yaml"))


if __name__ == "__main__":
    main()
