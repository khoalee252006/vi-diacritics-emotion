"""Phân tích CPU bổ sung theo nhận xét của thầy (review/17, E7). Không train, chỉ đọc dự đoán có sẵn.

`restore`: khôi phục dấu CÓ GẮN CỜ: p = w·p_thường + (1 − w)·p_khôi_phục, w từ bộ phát hiện (như detector gate).
  So với thường (sạch) và khôi phục toàn bộ (sạch, bỏ dấu 100%); 4 PLM × 3 = 12, Holm 12.
Chạy: python -m src.rev_cpu restore   -> results/ab/rev_restore.csv
"""
import argparse

import numpy as np
import pandas as pd

from .ab_analyze import P, R
from .ambiguity import MODELS
from .data import load_vsmec
from .natural import detector
from .preprocess import base_clean
from .stats import hier_bootstrap, holm, macro_f1_batch
from .utils import labels, load_config

PLMS = ["cafebert", "xlmr", "phobert", "visobert"]
LEVELS = ["test", "diac25", "diac50", "diac75", "diac100"]


def restore(cfg):
    labs = labels(cfg)
    K = len(labs)
    t = load_vsmec(cfg, "test")
    ids = t["id"].astype(str).tolist()
    gt = np.array(t["y"])
    score = detector(cfg)
    texts = {"test": [base_clean(x) for x in t["text"]]}
    for lv in [25, 50, 75, 100]:
        texts[f"diac{lv}"] = pd.read_csv(R(cfg) / "noise" / f"test_diac{lv}.csv", keep_default_na=False)["text"].tolist()
    W = {k: np.array([1.0 if m_ + a_ < 3 else a_ / (m_ + a_) for m_, a_, _ in (score(x) for x in v)])[None, :, None]
         for k, v in texts.items()}
    mf = lambda Pr: macro_f1_batch(np.broadcast_to(gt, Pr.shape), Pr, K).mean()
    desc, tests = [], []
    for m in PLMS:
        runs = [f"{m}_P0_none_s{s}" for s in MODELS[m]]
        S = {sn: np.stack([P(cfg, labs, "rq1", r, sn, m, ids) for r in runs]) for sn in LEVELS}
        Rr = {sn: np.stack([P(cfg, labs, "rq1", r, f"{sn}__rst", m, ids) for r in runs]) for sn in LEVELS}
        G = {sn: (W[sn] * S[sn] + (1 - W[sn]) * Rr[sn]).argmax(-1) for sn in LEVELS}
        S = {k: v.argmax(-1) for k, v in S.items()}
        Rr = {k: v.argmax(-1) for k, v in Rr.items()}
        for sn in LEVELS:
            desc.append({"model": m, "set": sn, "standard": mf(S[sn]), "restore_all": mf(Rr[sn]), "restore_gated": mf(G[sn]),
                         "share_w_lt_1": float((W[sn] < 1).mean())})
        for name, sn, B in [("gated_vs_standard_clean", "test", S), ("gated_vs_restoreall_clean", "test", Rr),
                            ("gated_vs_restoreall_diac100", "diac100", Rr)]:
            d, pv, lo, hi = hier_bootstrap(gt, G[sn], B[sn], K, 10000, 0)
            tests.append({"test": name, "model": m, "delta": d, "lo": lo, "hi": hi, "p": pv})
    D = pd.DataFrame(desc)
    T = pd.DataFrame(tests)
    T["p_holm"] = holm(T["p"].to_numpy())
    D.round(3).to_csv(R(cfg) / "ab" / "rev_restore_mf1.csv", index=False)
    T.round(4).to_csv(R(cfg) / "ab" / "rev_restore_tests.csv", index=False)
    print(D.round(2).to_string(index=False))
    print(T.round(3).to_string(index=False))


def fewshot(cfg):
    """E4: 7-shot ví dụ KHÔNG dấu vs ví dụ có dấu, test bỏ dấu 100%; bootstrap câu (một lần chạy), Holm 2."""
    import json
    labs = labels(cfg)
    K = len(labs)
    rng = np.random.default_rng(0)

    def load(tag):
        r = pd.DataFrame([json.loads(x) for x in open(R(cfg) / "llm" / tag / "responses.jsonl", encoding="utf-8")])
        r = r[(r.set == "diac100") & (r.shots == 7)].drop_duplicates("pos").sort_values("pos")
        assert len(r) == 693, tag
        return np.array([labs.index(x) if x in labs else -1 for x in r["pred"]]), np.array([labs.index(x) for x in r["gold"]])
    rows = []
    for m in ["gemini-3.5-flash", "gemini-3.5-flash-lite"]:
        a, g = load(f"{m}_stripex")
        b, g2 = load(m)
        assert (g == g2).all()
        f = lambda pr, ix: macro_f1_batch(g[ix][None], pr[ix][None], K)[0]
        d = f(a, np.arange(693)) - f(b, np.arange(693))
        bs = np.array([(lambda ix: f(a, ix) - f(b, ix))(rng.integers(0, 693, 693)) for _ in range(10000)])
        pv = (1 + np.sum(np.abs(bs - d) >= abs(d))) / 10001
        rows.append({"model": m, "mf1_strip_examples": f(a, np.arange(693)), "mf1_accented_examples": f(b, np.arange(693)),
                     "delta": d, "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5), "p": pv})
    T = pd.DataFrame(rows)
    T["p_holm"] = holm(T["p"].to_numpy())
    T.round(4).to_csv(R(cfg) / "llm" / "rev_fewshot_strip.csv", index=False)
    print(T.round(3).to_string(index=False))


def dv_llm(cfg):
    """E7: dual-view (dualkl, 5 seed; CafeBERT 3) vs LLM (một lần chạy) tại sạch và bỏ dấu 100%; Holm trong họ."""
    import json
    from .rev_analyze import SEEDS_A, mf1_boot
    labs = labels(cfg)
    K = len(labs)
    t = load_vsmec(cfg, "test")
    ids = t["id"].astype(str).tolist()
    gt = np.array(t["y"])
    tags = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemma-4-31b-it", "Qwen_Qwen2.5-72B-Instruct-AWQ"]
    rows = []
    for tag in tags:
        r = pd.DataFrame([json.loads(x) for x in open(R(cfg) / "llm" / tag / "responses.jsonl", encoding="utf-8")]).drop_duplicates(["set", "shots", "pos"])
        for sh in [0, 7]:
            for sset, sn in [("clean", "test"), ("diac100", "diac100")]:
                g = r[(r.set == sset) & (r.shots == sh)].sort_values("pos")
                assert len(g) == 693
                L = np.array([labs.index(x) if x in labs else -1 for x in g["pred"]])[None]
                for m in PLMS:
                    A = np.stack([P(cfg, labs, "dual", f"{m}_P0_dualkl_s{s}", sn, m, ids).argmax(1) for s in SEEDS_A[m]])
                    d, pv, bs = mf1_boot(gt, A, L, K)
                    rows.append({"llm": f"{tag} ({sh}-shot)", "set": sset, "dual_view": m, "delta": d,
                                 "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5), "p": pv})
    T = pd.DataFrame(rows)
    T["p_holm"] = holm(T["p"].to_numpy())
    T.round(4).to_csv(R(cfg) / "llm" / "rev_dualview_vs_llm.csv", index=False)
    s = T.assign(sig=T.p_holm < 0.05, win=lambda x: (x.delta > 0) & x.sig, loss=lambda x: (x.delta < 0) & x.sig)
    print(s.groupby(["llm", "set"])[["win", "loss"]].sum().to_string())
    print(T.round(2).to_string(index=False))


def calib_ts(cfg):
    """Nhận xét thầy (2026-10-21): so hiệu chỉnh CÔNG BẰNG — temperature scaling cho CẢ model thường và dual-view,
    T fit trên valid SẠCH từng run, đúng quy trình H1 (src/uncertainty.py: kẹp 1e-6, ece 15 bin).
    Kiểm định: ECE_ts(dual-view) − ECE_ts(thường) tại sạch và bỏ dấu 100%, 4 PLM × 2 = 8, Holm 8.
    Phân tích phát sinh SAU (theo yêu cầu phản biện), không thuộc thiết kế chốt trước."""
    from .rev_analyze import SEEDS_A, boot
    from .uncertainty import EPS, ece, fit_T, temp_scale
    labs = labels(cfg)
    t = load_vsmec(cfg, "test")
    ids = t["id"].astype(str).tolist()
    gt = np.array(t["y"])
    gv = np.array(load_vsmec(cfg, "valid")["y"])

    def load(exp, run, sn, chk=None):
        x = pd.read_csv(R(cfg) / exp / run / f"pred_{sn}.csv", dtype={"id": str})
        if chk is not None:
            assert x["id"].tolist() == chk
        q = np.clip(x[[f"s_{l}" for l in labs]].to_numpy(float), EPS, None)
        return q / q.sum(1, keepdims=True)
    rows, tests = [], []
    for m in PLMS:
        sysruns = {"standard": ("rq1", [f"{m}_P0_none_s{s}" for s in SEEDS_A[m]]),
                   "dualview": ("dual", [f"{m}_P0_dualkl_s{s}" for s in SEEDS_A[m]])}
        Q = {}
        for name, (exp, runs) in sysruns.items():
            Ts = [fit_T(load(exp, r, "valid"), gv) for r in runs]
            for sn in ["test", "diac100"]:
                raw = [load(exp, r, sn, ids) for r in runs]
                ts = [temp_scale(p, T) for p, T in zip(raw, Ts)]
                Q[(name, sn)] = ts
                rows.append({"model": m, "system": name, "set": sn, "T_mean": np.mean(Ts),
                             "ECE_raw": np.mean([ece(p, gt) for p in raw]), "ECE_ts": np.mean([ece(p, gt) for p in ts]),
                             "conf_wrong_ts": np.mean([p.max(1)[p.argmax(1) != gt].mean() for p in ts])})
        for sn in ["test", "diac100"]:
            obs, pv, d = boot(gt, Q[("dualview", sn)], Q[("standard", sn)], lambda Pr, ix: ece(Pr[ix], gt[ix]), n_boot=2000)
            tests.append({"model": m, "set": sn, "delta_ECE_ts": obs, "lo": np.percentile(d, 2.5),
                          "hi": np.percentile(d, 97.5), "p": pv})
    D = pd.DataFrame(rows)
    T = pd.DataFrame(tests)
    T["p_holm"] = holm(T["p"].to_numpy())
    D.round(3).to_csv(R(cfg) / "ab" / "rev_calib_ts.csv", index=False)
    T.round(4).to_csv(R(cfg) / "ab" / "rev_calib_ts_tests.csv", index=False)
    print(D.round(2).to_string(index=False))
    print("\nECE_ts(dual-view) − ECE_ts(thường), Holm 8:\n", T.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["restore", "fewshot", "dvllm", "calibts"])
    a = ap.parse_args()
    cfg = load_config("configs/base.yaml")
    {"restore": restore, "fewshot": fewshot, "dvllm": dv_llm, "calibts": calib_ts}[a.step](cfg)


if __name__ == "__main__":
    main()
