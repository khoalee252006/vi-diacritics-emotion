"""Phân tích E1–E3 theo đúng tiêu chí chốt trước (review/17_thiet_ke_bo_sung_theo_nhan_xet.md).

E1  NI1/NI2 (biên δ = −2, CI 95% và 98.75%), KL1 (dualkl − dual, Holm 12), CAL (ECE thô tại bỏ dấu 100%, Holm 8)
E2  độ nhạy λ (mô tả: valid TB sạch + 25–100%, test sạch/50/100)
E3  Qwen2.5-72B-Instruct-AWQ: vs model thường (sạch, 100%) và vs model không dấu (100%), Holm trong họ
Chạy: python -m src.rev_analyze  -> results/ab/rev_E1_*.csv, rev_E2_lambda.csv, results/llm/rev_E3_72b.csv
"""
import json

import numpy as np
import pandas as pd

from .ab_analyze import P, R
from .ambiguity import MODELS
from .data import load_vsmec
from .stats import holm, macro_f1_batch
from .utils import labels, load_config

N_BOOT = 10000
ENC = ["xlmr", "phobert", "visobert", "cafebert"]
SEEDS_A = {"xlmr": [42, 43, 44, 45, 46], "phobert": [42, 43, 44, 45, 46], "visobert": [42, 43, 44, 45, 46],  # CafeBERT 5 seed từ 2026-10-21
           "cafebert": [42, 43, 44, 45, 46]}
MARGIN = -2.0


def boot(gold, A, B, stat, n_boot=N_BOOT, seed=0):
    """Bootstrap hai tầng tổng quát: câu dùng chung, seed lấy lại riêng cho A và B. A, B: list mảng theo seed.
    stat(arr_seed, idx) -> số. Trả về (quan sát, p hai phía, mảng delta bootstrap)."""
    rng = np.random.default_rng(seed)
    N = len(gold)
    obs = np.mean([stat(a, np.arange(N)) for a in A]) - np.mean([stat(b, np.arange(N)) for b in B])
    d = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.integers(0, N, N)
        sa = rng.integers(0, len(A), len(A))
        sb = rng.integers(0, len(B), len(B))
        d[i] = np.mean([stat(A[j], ix) for j in sa]) - np.mean([stat(B[j], ix) for j in sb])
    p = (1 + np.sum(np.abs(d - obs) >= abs(obs))) / (n_boot + 1)
    return obs, p, d


def mf1_boot(gold, A, B, K, n_boot=N_BOOT, seed=0):
    """Như stats.hier_bootstrap (vector hóa) nhưng trả về cả mảng delta để lấy CI tùy mức."""
    rng = np.random.default_rng(seed)
    N = len(gold)
    obs = macro_f1_batch(gold[None], A, K).mean() - macro_f1_batch(gold[None], B, K).mean()
    out = []
    for start in range(0, n_boot, 500):
        b = min(500, n_boot - start)
        idx = rng.integers(0, N, size=(b, N))
        g = gold[idx]
        fa = np.stack([macro_f1_batch(g, a[idx], K) for a in A], 1)
        fb = np.stack([macro_f1_batch(g, bb[idx], K) for bb in B], 1)
        sa = rng.integers(0, len(A), size=(b, len(A)))
        sb = rng.integers(0, len(B), size=(b, len(B)))
        out.append(np.take_along_axis(fa, sa, 1).mean(1) - np.take_along_axis(fb, sb, 1).mean(1))
    d = np.concatenate(out)
    p = (1 + np.sum(np.abs(d - obs) >= abs(obs))) / (n_boot + 1)
    return obs, p, d


def ece_idx(Pr, y, ix, bins=15):
    p = Pr[ix]
    yy = y[ix]
    conf, pred = p.max(1), p.argmax(1)
    acc = pred == yy
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    for k in range(bins):
        m = b == k
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return 100 * e


def main():
    cfg = load_config("configs/base.yaml")
    labs = labels(cfg)
    K = len(labs)
    t = load_vsmec(cfg, "test")
    ids = t["id"].astype(str).tolist()
    gt = np.array(t["y"])

    def probs(exp, runs, sn, m):
        return [P(cfg, labs, exp, r, sn, m, ids) for r in runs]

    sysruns = lambda m: {
        "standard": ("rq1", [f"{m}_P0_none_s{s}" for s in SEEDS_A[m]]),
        "PS": ("strip", [f"{m}_PS_none_s{s}" for s in SEEDS_A[m]]),
        "dual": ("dual", [f"{m}_P0_dual_s{s}" for s in SEEDS_A[m]]),
        "dualkl": ("dual", [f"{m}_P0_dualkl_s{s}" for s in SEEDS_A[m]]),
    }
    # ------------------------------------------------------------------ mô tả MF1
    desc = []
    for m in ENC:
        for sysn, (exp, runs) in sysruns(m).items():
            rec = {"model": m, "system": sysn, "n_seeds": len(runs)}
            for sn in ["test", "diac25", "diac50", "diac75", "diac100"]:
                Pr = np.stack([p.argmax(1) for p in probs(exp, runs, sn, m)])
                rec[sn] = macro_f1_batch(np.broadcast_to(gt, Pr.shape), Pr, K).mean()
            desc.append(rec)
    D = pd.DataFrame(desc)
    D.round(3).to_csv(R(cfg) / "ab" / "rev_E1_mf1.csv", index=False)
    print(D.round(2).to_string(index=False))

    # ------------------------------------------------------------------ NI + KL1
    ni, kl = [], []
    for m in ENC:
        S = sysruns(m)
        pr = lambda key, sn: np.stack([p.argmax(1) for p in probs(*S[key], sn, m)])
        for crit, sn, other in [("NI1_clean_vs_standard", "test", "standard"), ("NI2_diac100_vs_PS", "diac100", "PS")]:
            obs, p, d = mf1_boot(gt, pr("dualkl", sn), pr(other, sn), K)
            lo95, lo9875 = np.percentile(d, 2.5), np.percentile(d, 0.625)
            ni.append({"crit": crit, "model": m, "delta": obs, "ci95_lo": lo95, "ci95_hi": np.percentile(d, 97.5),
                       "ci9875_lo": lo9875, "p": p, "pass_margin_-2_ci95": lo95 > MARGIN, "pass_margin_-2_ci9875": lo9875 > MARGIN})
        for sn in ["test", "diac75", "diac100"]:
            obs, p, d = mf1_boot(gt, pr("dualkl", sn), pr("dual", sn), K)
            kl.append({"model": m, "set": sn, "delta": obs, "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5), "p": p})
    NI = pd.DataFrame(ni)
    KL = pd.DataFrame(kl)
    KL["p_holm"] = holm(KL["p"].to_numpy())
    NI.round(4).to_csv(R(cfg) / "ab" / "rev_E1_noninferiority.csv", index=False)
    KL.round(4).to_csv(R(cfg) / "ab" / "rev_E1_kl.csv", index=False)
    print("\nNI (biên −2):\n", NI.round(3).to_string(index=False))
    print("\nKL1 (dualkl − dual), Holm 12:\n", KL.round(3).to_string(index=False))
    n_kl = KL[KL.set.isin(["diac75", "diac100"]) & (KL.delta > 0) & (KL.p_holm < 0.05)].model.nunique()
    print(f"-> KL có tác dụng về độ chính xác (≥2/4 model có ý nghĩa dương ở 75% hoặc 100%): {n_kl}/4 "
          f"{'ĐẠT' if n_kl >= 2 else 'KHÔNG ĐẠT'}")

    # ------------------------------------------------------------------ CAL (ECE thô, 100%)
    cal = []
    for m in ENC:
        S = sysruns(m)
        A = probs(*S["dualkl"], "diac100", m)
        for other in ["standard", "dual"]:
            B = probs(*S[other], "diac100", m)
            obs, p, d = boot(gt, A, B, lambda Pr, ix: ece_idx(Pr, gt, ix), n_boot=2000)
            cal.append({"model": m, "vs": other, "ECE_dualkl": np.mean([ece_idx(a, gt, np.arange(len(gt))) for a in A]),
                        "ECE_other": np.mean([ece_idx(b, gt, np.arange(len(gt))) for b in B]), "delta": obs,
                        "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5), "p": p})
    CAL = pd.DataFrame(cal)
    CAL["p_holm"] = holm(CAL["p"].to_numpy())
    CAL.round(4).to_csv(R(cfg) / "ab" / "rev_E1_calibration.csv", index=False)
    print("\nCAL (ECE thô 15 bin, bỏ dấu 100%; âm = dualkl hiệu chỉnh tốt hơn), Holm 8:\n", CAL.round(3).to_string(index=False))
    for other, need in [("standard", 3), ("dual", 2)]:
        n = int(((CAL.vs == other) & (CAL.delta < 0) & (CAL.p_holm < 0.05)).sum())
        print(f"-> dualkl < {other}: {n}/4 (cần ≥{need}) {'ĐẠT' if n >= need else 'KHÔNG ĐẠT'}")

    # ------------------------------------------------------------------ E2 λ (mô tả)
    gv = np.array(load_vsmec(cfg, "valid")["y"])
    lam = []
    for m in ["xlmr", "phobert", "visobert"]:
        for tn in ["dualkl_l0.5", "dualkl", "dualkl_l2"]:
            exp = "dual" if tn == "dualkl" else "dual_lam"
            runs = [f"{m}_P0_{tn}_s{s}" for s in [42, 43, 44]]
            v = []
            for r in runs:
                vv = [macro_f1_batch(gv[None], P(cfg, labs, exp, r, "valid", m).argmax(1)[None], K)[0]]
                vv += [macro_f1_batch(gv[None], P(cfg, labs, exp, r, f"valid_diac{lv}", m).argmax(1)[None], K)[0] for lv in [25, 50, 75, 100]]
                v.append(np.mean(vv))
            rec = {"model": m, "lambda": {"dualkl_l0.5": 0.5, "dualkl": 1.0, "dualkl_l2": 2.0}[tn], "valid_mean": np.mean(v)}
            for sn in ["test", "diac50", "diac100"]:
                Pr = np.stack([P(cfg, labs, exp, r, sn, m, ids).argmax(1) for r in runs])
                rec[sn] = macro_f1_batch(np.broadcast_to(gt, Pr.shape), Pr, K).mean()
            lam.append(rec)
    LAM = pd.DataFrame(lam)
    LAM.round(3).to_csv(R(cfg) / "ab" / "rev_E2_lambda.csv", index=False)
    print("\nE2 λ (seed 42–44, mô tả):\n", LAM.round(2).to_string(index=False))

    # ------------------------------------------------------------------ E3 LLM 72B
    rr = pd.DataFrame([json.loads(x) for x in open(R(cfg) / "llm" / "Qwen_Qwen2.5-72B-Instruct-AWQ" / "responses.jsonl", encoding="utf-8")])
    rr = rr.drop_duplicates(["set", "shots", "pos"])
    llm = lambda sset, sh: np.array([labs.index(x) if x in labs else -1 for x in rr[(rr.set == sset) & (rr.shots == sh)].sort_values("pos")["pred"]])[None]
    tests = []
    for sh in [0, 7]:
        for m in ["cafebert", "xlmr", "phobert", "visobert"]:
            for sset, sn in [("clean", "test"), ("diac100", "diac100")]:
                B = np.stack([p.argmax(1) for p in probs("rq1", [f"{m}_P0_none_s{s}" for s in SEEDS_A[m]], sn, m)])
                obs, p, d = mf1_boot(gt, llm(sset, sh), B, K)
                tests.append({"family": f"vs_standard@{sset}", "shots": sh, "b": m, "delta": obs,
                              "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5), "p": p})
            B = np.stack([p.argmax(1) for p in probs("strip", [f"{m}_PS_none_s{s}" for s in SEEDS_A[m]], "diac100", m)])
            obs, p, d = mf1_boot(gt, llm("diac100", sh), B, K)
            tests.append({"family": "vs_PS@diac100", "shots": sh, "b": f"{m}-PS", "delta": obs,
                          "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5), "p": p})
        for cm in ["lr", "svm"]:
            B = P(cfg, labs, "strip", f"{cm}_PS_none_s42", "diac100", cm, ids).argmax(1)[None]
            obs, p, d = mf1_boot(gt, llm("diac100", sh), B, K)
            tests.append({"family": "vs_PS@diac100", "shots": sh, "b": f"{cm}-PS", "delta": obs,
                          "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5), "p": p})
    E3 = pd.DataFrame(tests)
    E3["p_holm"] = holm(E3["p"].to_numpy())
    E3.round(4).to_csv(R(cfg) / "llm" / "rev_E3_72b.csv", index=False)
    print(f"\nE3 Qwen2.5-72B, Holm {len(E3)}:\n", E3.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
