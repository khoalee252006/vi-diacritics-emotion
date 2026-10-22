"""Kiểm định thống kê và bảng tổng hợp.

Theo Dror et al. (2018): F-score không có phân phối đã biết nên dùng kiểm định phi tham số dựa trên lấy mẫu
(paired bootstrap, tr.4, tr.6); McNemar cho đúng/sai từng câu ghép cặp (tr.5).

Một "hệ" = (exp, model, prep, train_noise, tập đánh giá), gồm prediction của nhiều seed trên cùng các câu.

Paired bootstrap cho Macro-F1:
  thống kê δ = mean_seed MF1(A) − mean_seed MF1(B). Mỗi vòng lấy mẫu có hoàn lại N chỉ số câu, dùng CHUNG
  cho A và B (ghép cặp), tính δ_b. p hai phía = (1 + #{|δ_b − δ| ≥ |δ|}) / (B + 1) (dịch phân phối bootstrap
  về giả thuyết không). Báo cáo thêm khoảng tin cậy 95% percentile của δ.
Bootstrap hai tầng (hboot): như trên nhưng mỗi vòng lấy mẫu lại CẢ seed của A, seed của B (có hoàn lại) lẫn câu,
  nên p và khoảng tin cậy tính cả dao động giữa các seed (fine-tune không ổn định theo seed: Mosbach et al., 2021,
  Bảng 1, tr.9), không chỉ độ bất định của tập test.
McNemar: ghép seed thứ i của A với seed thứ i (vòng lại nếu B ít seed hơn) của B; kiểm định nhị thức chính xác
  trên các câu bất đồng. Báo cáo p từng cặp seed, trung vị p, số cặp có p < alpha.
So sánh bội: hiệu chỉnh Holm trong từng bảng (cột *_holm), vì Dror et al. (2018, tr.1) không xử lý vấn đề này.

Lệnh:
  python -m src.stats --config configs/base.yaml           # bảng RQ1 + kiểm định RQ1 + RQ3 (clean vs bù)
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .utils import collect_metrics, enumerate_runs, is_done, labels, load_config, mean_std, p, run_dir, save_table


# ----------------------------------------------------------------------------- loading
def load_system(cfg, exp, model, prep, train_noise, set_name):
    """Trả về (gold int array N, pred int array S×N, seeds) hoặc None."""
    labs = labels(cfg)
    preds, gold, seeds, ids = [], None, [], None
    for spec in enumerate_runs(cfg, exp):
        if (spec.model, spec.prep, spec.train_noise) != (model, prep, train_noise):
            continue
        f = run_dir(cfg, spec) / f"pred_{set_name}.csv"
        if not (is_done(run_dir(cfg, spec)) and f.exists()):
            continue
        df = pd.read_csv(f, dtype={"id": str})
        if ids is None:
            ids = df["id"].tolist()
            gold = df["gold"].map(labs.index).to_numpy()
        elif df["id"].tolist() != ids:
            raise ValueError(f"Thứ tự câu khác nhau giữa các seed: {f}")
        preds.append(df["pred"].map(labs.index).to_numpy())
        seeds.append(spec.seed)
    if not preds:
        return None
    return gold, np.stack(preds), seeds


# ----------------------------------------------------------------------------- metrics (vectorized)
def macro_f1_batch(gold, pred, K):
    """gold, pred: (B, N) int. Trả về macro-F1 (B,) theo thang 0–100, tính trên đủ K lớp (F1 = 0 khi lớp vắng)."""
    f1s = []
    for c in range(K):
        g, pr = gold == c, pred == c
        tp = (g & pr).sum(1)
        denom = g.sum(1) + pr.sum(1)
        f1s.append(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0))
    return 100 * np.mean(f1s, axis=0)


def paired_bootstrap(gold, A, B, K, n_boot, seed, chunk=500):
    N = len(gold)
    rng = np.random.default_rng(seed)
    d_obs = macro_f1_batch(gold[None], A, K).mean() - macro_f1_batch(gold[None], B, K).mean()
    deltas = []
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, N, size=(b, N))
        g = gold[idx]
        fa = np.mean([macro_f1_batch(g, a[idx], K) for a in A], axis=0)
        fb = np.mean([macro_f1_batch(g, bb[idx], K) for bb in B], axis=0)
        deltas.append(fa - fb)
    deltas = np.concatenate(deltas)
    pval = (1 + np.sum(np.abs(deltas - d_obs) >= abs(d_obs))) / (n_boot + 1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return d_obs, pval, lo, hi


def hier_bootstrap(gold, A, B, K, n_boot, seed, chunk=500):
    """Bootstrap hai tầng: mỗi vòng lấy mẫu lại seed của A, seed của B và N câu (câu dùng chung cho A và B)."""
    N = len(gold)
    rng = np.random.default_rng(seed)
    d_obs = macro_f1_batch(gold[None], A, K).mean() - macro_f1_batch(gold[None], B, K).mean()
    deltas = []
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, N, size=(b, N))
        g = gold[idx]
        fa = np.stack([macro_f1_batch(g, a[idx], K) for a in A], 1)      # (b, S_A)
        fb = np.stack([macro_f1_batch(g, bb[idx], K) for bb in B], 1)    # (b, S_B)
        sa = rng.integers(0, len(A), size=(b, len(A)))
        sb = rng.integers(0, len(B), size=(b, len(B)))
        deltas.append(np.take_along_axis(fa, sa, 1).mean(1) - np.take_along_axis(fb, sb, 1).mean(1))
    deltas = np.concatenate(deltas)
    pval = (1 + np.sum(np.abs(deltas - d_obs) >= abs(d_obs))) / (n_boot + 1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return d_obs, pval, lo, hi


def holm(pvals):
    """Hiệu chỉnh Holm–Bonferroni; trả về p đã hiệu chỉnh, cùng thứ tự đầu vào."""
    pvals = np.asarray(pvals, float)
    adj = np.empty(len(pvals))
    run = 0.0
    for k, i in enumerate(np.argsort(pvals)):
        run = max(run, (len(pvals) - k) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def add_holm(df, alpha):
    for col in ["boot_p", "hboot_p"]:
        df[f"{col}_holm"] = holm(df[col].to_numpy())
        df[f"{col}_holm_sig"] = df[f"{col}_holm"] < alpha
    return df


def mcnemar(gold, A, B, alpha):
    ps = []
    for i, a in enumerate(A):
        b = B[i % len(B)]
        ca, cb = a == gold, b == gold
        n01 = int(np.sum(ca & ~cb))
        n10 = int(np.sum(~ca & cb))
        ps.append(1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue)
    return float(np.median(ps)), int(sum(x < alpha for x in ps)), len(ps), ps


def compare(cfg, sa, sb):
    """sa, sb: tuple (exp, model, prep, train_noise, set)."""
    A, B = load_system(cfg, *sa), load_system(cfg, *sb)
    if A is None or B is None:
        return None
    if len(A[0]) != len(B[0]) or not np.array_equal(A[0], B[0]):
        raise ValueError(f"Gold khác nhau: {sa} vs {sb}")
    K = len(labels(cfg))
    sc = cfg["stats"]
    d, pv, lo, hi = paired_bootstrap(A[0], A[1], B[1], K, sc["n_boot"], sc["seed"])
    _, hpv, hlo, hhi = hier_bootstrap(A[0], A[1], B[1], K, sc["n_boot"], sc["seed"])
    mp, nsig, npair, ps = mcnemar(A[0], A[1], B[1], sc["alpha"])
    return {"A": "/".join(map(str, sa)), "B": "/".join(map(str, sb)), "seeds_A": len(A[2]), "seeds_B": len(B[2]),
            "mean_MF1_A": float(macro_f1_batch(A[0][None], A[1], K).mean()),
            "mean_MF1_B": float(macro_f1_batch(B[0][None], B[1], K).mean()),
            "delta_MF1": d, "boot_p": pv, "boot_ci95_lo": lo, "boot_ci95_hi": hi,
            "hboot_p": hpv, "hboot_ci95_lo": hlo, "hboot_ci95_hi": hhi,
            "mcnemar_median_p": mp, "mcnemar_sig_pairs": f"{nsig}/{npair}",
            "mcnemar_p_each": ";".join(f"{x:.4g}" for x in ps)}


# ----------------------------------------------------------------------------- reports
def rq1_tables(cfg, out):
    df = collect_metrics(cfg, ["rq1"])
    if df.empty:
        print("[stats] chưa có run rq1")
        return
    rows = []
    for (model, prep), g in df[df["set"] == "test"].groupby(["model", "prep"], sort=False):
        be = g["best_epoch"].dropna()
        rows.append({"model": model, "prep": prep, "n_seeds": g["seed"].nunique(),
                     "Acc": mean_std(g["acc"]), "WF1": mean_std(g["wf1"]), "MF1": mean_std(g["mf1"]),
                     "best_epoch": mean_std(be, 1) if len(be) else "-", "MF1_mean": g["mf1"].mean()})
    t = pd.DataFrame(rows).sort_values(["prep", "MF1_mean"], ascending=[True, False])
    save_table(t.drop(columns="MF1_mean"), out / "rq1_main", "RQ1: UIT-VSMEC test, mean ± std qua các seed.")

    cost = []
    for model, g in df[df["set"] == "test"].groupby("model", sort=False):
        cost.append({"model": model, "n_params": int(g["n_params"].iloc[0]),
                     "train_min": mean_std(g["train_time_s"] / 60),
                     "train_min_median": float((g["train_time_s"] / 60).median()),
                     "infer_ms_per_sent": mean_std(g["test_inference_ms_per_sent"], 3),
                     "peak_vram_gb": mean_std(g["peak_vram_gb"])})
    save_table(pd.DataFrame(cost), out / "rq1_cost",
               "Chi phí: số tham số, thời gian train (mean ± std và trung vị; với TF-IDF, run đầu của mỗi nhóm gồm cả dò lưới siêu tham số), suy luận, VRAM tối đa.")

    # Hạng của từng model theo từng seed (độ ổn định thứ hạng)
    rk = df[df["set"] == "test"].copy()
    rk["seed_idx"] = rk.groupby(["model", "prep"])["seed"].rank(method="first").astype(int)
    rk["rank"] = rk.groupby(["prep", "seed_idx"])["mf1"].rank(ascending=False, method="min")
    rt = rk.groupby(["prep", "model"])["rank"].agg(["mean", "min", "max"]).reset_index()
    save_table(rt, out / "rq1_rank_stability", "Hạng theo MF1 giữa các seed (seed thứ i của mỗi model).")


def rq1_tests(cfg, out):
    exp = cfg["experiments"]["rq1"]
    models = list(exp.get("classic_models", []) or []) + list(exp.get("plm_models", []) or [])
    rows = []
    for prep in exp["preps"]:
        for a, b in itertools.combinations(models, 2):
            r = compare(cfg, ("rq1", a, prep, "none", "test"), ("rq1", b, prep, "none", "test"))
            if r:
                rows.append({"comparison": f"model ({prep})", **r})
    if len(exp["preps"]) == 2:
        p0, p1 = exp["preps"]
        for m in models:
            r = compare(cfg, ("rq1", m, p1, "none", "test"), ("rq1", m, p0, "none", "test"))
            if r:
                rows.append({"comparison": "prep", **r})
    if rows:
        save_table(add_holm(pd.DataFrame(rows), cfg["stats"]["alpha"]), out / "rq1_significance",
                   "RQ1: paired bootstrap (câu) + bootstrap hai tầng (seed và câu), hiệu chỉnh Holm; McNemar từng cặp seed.")


def rq3_tests(cfg, out):
    """Trên từng tập nhiễu: model P0 train sạch so với (i) chính nó + chuẩn hóa từ điển, (ii) bản augment."""
    from .data import noise_test_sets
    sets = ["test"] + noise_test_sets(cfg)
    exp = cfg["experiments"]["rq1"]
    models = list(exp.get("classic_models", []) or []) + list(exp.get("plm_models", []) or [])
    rows = []
    for m in models:
        for s in sets:
            base = ("rq1", m, "P0", "none", s)
            r = compare(cfg, ("rq1", m, "P0", "none", f"{s}__dn"), base)
            if r:
                rows.append({"comparison": "dictnorm vs none", "set": s, **r})
            r = compare(cfg, ("rq3_aug", m, "P0", "mix", s), base)
            if r:
                rows.append({"comparison": "augment vs none", "set": s, **r})
    if rows:
        save_table(add_holm(pd.DataFrame(rows).drop(columns="mcnemar_p_each"), cfg["stats"]["alpha"]),
                   out / "rq3_significance", "RQ3: cách bù so với không bù trên từng tập nhiễu (model P0), Holm.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = p(cfg, "results_dir") / "tables"
    out.mkdir(parents=True, exist_ok=True)
    rq1_tables(cfg, out)
    rq1_tests(cfg, out)
    rq3_tests(cfg, out)


if __name__ == "__main__":
    main()
