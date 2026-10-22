"""Phân tích phương pháp A và B theo đúng tiêu chí chốt trước (review/12_thiet_ke_phuong_phap_AB.md).

Chạy (sau khi tải results/dual, results/strip_ck về): python -m src.ab_analyze  -> results/ab/*.csv
"""
import numpy as np
import pandas as pd

from .ambiguity import MODELS
from .data import add_groups, load_vigo, load_vsmec
from .gate_fusion import _probs, diac_ratio, gate_weights
from .natural import detector
from .preprocess import base_clean
from .stats import hier_bootstrap, holm, macro_f1_batch
from .utils import labels, lenient_targets, load_config, p

N_BOOT = 10000
A_MODELS = ["xlmr", "phobert", "visobert"]
B_MODELS = ["cafebert", "xlmr", "phobert", "visobert"]
A_SEEDS = [42, 43, 44]


def R(cfg):
    return p(cfg, "results_dir")


def read(f, labs, ids=None):
    x = pd.read_csv(f, dtype={"id": str}, keep_default_na=False)
    if ids is not None:
        assert x["id"].tolist() == ids, f
    return x


def probs(x, labs, model):
    return _probs(x[[f"s_{l}" for l in labs]].to_numpy(float), model)


def P(cfg, labs, exp, run, set_name, model, ids=None):
    return probs(read(R(cfg) / exp / run / f"pred_{set_name}.csv", labs, ids), labs, model)


def mf1(gold, preds, K):
    return macro_f1_batch(np.broadcast_to(gold, preds.shape), preds, K).mean()


def out_dir(cfg):
    d = R(cfg) / "ab"
    d.mkdir(exist_ok=True)
    return d


# ----------------------------------------------------------------------------- lenient (ViGo, tập tự nhiên)
def hits(pred_idx, groups, labs):
    return np.array([labs[i] in g for i, g in zip(pred_idx, groups)], float)


def gap_boot(hA, hB, rng):
    """hX = dict(N=(S,nN), C=(S,nC)) chính xác lenient theo seed. Thống kê: (C−N)_A − (C−N)_B (âm = A thu hẹp khoảng cách).
    Bootstrap: câu N, câu C (dùng chung cho A và B) và seed của từng hệ."""
    g = lambda h, iN, iC, sA: h["C"][sA][:, iC].mean() - h["N"][sA][:, iN].mean()
    nN, nC = hA["N"].shape[1], hA["C"].shape[1]
    obs = g(hA, np.arange(nN), np.arange(nC), np.arange(len(hA["N"]))) - g(hB, np.arange(nN), np.arange(nC), np.arange(len(hB["N"])))
    bs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        iN, iC = rng.integers(0, nN, nN), rng.integers(0, nC, nC)
        bs[i] = g(hA, iN, iC, rng.integers(0, len(hA["N"]), len(hA["N"]))) - g(hB, iN, iC, rng.integers(0, len(hB["N"]), len(hB["N"])))
    pv = (1 + np.sum(np.abs(bs - obs) >= abs(obs))) / (N_BOOT + 1)
    return 100 * obs, pv, 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def lenient_boot(PA, PB, groups, votes, labs, rng, n_boot=2000):
    """MF1 lenient: gold hiệu dụng tính riêng cho từng hệ/seed; bootstrap câu + seed. Trả về delta = A − B."""
    K = len(labs)
    eff = lambda Pr: np.stack([[labs.index(e) for e in lenient_targets([labs[i] for i in r], groups, votes, labs)] for r in Pr])
    GA, GB = eff(PA), eff(PB)
    f = lambda G, Pp: macro_f1_batch(G, Pp, K)
    d = f(GA, PA).mean() - f(GB, PB).mean()
    N = PA.shape[1]
    bs = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.integers(0, N, N)
        sa, sb = rng.integers(0, len(PA), len(PA)), rng.integers(0, len(PB), len(PB))
        bs[i] = f(GA[sa][:, ix], PA[sa][:, ix]).mean() - f(GB[sb][:, ix], PB[sb][:, ix]).mean()
    pv = (1 + np.sum(np.abs(bs - d) >= abs(d))) / (n_boot + 1)
    return d, pv, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    import sys
    only_b = "--only-b" in sys.argv   # kiểm tra phần B khi chưa có kết quả A (B4 cần strip_ck nên bị bỏ)
    cfg = load_config("configs/base.yaml")
    labs = labels(cfg)
    K = len(labs)
    rng = np.random.default_rng(0)
    t = load_vsmec(cfg, "test")
    test_ids = t["id"].astype(str).tolist()
    gt = np.array(t["y"])
    gv = np.array(load_vsmec(cfg, "valid")["y"])
    S = {k: pd.read_pickle(R(cfg) / "natural" / f"set_{k}.pkl").reset_index(drop=True) for k in ["N", "C"]}
    groupsN = {k: [set(g) for g in v["groups"]] for k, v in S.items()}

    if only_b:
        A_MODELS_B4 = []
    else:
        A_MODELS_B4 = A_MODELS
    # =========================================================== A: chọn biến thể trên VALID
    sel = []
    if only_b:
        return run_b(cfg, labs, K, rng, t, test_ids, gt, S, groupsN, A_MODELS_B4)
    for var in ["dual", "dualkl"]:
        sc = []
        for m in A_MODELS:
            for s in A_SEEDS:
                run = f"{m}_P0_{var}_s{s}"
                vals = [mf1(gv, P(cfg, labs, "dual", run, "valid", m).argmax(1)[None], K)]
                vals += [mf1(gv, P(cfg, labs, "dual", run, f"valid_diac{lv}", m).argmax(1)[None], K) for lv in [25, 50, 75, 100]]
                sc.append(np.mean(vals))
        sel.append({"variant": var, "valid_mean_mf1": np.mean(sc)})
    sel = pd.DataFrame(sel)
    main_var = sel.sort_values("valid_mean_mf1", ascending=False).iloc[0]["variant"]
    print("Chọn biến thể A trên valid:\n", sel.round(3).to_string(index=False), f"\n-> biến thể chính: {main_var}")

    def preds(exp, runs, set_name, model):
        return np.stack([P(cfg, labs, exp, r, set_name, model, test_ids).argmax(1) for r in runs])

    # =========================================================== bảng mô tả MF1
    rows = []
    for m in A_MODELS:
        sysruns = {"standard": ("rq1", [f"{m}_P0_none_s{s}" for s in MODELS[m]]),
                   "PS": ("strip", [f"{m}_PS_none_s{s}" for s in MODELS[m]]),
                   "aug_mix": ("rq3_aug", [f"{m}_P0_mix_s{s}" for s in MODELS[m] if (R(cfg) / "rq3_aug" / f"{m}_P0_mix_s{s}").exists()]),
                   "A_dual": ("dual", [f"{m}_P0_dual_s{s}" for s in A_SEEDS]),
                   "A_kl": ("dual", [f"{m}_P0_dualkl_s{s}" for s in A_SEEDS])}
        for sysn, (exp, runs) in sysruns.items():
            rec = {"model": m, "system": sysn, "n_runs": len(runs)}
            for sn in ["test", "diac25", "diac50", "diac75", "diac100"]:
                rec[sn] = mf1(gt, preds(exp, runs, sn, m), K)
            rows.append(rec)
    desc = pd.DataFrame(rows)
    print("\nMF1 test (sạch, bỏ dấu 25–100%):\n", desc.round(2).to_string(index=False))

    # =========================================================== A: kiểm định (Holm 15)
    av = "A_dual" if main_var == "dual" else "A_kl"
    tests = []
    for m in A_MODELS:
        A = lambda sn: preds("dual", [f"{m}_P0_{main_var}_s{s}" for s in A_SEEDS], sn, m)
        STD = lambda sn: preds("rq1", [f"{m}_P0_none_s{s}" for s in MODELS[m]], sn, m)
        PSs = lambda sn: preds("strip", [f"{m}_PS_none_s{s}" for s in MODELS[m]], sn, m)
        AUG = lambda sn: preds("rq3_aug", [f"{m}_P0_mix_s{s}" for s in MODELS[m] if (R(cfg) / "rq3_aug" / f"{m}_P0_mix_s{s}").exists()], sn, m)
        for crit, sn, B in [("A1_clean_vs_standard", "test", STD), ("A2_diac100_vs_PS", "diac100", PSs),
                            ("A3_diac75_vs_augmix", "diac75", AUG), ("A3_diac100_vs_augmix", "diac100", AUG)]:
            d, pv, lo, hi = hier_bootstrap(gt, A(sn), B(sn), K, N_BOOT, 0)
            tests.append({"crit": crit, "model": m, "delta": d, "ci_lo": lo, "ci_hi": hi, "p": pv})
        hA = {k: np.stack([hits(P(cfg, labs, "dual", f"{m}_P0_{main_var}_s{s}", f"nat_{k}", m).argmax(1), groupsN[k], labs) for s in A_SEEDS]) for k in "NC"}
        hS = {k: np.stack([hits(probs(read(R(cfg) / "natural" / f"{m}_P0_none_s{s}" / f"pred_{k}.csv", labs), labs, m).argmax(1), groupsN[k], labs)
                           for s in MODELS[m]]) for k in "NC"}
        d, pv, lo, hi = gap_boot(hA, hS, rng)
        tests.append({"crit": "A4_natural_gap_vs_standard", "model": m, "delta": d, "ci_lo": lo, "ci_hi": hi, "p": pv,
                      "accN_A": 100 * hA["N"].mean(), "accC_A": 100 * hA["C"].mean(),
                      "accN_std": 100 * hS["N"].mean(), "accC_std": 100 * hS["C"].mean()})
    ta = pd.DataFrame(tests)
    ta["p_holm"] = holm(ta["p"].to_numpy())
    ta["sig"] = ta["p_holm"] < 0.05
    no_loss = lambda c: not (ta.crit.eq(c) & ta.sig & (ta.delta < 0)).any()
    a1, a2 = no_loss("A1_clean_vs_standard"), no_loss("A2_diac100_vs_PS")
    a3 = all(int((ta.crit.eq(c) & ta.sig & (ta.delta > 0)).sum()) >= 2 for c in ["A3_diac75_vs_augmix", "A3_diac100_vs_augmix"])
    a4 = int((ta.crit.eq("A4_natural_gap_vs_standard") & ta.sig & (ta.delta < 0)).sum())
    print(f"\nA ({main_var}), Holm 15:\n", ta.round(3).to_string(index=False))
    print(f"\nTIÊU CHÍ A: A1 {a1} | A2 {a2} | A3 {a3} | A4 {a4}/3 -> {'THÀNH CÔNG' if a1 and a2 and a3 else 'KHÔNG ĐẠT'}")

    run_b(cfg, labs, K, rng, t, test_ids, gt, S, groupsN, A_MODELS_B4)
    d = out_dir(cfg)
    sel.to_csv(d / "A_variant_selection.csv", index=False)
    desc.round(3).to_csv(d / "A_mf1.csv", index=False)
    ta.round(4).to_csv(d / "A_tests.csv", index=False)


def run_b(cfg, labs, K, rng, t, test_ids, gt, S, groupsN, a_models_b4):
    # =========================================================== B: gating bằng bộ phát hiện
    score = detector(cfg)
    r_train = np.nanmean([diac_ratio(x) for x in load_vsmec(cfg, "train")["text"]])

    def w_det(texts):
        sc = [score(x) for x in texts]
        return np.array([1.0 if m_ + a_ < 3 else a_ / (m_ + a_) for m_, a_, _ in sc])
    texts = {"test": [base_clean(x) for x in t["text"]]}
    for lv in [25, 50, 75, 100]:
        texts[f"diac{lv}"] = pd.read_csv(R(cfg) / "noise" / f"test_diac{lv}.csv", keep_default_na=False)["text"].tolist()
    vt = add_groups(load_vigo(cfg, "new", "test"), cfg, "a")
    keep = (vt["groups"].map(len) > 0).to_numpy()
    texts["vigo"] = vt["text"].astype(str).tolist()
    WB = {k: w_det(v) for k, v in texts.items()}
    W0 = {k: gate_weights(v, r_train) for k, v in texts.items()}
    tb, descB = [], []
    for m in B_MODELS:
        seeds = MODELS[m]
        def sysp(sn):
            fn = "vigo_new_test" if sn == "vigo" else sn
            ids = None if sn == "vigo" else test_ids
            A = np.stack([P(cfg, labs, "rq1", f"{m}_P0_none_s{s}", fn, m, ids) for s in seeds])
            Bp = np.stack([P(cfg, labs, "strip", f"{m}_PS_none_s{s}", fn, m, ids) for s in seeds])
            w0, wb = W0[sn][None, :, None], WB[sn][None, :, None]
            return {"standard": A.argmax(-1), "PS": Bp.argmax(-1), "G0": (w0 * A + (1 - w0) * Bp).argmax(-1),
                    "B": (wb * A + (1 - wb) * Bp).argmax(-1)}
        sysd = {sn: sysp(sn) for sn in texts}
        for sn in ["test", "diac25", "diac50", "diac75", "diac100"]:
            descB.append({"model": m, "set": sn, **{k: mf1(gt, v, K) for k, v in sysd[sn].items()}})
        g_, v_ = vt["groups"][keep].map(set).tolist(), vt["votes"][keep].tolist()
        for crit, sn, other in [("B1_clean_vs_standard", "test", "standard"), ("B2_diac75_vs_G0", "diac75", "G0"),
                                ("B2_diac100_vs_G0", "diac100", "G0")]:
            d, pv, lo, hi = hier_bootstrap(gt, sysd[sn]["B"], sysd[sn][other], K, N_BOOT, 0)
            tb.append({"crit": crit, "model": m, "delta": d, "ci_lo": lo, "ci_hi": hi, "p": pv})
        d, pv, lo, hi = lenient_boot(sysd["vigo"]["B"][:, keep], sysd["vigo"]["standard"][:, keep], g_, v_, labs, rng)
        tb.append({"crit": "B3_vigo_vs_standard", "model": m, "delta": d, "ci_lo": lo, "ci_hi": hi, "p": pv})
        descB.append({"model": m, "set": "vigo_lenient_flagged_by_G0(w<0.5)",
                      **{k: 100 * np.mean([np.mean([labs[i] in gg for i, gg in zip(r[keep & (W0['vigo'] < 0.5)],
                                                                                        [set(x) for x in vt['groups'][keep & (W0['vigo'] < 0.5)]])])
                                            for r in v]) for k, v in sysd["vigo"].items()}})
        if m in a_models_b4:   # B4: tập tự nhiên, PS từ strip_ck (seed 42–44) ghép với P0 cùng seed
            hS, hB = {}, {}
            for k in "NC":
                txt = S[k]["text"].astype(str).tolist()
                wb = w_det(txt)[:, None]
                Ps_ = [probs(read(R(cfg) / "natural" / f"{m}_P0_none_s{s}" / f"pred_{k}.csv", labs), labs, m) for s in A_SEEDS]
                Pf_ = [P(cfg, labs, "strip_ck", f"{m}_PS_none_s{s}", f"nat_{k}", m) for s in A_SEEDS]
                hS[k] = np.stack([hits(a.argmax(1), groupsN[k], labs) for a in Ps_])
                hB[k] = np.stack([hits((wb * a + (1 - wb) * b).argmax(1), groupsN[k], labs) for a, b in zip(Ps_, Pf_)])
            d, pv, lo, hi = gap_boot(hB, hS, rng)
            tb.append({"crit": "B4_natural_gap_vs_standard", "model": m, "delta": d, "ci_lo": lo, "ci_hi": hi, "p": pv,
                       "accN_B": 100 * hB["N"].mean(), "accC_B": 100 * hB["C"].mean(),
                       "accN_std": 100 * hS["N"].mean(), "accC_std": 100 * hS["C"].mean()})
    tb = pd.DataFrame(tb)
    tb["p_holm"] = holm(tb["p"].to_numpy())
    tb["sig"] = tb["p_holm"] < 0.05
    nl = lambda c: not (tb.crit.str.startswith(c) & tb.sig & (tb.delta < 0)).any()
    b1, b2, b3 = nl("B1"), nl("B2"), nl("B3")
    b4 = int((tb.crit.eq("B4_natural_gap_vs_standard") & tb.sig & (tb.delta < 0)).sum())
    print("\nB: MF1 mô tả\n", pd.DataFrame(descB).round(2).to_string(index=False))
    print(f"\nB, Holm {len(tb)}:\n", tb.round(3).to_string(index=False))
    print(f"\nTIÊU CHÍ B: B1 {b1} | B2 {b2} | B3 {b3} | B4 {b4}/3 -> {'THÀNH CÔNG' if b1 and b2 and b3 else 'KHÔNG ĐẠT'}")
    d = out_dir(cfg)
    pd.DataFrame(descB).round(3).to_csv(d / "B_mf1.csv", index=False)
    tb.round(4).to_csv(d / "B_tests.csv", index=False)


if __name__ == "__main__":
    main()
