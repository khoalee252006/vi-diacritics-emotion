"""Bảng và hình cho bản thảo bản B (2026-10-17, sau thí nghiệm bổ sung theo nhận xét của thầy, review/17).

Chỉ đọc kết quả đã có (results/*), không train hay chấm lại model.
Chạy: python -m src.paper_v2  -> manuscript/tables/v2_*.tex, manuscript/figures/v2_*.pdf|png, results/ab/main_fix.csv
"""
import json

import numpy as np
import pandas as pd

from .ab_analyze import P, R, hits, mf1, probs, read
from .ambiguity import MODELS
from .data import add_groups, load_vigo, load_vsmec
from .gate_fusion import gate_weights, diac_ratio
from .natural import detector
from .preprocess import base_clean
from .stats import macro_f1_batch
from .utils import compute_metrics, labels, lenient_targets, load_config

TAB = "manuscript/tables"
FIG = "manuscript/figures"
FIX_MODELS = ["cafebert", "xlmr", "phobert", "visobert"]
SEEDS_A = {"xlmr": [42, 43, 44, 45, 46], "phobert": [42, 43, 44, 45, 46], "visobert": [42, 43, 44, 45, 46],  # CafeBERT 5 seed từ 2026-10-21
           "cafebert": [42, 43, 44, 45, 46]}
CK_SEEDS = [42, 43, 44]          # PS có checkpoint (strip_ck) -> chấm được tập tự nhiên
NAME = {"cafebert": "CafeBERT", "xlmr": "XLM-R", "phobert": "PhoBERT", "visobert": "ViSoBERT",
        "lr": "TF-IDF+LR", "svm": "TF-IDF+SVM"}
LEVELS = ["test", "diac25", "diac50", "diac75", "diac100"]
OURS = "Dual-view consistency (ours)"
DUP = "Duplication (no KL)"


def f1(x, nd=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write(path, s):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(s)
    print("->", path)


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    acc = pred == y
    b = np.clip(np.digitize(conf, np.linspace(0, 1, bins + 1)[1:-1], right=True), 0, bins - 1)
    return 100 * sum((b == k).mean() * abs(acc[b == k].mean() - conf[b == k].mean()) for k in range(bins) if (b == k).any())


def table(caption, label, colspec, header, rows, small=False):
    L = [r"\begin{table}[!htbp]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
         r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{" + colspec + "}", r"\toprule", header + r" \\", r"\midrule"]
    L += rows
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""]
    return "\n".join(L)


# ============================================================================ bảng chính mục 6
def main_fix(cfg, labs, K):
    t = load_vsmec(cfg, "test")
    ids = t["id"].astype(str).tolist()
    gt = np.array(t["y"])
    vt = add_groups(load_vigo(cfg, "new", "test"), cfg, "a")
    keep = (vt["groups"].map(len) > 0).to_numpy()
    gsets, votes = vt["groups"][keep].map(set).tolist(), vt["votes"][keep].tolist()
    single = np.array([len(g) == 1 for g in gsets])                      # strict: chỉ câu thuộc đúng 1 nhóm
    gold_strict = np.array([labs.index(next(iter(g))) for g, s in zip(gsets, single) if s])
    S = {k: pd.read_pickle(R(cfg) / "natural" / f"set_{k}.pkl").reset_index(drop=True) for k in ["N", "C"]}
    gN = {k: [set(g) for g in v["groups"]] for k, v in S.items()}
    score = detector(cfg)
    r_train = np.nanmean([diac_ratio(x) for x in load_vsmec(cfg, "train")["text"]])

    def w_det(texts):
        return np.array([1.0 if m_ + a_ < 3 else a_ / (m_ + a_) for m_, a_, _ in (score(x) for x in texts)])

    texts = {"test": [base_clean(x) for x in t["text"]]}
    for lv in [25, 50, 75, 100]:
        texts[f"diac{lv}"] = pd.read_csv(R(cfg) / "noise" / f"test_diac{lv}.csv", keep_default_na=False)["text"].tolist()
    texts["vigo"] = vt["text"].astype(str).tolist()
    for k in "NC":
        texts[f"nat_{k}"] = S[k]["text"].astype(str).tolist()
    WB = {k: w_det(v)[None, :, None] for k, v in texts.items()}
    W0 = {k: gate_weights(v, r_train)[None, :, None] for k, v in texts.items()}

    def len_mf1(Pr):   # MF1 lenient trên ViGo, gold hiệu dụng theo từng seed
        Pr = Pr[:, keep]
        G = np.stack([[labs.index(e) for e in lenient_targets([labs[i] for i in r], gsets, votes, labs)] for r in Pr])
        return macro_f1_batch(G, Pr, K).mean()

    def strict_mf1(Pr):
        Pr = Pr[:, keep][:, single]
        return macro_f1_batch(np.broadcast_to(gold_strict, Pr.shape), Pr, K).mean()

    def nat(Pr, k):
        return 100 * np.mean([hits(r, gN[k], labs).mean() for r in Pr])

    rows = []
    for m in FIX_MODELS:
        # 2026-10-21: CafeBERT có 5 seed cho thường / không dấu / nhân đôi / dual-view (SEEDS_A); khôi phục dấu (__rst) và
        # dự đoán thường trên tập tự nhiên chỉ có cho seed của src.ambiguity.MODELS (CafeBERT 42–44)
        std_runs = [("rq1", f"{m}_P0_none_s{s}") for s in SEEDS_A[m]]
        std3_runs = [("rq1", f"{m}_P0_none_s{s}") for s in MODELS[m]]
        ps_runs = [("strip", f"{m}_PS_none_s{s}") for s in SEEDS_A[m]]
        aug_runs = [("rq3_aug", f"{m}_P0_mix_s{s}") for s in MODELS[m] if (R(cfg) / "rq3_aug" / f"{m}_P0_mix_s{s}").exists()]
        dup_runs = [("dual", f"{m}_P0_dual_s{s}") for s in SEEDS_A[m]]
        a_runs = [("dual", f"{m}_P0_dualkl_s{s}") for s in SEEDS_A[m]]

        def prob(runs, sn, suffix=""):
            fn = {"vigo": "vigo_new_test"}.get(sn, sn) + suffix
            out = []
            for exp, r in runs:
                if sn.startswith("nat_") and exp == "rq1":
                    if int(r.rsplit("_s", 1)[1]) not in MODELS[m]:
                        continue
                    f = R(cfg) / "natural" / r / f"pred_{sn[4:]}.csv"
                elif sn.startswith("nat_") and exp == "strip":
                    if int(r.rsplit("_s", 1)[1]) not in CK_SEEDS:
                        continue
                    f = R(cfg) / "strip_ck" / r / f"pred_{sn}.csv"
                else:
                    f = R(cfg) / exp / r / f"pred_{fn}.csv"
                if not f.exists():
                    return None
                out.append(probs(read(f, labs, ids if sn in LEVELS else None), labs, m))
            return np.stack(out) if out else None

        def gated(sn, W, other="PS"):
            A_ = prob(std_runs if other == "PS" else std3_runs, sn)
            B_ = prob(ps_runs, sn) if other == "PS" else (prob(std3_runs, sn, "__rst") if sn in LEVELS else None)
            if A_ is None or B_ is None:
                return None
            if sn.startswith("nat_"):
                A_ = A_[[MODELS[m].index(s) for s in CK_SEEDS]]
            return W[sn] * A_ + (1 - W[sn]) * B_

        systems = {
            "Standard": (1, lambda sn: prob(std_runs, sn)),
            "Restore-then-classify": (2, lambda sn: prob(std3_runs, sn, "__rst") if sn in LEVELS else None),
            "Detector-gated restoration": (2, lambda sn: gated(sn, WB, "rst")),
            "Mixed-noise augmentation": (1, lambda sn: prob(aug_runs, sn) if sn in LEVELS else None),
            "Diacritic-free model": (1, lambda sn: prob(ps_runs, sn)),
            "Accent-ratio gate": (2, lambda sn: gated(sn, W0)),
            "Detector gate (ours)": (2, lambda sn: gated(sn, WB)),
            DUP: (1, lambda sn: prob(dup_runs, sn)),
            OURS: (1, lambda sn: prob(a_runs, sn)),
        }
        for name, (nm, fn) in systems.items():
            rec = {"model": m, "system": name, "n_models": nm}
            for sn in LEVELS:
                Pr = fn(sn)
                rec[sn] = None if Pr is None else mf1(gt, Pr.argmax(-1), K)
                if sn == "diac100":
                    rec["ece100"] = None if Pr is None else float(np.mean([ece(p, gt) for p in Pr]))
                if sn == "test":
                    rec["ece_clean"] = None if Pr is None else float(np.mean([ece(p, gt) for p in Pr]))
            Pv = fn("vigo")
            rec["vigo"] = None if Pv is None else len_mf1(Pv.argmax(-1))
            rec["vigo_strict"] = None if Pv is None else strict_mf1(Pv.argmax(-1))
            PN, PC = fn("nat_N"), fn("nat_C")
            rec["nat_N"] = None if PN is None else nat(PN.argmax(-1), "N")
            rec["nat_C"] = None if PC is None else nat(PC.argmax(-1), "C")
            rows.append(rec)
    # TF-IDF (nhận xét thầy 2026-10-21): thường / không dấu / nhân đôi; tất định -> 1 run; ECE chỉ cho LR (SVM không ra xác suất)
    from .baseline_tfidf import load_model
    from .preprocess import prepare
    for m in ["lr", "svm"]:
        for name, exp, prep, tn in [("Standard", "rq1", "P0", "none"), ("Diacritic-free model", "strip", "PS", "none"),
                                    (DUP, "dual_tfidf", "P0", "dual")]:
            run = f"{m}_{prep}_{tn}_s42"

            def pr(sn):
                f = R(cfg) / exp / run / f"pred_{ {'vigo': 'vigo_new_test'}.get(sn, sn) }.csv"
                return probs(read(f, labs, ids if sn in LEVELS else None), labs, m)[None]
            rec = {"model": m, "system": name, "n_models": 1}
            for sn in LEVELS:
                Pr = pr(sn)
                rec[sn] = mf1(gt, Pr.argmax(-1), K)
                if sn == "diac100":
                    rec["ece100"] = ece(Pr[0], gt) if m == "lr" else None
                if sn == "test":
                    rec["ece_clean"] = ece(Pr[0], gt) if m == "lr" else None
            Pv = pr("vigo")
            rec["vigo"], rec["vigo_strict"] = len_mf1(Pv.argmax(-1)), strict_mf1(Pv.argmax(-1))
            model = load_model(R(cfg) / exp / run / "model.joblib")
            for k in "NC":
                rec[f"nat_{k}"] = 100 * hits(model.scores(prepare(texts[f"nat_{k}"], prep, cfg)).argmax(1), gN[k], labs).mean()
            rows.append(rec)
    df = pd.DataFrame(rows)
    df["nat_gap"] = df["nat_C"] - df["nat_N"]
    df.round(3).to_csv(R(cfg) / "ab" / "main_fix.csv", index=False)
    print(df.round(2).to_string(index=False))

    out = []
    ALL = FIX_MODELS + ["lr", "svm"]
    for i, m in enumerate(ALL):
        for j, (_, r) in enumerate(df[df.model == m].iterrows()):
            name = r["system"] if r["system"] != OURS else r"\textbf{" + OURS + "}"
            out.append(" & ".join([NAME[m] if j == 0 else "", name, str(r["n_models"]), f1(r["test"]), f1(r["diac50"]),
                                   f1(r["diac100"]), f1(r["ece100"]), f1(r["vigo"]), f1(r["nat_N"])]) + r" \\")
        if i < len(ALL) - 1:
            out.append(r"\midrule")
    cap = (r"Remedies for missing diacritics. Clean, 50\%, 100\%: Macro-F1 on UIT-VSMEC test with diacritics removed from "
           r"that share of accented words; ECE: expected calibration error (15 bins, raw probabilities) at 100\% removal; "
           r"ViGo: lenient Macro-F1 on the new part of ViGoEmotions; Nat.: lenient accuracy on the 94 naturally undiacritised "
           r"ViGoEmotions sentences. \#M: models at inference. All systems use preprocessing P0 (the diacritic-free models "
           r"strip all diacritics instead). Means over 5 seeds (CafeBERT restoration and natural-text standard rows: 3; TF-IDF is deterministic, one run; gates on "
           r"natural text use the 3 seeds with a saved diacritic-free checkpoint, none for CafeBERT). Duplication = training "
           r"on each sentence and its fully undiacritised copy without the consistency term. ECE is not reported for the "
           r"SVM, which does not output probabilities.")
    write(f"{TAB}/v2_mainfix.tex", table(cap, "tab:mainfix", "llcrrrrrr",
                                        r"Model & System & \#M & Clean & 50\% & 100\% & ECE & ViGo & Nat.", out))
    # phụ lục: ViGo strict
    out = []
    for m in FIX_MODELS:
        for j, (_, r) in enumerate(df[(df.model == m) & df.vigo.notna()].iterrows()):
            out.append(f"{NAME[m] if j == 0 else ''} & {r['system']} & {f1(r['vigo'])} & {f1(r['vigo_strict'])} \\\\")
        out.append(r"\midrule") if m != FIX_MODELS[-1] else None
    write(f"{TAB}/v2_vigo_strict.tex", table(
        r"ViGoEmotions (new part, test) under lenient scoring (correct if the prediction matches any gold group) and strict "
        r"scoring (sentences whose labels map to exactly one group; " + f"{int(single.sum())}" + r" of " + f"{int(keep.sum())}" + r" sentences).",
        "tab:vigostrict", "llrr", r"Model & System & Lenient & Strict", out))
    return df


# ============================================================================ hình: MF1 theo mức bỏ dấu
def fig_fix(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [0, 25, 50, 75, 100]
    show = [("Standard", "#7f7f7f", "o", "-"), ("Diacritic-free model", "#1f77b4", "s", "--"),
            ("Mixed-noise augmentation", "#ff7f0e", "^", "-"), ("Detector gate (ours)", "#2ca02c", "D", "-"),
            (DUP, "#9467bd", "v", ":"), (OURS, "#d62728", "*", "-")]
    plt.rcParams.update({"font.size": 12})
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
    for ax, m in zip(axs, FIX_MODELS):
        for name, c, mk, ls in show:
            r = df[(df.model == m) & (df.system == name)].iloc[0]
            ax.plot(x, [r[s] for s in LEVELS], color=c, marker=mk, ls=ls, lw=2, ms=6, label=name)
        ax.set_title(NAME[m], fontsize=13)
        ax.set_xticks(x)
        ax.tick_params(labelsize=11)
        ax.grid(alpha=.3)
    axs[0].set_ylabel("Macro-F1", fontsize=12)
    fig.supxlabel("% of accented words stripped", fontsize=12, y=0.13)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ["pdf", "png"]:
        fig.savefig(f"{FIG}/v2_fix_curves.{ext}", dpi=200, bbox_inches="tight")
    print("->", f"{FIG}/v2_fix_curves.pdf")


# ============================================================================ hiệu chỉnh công bằng (nhận xét thầy 2026-10-21)
def tab_calib_dv(cfg):
    d = pd.read_csv(R(cfg) / "ab" / "rev_calib_ts.csv")
    t = pd.read_csv(R(cfg) / "ab" / "rev_calib_ts_tests.csv").set_index(["model", "set"])
    out = []
    for m in FIX_MODELS:
        g = lambda sysn, sn, col: d[(d.model == m) & (d.system == sysn) & (d.set == sn)][col].iloc[0]
        cells = [NAME[m]]
        for sn in ["test", "diac100"]:
            r = t.loc[(m, sn)]
            star = "$^{*}$" if r.p_holm < 0.05 else ""
            cells += [f"{g('standard', sn, 'ECE_raw'):.1f}", f"{g('dualview', sn, 'ECE_raw'):.1f}",
                      f"{g('standard', sn, 'ECE_ts'):.1f}", f"{g('dualview', sn, 'ECE_ts'):.1f}{star}"]
        out.append(" & ".join(cells) + r" \\")
    cap = (r"Calibration with and without temperature scaling (ECE, 15 bins). Std: standard model; DV: dual-view "
           r"consistency training. The temperature is fitted per run on clean validation data for both systems. "
           r"$^{*}$: DV lower than Std after scaling (two-level bootstrap, Holm over 8, $p_{\text{Holm}}\le0.02$); no "
           r"clean-text difference is significant. This comparison was added after review and was not pre-specified.")
    write(f"{TAB}/v2_calib_dv.tex", table(
        cap, "tab:calibdv", "lrrrrrrrr",
        r"& \multicolumn{4}{c}{Clean} & \multicolumn{4}{c}{100\% stripped} \\ "
        r"& \multicolumn{2}{c}{Raw} & \multicolumn{2}{c}{Scaled} & \multicolumn{2}{c}{Raw} & \multicolumn{2}{c}{Scaled} \\ "
        r"Model & Std & DV & Std & DV & Std & DV & Std & DV", out))


# ============================================================================ ablation: KL và λ
def tab_ablation(cfg, df):
    kl = pd.read_csv(R(cfg) / "ab" / "rev_E1_kl.csv")
    lam = pd.read_csv(R(cfg) / "ab" / "rev_E2_lambda.csv")
    out = []
    for m in FIX_MODELS:
        a = df[(df.model == m) & (df.system == OURS)].iloc[0]
        d = df[(df.model == m) & (df.system == DUP)].iloc[0]
        k = kl[(kl.model == m) & (kl.set == "diac100")].iloc[0]
        lam_str = "--"
        if m in set(lam.model):
            l = lam[lam.model == m].set_index("lambda")["diac100"]
            lam_str = f"{l[0.5]:.1f} / {l[1.0]:.1f} / {l[2.0]:.1f}"
        out.append(f"{NAME[m]} & {d['test']:.1f} & {a['test']:.1f} & {d['diac100']:.1f} & {a['diac100']:.1f} & "
                   f"${k['delta']:+.1f}$ [${k['lo']:.1f}$, ${k['hi']:.1f}$] & {d['ece100']:.1f} & {a['ece100']:.1f} & {lam_str} \\\\")
    cap = (r"Ablation of the consistency term. Dup.: duplication without KL; DV: dual-view consistency ($\lambda=1$). "
           r"$\Delta$: DV minus Dup.\ at 100\% removal (two-level bootstrap, unadjusted 95\% CI; after Holm correction over "
           r"12 comparisons only PhoBERT is significant). ECE at 100\% removal (DV lower than Dup.\ significantly for XLM-R "
           r"and PhoBERT, Holm over 8). "
           r"Last column: Macro-F1 at 100\% removal for $\lambda$ = 0.5 / 1 / 2 (seeds 42--44).")
    write(f"{TAB}/v2_ablation.tex", table(cap, "tab:ablation", "lrrrrlrrl",
                                         r"Model & \multicolumn{2}{c}{Clean} & \multicolumn{2}{c}{100\%} & $\Delta$ 100\% & "
                                         r"\multicolumn{2}{c}{ECE 100\%} & $\lambda$ sweep \\ & Dup. & DV & Dup. & DV & & Dup. & DV &", out))


# ============================================================================ bảng can thiệp (mục 4) + E5
def tab_intervention(cfg):
    d = pd.read_csv(R(cfg) / "ambiguity" / "intervention" / "multi_results.csv").set_index("model")
    e5 = pd.read_csv(R(cfg) / "ambiguity" / "intervention_rev" / "E5_results.csv")
    a5 = e5[e5.test == "clear_vs_amb"].set_index("model")
    r5 = e5[e5.test == "clear_vs_rand"].set_index("model")
    out = []
    for m in ["cafebert", "xlmr", "phobert", "visobert", "lr", "svm"]:
        r = d.loc[m]
        out.append(f"{NAME[m]} & {r['amb_emo']:.1f} & {r['rand']:.1f} & ${r['d_vs_rand']:.1f}$ & "
                   f"{a5.loc[m, 'amb']:.1f} & {a5.loc[m, 'clear']:.1f} & ${a5.loc[m, 'delta']:.1f}$ [${a5.loc[m, 'lo']:.1f}$, ${a5.loc[m, 'hi']:.1f}$] & "
                   f"${r5.loc[m, 'delta']:+.1f}$ \\\\")
    cap = (r"Targeted stripping (Macro-F1). Left: on the 416 test sentences with an ambiguous class-indicative word, "
           r"stripping these words (Amb.) versus the same number of random syllables (Rand.; 5 draws). Middle: on the 434 "
           r"sentences that contain both kinds, stripping $k$ ambiguous (Amb.) versus $k$ unambiguous (Unamb.) "
           r"class-indicative words. Right: stripping all unambiguous class-indicative words minus a matched random "
           r"control (513 sentences). Two-level bootstrap; left and middle differences are all significant after Holm "
           r"correction ($p_{\text{Holm}}\le0.015$), right ones are not.")
    write(f"{TAB}/v2_intervention.tex", table(
        cap, "tab:intervention", "lrrrrrlr",
        r"& \multicolumn{3}{c}{Ambiguous vs.\ random} & \multicolumn{3}{c}{Ambiguous vs.\ unambiguous} & Unamb. \\ "
        r"Model & Amb. & Rand. & $\Delta$ & Amb. & Unamb. & $\Delta$ [95\% CI] & vs.\ rand.", out))
    # phụ lục E6
    e6 = pd.read_csv(R(cfg) / "ambiguity" / "intervention_rev" / "E6_results.csv").set_index("model")
    L = pd.read_csv(R(cfg) / "ambiguity" / "intervention_rev" / "E6_word_lists.csv")
    out = []
    for m in ["cafebert", "xlmr", "phobert", "visobert"]:
        r = e6.loc[m]
        lw = L[L.model == m]
        out.append(f"{NAME[m]} & {lw.word.nunique()} & {100 * (lw.confus >= 0.2).mean():.0f}\\% & {int(r.n_sent)} & "
                   f"{r.amb_emo:.1f} & {r.rand:.1f} & ${r.d_vs_rand:.1f}$ [${r.lo_vs_rand:.1f}$, ${r.hi_vs_rand:.1f}$] & ${r.d_vs_ambother:.1f}$ \\\\")
    write(f"{TAB}/v2_e6.tex", table(
        r"Intervention with model-specific word lists: for each encoder, class-indicative words are the top 50 per emotion "
        r"class by occlusion importance on the training set (seed-42 checkpoint), instead of TF-IDF+LR coefficients. "
        r"$\Delta$: Macro-F1 when stripping the ambiguous words of the list minus random (or other ambiguous) syllables; "
        r"all 8 differences significant after Holm correction ($p_{\text{Holm}}\le0.014$).",
        "tab:e6", "lrrrrrlr", r"Model & Words & Amb. & Sent. & Amb. & Rand. & $\Delta$ vs.\ rand. & $\Delta$ vs.\ amb.\ other", out))


# ============================================================================ bảng hiệu chỉnh (mục 5)
def tab_calibration(cfg):
    c = pd.read_csv(R(cfg) / "uncertainty" / "h1_calibration.csv")
    s = pd.read_csv(R(cfg) / "uncertainty" / "h2_selective.csv")
    t = pd.read_csv(R(cfg) / "uncertainty" / "h1_tests.csv").set_index(["model", "version"])
    out = []
    for m in ["cafebert", "xlmr", "phobert", "visobert"]:
        g = lambda st, v, col: c[(c.model == m) & (c.set == st) & (c.version == v)][col].iloc[0]
        a = lambda st: s[(s.model == m) & (s.set == st) & (s.signal == "MSP")]["AUROC"].iloc[0]
        out.append(" & ".join([NAME[m],
                               f"{g('test', 'raw', 'acc'):.1f} $\\rightarrow$ {g('diac100', 'raw', 'acc'):.1f}",
                               f"{g('test', 'raw', 'conf_on_wrong'):.2f} $\\rightarrow$ {g('diac100', 'raw', 'conf_on_wrong'):.2f}",
                               f"{g('test', 'raw', 'ECE'):.1f} $\\rightarrow$ {g('diac100', 'raw', 'ECE'):.1f}",
                               f"{g('test', 'ts', 'ECE'):.1f} $\\rightarrow$ {g('diac100', 'ts', 'ECE'):.1f} ($+{t.loc[(m, 'ts'), 'dECE']:.1f}$)",
                               f"{a('test'):.2f} $\\rightarrow$ {a('diac100'):.2f}"]) + r" \\")
    cap = (r"Standard models become confidently wrong (clean $\rightarrow$ 100\% stripped). Acc.: accuracy; Conf.\ wrong: "
           r"mean confidence on misclassified sentences; ECE (15 bins) before and after temperature scaling fitted on clean "
           r"validation data, with the increase in parentheses (significant for all four models, two-level bootstrap, "
           r"Holm over 8, $p_{\text{Holm}}\le0.001$); AUROC of maximum softmax probability for separating errors.")
    write(f"{TAB}/v2_calib.tex", table(cap, "tab:calib", "lccccc",
                                      r"Model & Acc. & Conf.\ wrong & ECE (raw) & ECE (scaled) & AUROC", out))


# ============================================================================ bảng LLM (mục 7)
def tab_llm(cfg, fix):
    labs = labels(cfg)
    K = len(labs)
    d = pd.read_csv(R(cfg) / "llm" / "compare_mf1.csv")
    q = pd.DataFrame([json.loads(x) for x in open(R(cfg) / "llm" / "Qwen_Qwen2.5-72B-Instruct-AWQ" / "responses.jsonl", encoding="utf-8")])
    q = q.drop_duplicates(["set", "shots", "pos"])

    def qmf(sset, sh):
        g = q[(q.set == sset) & (q.shots == sh)].sort_values("pos")
        return compute_metrics([labs.index(x) for x in g.gold], [labs.index(x) if x in labs else -1 for x in g.pred], K)["mf1"]
    llm_name = {"gemini-3.5-flash": "Gemini 3.5 Flash", "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
                "gemma-4-31b-it": "Gemma 4 31B", "QWEN72": "Qwen2.5-72B-Instruct (AWQ)",
                "Qwen_Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct", "Qwen_Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct",
                "arcee-ai_Arcee-VyLinh": "Arcee-VyLinh (3B)"}
    out = []
    for key, nm in llm_name.items():
        for sh in [0, 7]:
            if key == "QWEN72":
                c, h, f = qmf("clean", sh), qmf("diac50", sh), qmf("diac100", sh)
            else:
                r = d[d.system == f"{key} ({sh}-shot)"].iloc[0]
                c, h, f = r.clean, r.diac50, r.diac100
            out.append(f"{nm if sh == 0 else ''} & {sh} & {c:.1f} & {h:.1f} & {f:.1f} & {c - f:.1f} \\\\")
        if key in ("gemma-4-31b-it", "QWEN72"):
            out.append(r"\midrule")
    out.append(r"\midrule")
    for m in FIX_MODELS:
        r = fix[(fix.model == m) & (fix.system == "Standard")].iloc[0]
        out.append(f"{NAME[m]} (standard) & -- & {r.test:.1f} & {r.diac50:.1f} & {r.diac100:.1f} & {r.test - r.diac100:.1f} \\\\")
    for m in FIX_MODELS:
        r = fix[(fix.model == m) & (fix.system == OURS)].iloc[0]
        out.append(f"{NAME[m]} + dual-view (ours) & -- & {r.test:.1f} & {r.diac50:.1f} & {r.diac100:.1f} & {r.test - r.diac100:.1f} \\\\")
    cap = (r"LLMs versus fine-tuned models on UIT-VSMEC test (Macro-F1; LLMs: one run, temperature 0, few-shot examples "
           r"with diacritics). Drop: clean minus 100\% stripped. Fine-tuned rows are means over seeds.")
    write(f"{TAB}/v2_llm.tex", table(cap, "tab:llm", "llrrrr", r"System & Shots & Clean & 50\% & 100\% & Drop", out))


# ============================================================================ bảng mất dấu tự nhiên (mục 8)
def tab_natural(cfg, fix):
    d = pd.read_csv(R(cfg) / "natural" / "rq4_natural_acc.csv").set_index("system")
    t = pd.read_csv(R(cfg) / "natural" / "rq4_natural_tests.csv")
    t = t[t.test == "N1"].set_index("system")
    out = []
    for m in ["cafebert", "xlmr", "phobert", "visobert", "lr", "svm"]:
        r = d.loc[m]
        a = fix[(fix.model == m) & (fix.system == OURS)]
        aN = f1(a.iloc[0]["nat_N"]) if len(a) else "--"
        out.append(f"{NAME[m]} & {r.acc_N:.1f} & {aN} & {r.acc_C:.1f} & {r.acc_C_strip:.1f} & {r['gap_natural(C-N)']:.1f} & "
                   f"{r['gap_synthetic(C-Cstrip)']:.1f} & {t.loc[m, 'p_holm']:.3f} \\\\")
    out.append(r"\midrule")
    for m, nm in [("lr-PS", "TF-IDF+LR (diacritic-free)"), ("svm-PS", "TF-IDF+SVM (diacritic-free)")]:
        r = d.loc[m]
        out.append(f"{nm} & {r.acc_N:.1f} & -- & {r.acc_C:.1f} & {r.acc_C_strip:.1f} & {r['gap_natural(C-N)']:.1f} & "
                   f"{r['gap_synthetic(C-Cstrip)']:.1f} & -- \\\\")
    cap = (r"Natural versus synthetic diacritic loss (lenient accuracy, models trained on UIT-VSMEC). N: 94 ViGoEmotions "
           r"sentences that naturally lack diacritics (main measure); +DV: the same encoder with dual-view training; C: 282 "
           r"fully accented sentences matched on label and length; C-strip: C with all diacritics removed. Gaps: C$-$N and "
           r"C$-$C-strip. $p_{\text{Holm}}$: two-level bootstrap test of C$-$N, Holm over 6 systems.")
    write(f"{TAB}/v2_natural.tex", table(cap, "tab:natural", "lrrrrrrr",
                                        r"System & N & N (+DV) & C & C-strip & Natural gap & Synthetic gap & $p_{\text{Holm}}$", out))


# ============================================================================ phụ lục: theo lớp
def tab_perclass(cfg):
    c = pd.read_csv(R(cfg) / "ambiguity" / "class_drop.csv")
    order = ["Enjoyment", "Other", "Disgust", "Sadness", "Anger", "Surprise", "Fear"]
    ms = ["cafebert", "xlmr", "phobert", "visobert", "lr"]
    out = []
    for cl in order:
        g = c[c["class"] == cl].set_index("model")
        out.append(f"{cl} & {g['class_emo_confus'].iloc[0]:.3f} & " + " & ".join(f"{g.loc[m, 'drop']:.1f}" for m in ms) + r" \\")
    write(f"{TAB}/v2_perclass.tex", table(
        r"Per-class F1 drop (clean minus 100\% stripped) and the mean confusability of the class's indicative words. "
        r"Classes are ordered by confusability.",
        "tab:perclass", "lrrrrrr", r"Class & Confus. & " + " & ".join(NAME[m] for m in ms), out))


def main():
    cfg = load_config("configs/base.yaml")
    labs = labels(cfg)
    K = len(labs)
    fix = main_fix(cfg, labs, K)
    fig_fix(fix)
    tab_calib_dv(cfg)
    tab_ablation(cfg, fix)
    tab_intervention(cfg)
    tab_calibration(cfg)
    tab_llm(cfg, fix)
    tab_natural(cfg, fix)
    tab_perclass(cfg)


if __name__ == "__main__":
    main()
