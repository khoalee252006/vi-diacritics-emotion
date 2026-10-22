"""Sinh bảng LaTeX (tiếng Anh, booktabs) cho bài từ các CSV trong results/tables/. Không chép tay số nào.

Chạy: python -m src.paper_tables          -> manuscript/tables/*.tex
"""
import argparse
from pathlib import Path

import pandas as pd

from .utils import load_config, p

NAMES = {"cafebert": "CafeBERT", "xlmr": "XLM-R", "phobert": "PhoBERT", "visobert": "ViSoBERT",
         "lr": "TF-IDF+LR", "svm": "TF-IDF+SVM"}
PLMS = ["cafebert", "xlmr", "phobert", "visobert"]
METH = {"normal": "Standard", "aug": "Augmentation", "restore": "Restore-then-classify", "strip": "Diacritic-free only",
        "ens2": "Ensemble of 2 standard", "gate": "Accent-ratio gating (G0)", "gate_rst": "G0 with restorer",
        "gate_aug": "G0 + augmentation", "avg": "Uniform average", "dictnorm": "Dictionary normalisation"}


def pm(s):
    """'63.51 ± 2.50' -> '63.51$_{\\pm 2.50}$'."""
    s = str(s)
    if "±" not in s:
        return s
    a, b = [x.strip() for x in s.split("±")]
    return f"{a}$_{{\\pm {b}}}$"


def mean_of(s):
    try:
        return float(str(s).split("±")[0])
    except ValueError:
        return float("nan")


def write(out, name, body, caption, label, cols, size="\\small", star=False):
    """Kiểu Springer LNCS: chú thích ĐẶT TRÊN bảng; bảng chỉ bị thu nhỏ nếu rộng hơn khổ chữ (adjustbox max width).
    `star` giữ để tương thích, LNCS một cột nên luôn dùng môi trường table."""
    tex = ["\\begin{table}[!htbp]","\\centering", f"\\caption{{{caption}}}\\label{{{label}}}", size,
           "\\begin{adjustbox}{max width=\\textwidth}", f"\\begin{{tabular}}{{{cols}}}", "\\toprule", *body,
           "\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}", ""]
    (out / f"{name}.tex").write_text("\n".join(tex), encoding="utf-8", newline="\n")
    print(f"[tex] {out / name}.tex")


def t_rq1(tab, out):
    m = pd.read_csv(tab / "rq1_main.csv")
    s = pd.read_csv(tab / "rq1_significance.csv")
    body = ["Prep. & Model & Seeds & Acc & WF1 & MF1 & Best ep. \\\\", "\\midrule"]
    for k, pr in enumerate(["P0", "P1"]):
        g = m[m["prep"] == pr].assign(_m=lambda x: x["MF1"].map(mean_of)).sort_values("_m", ascending=False)
        for i, r in enumerate(g.itertuples()):
            best = "\\textbf{" if i == 0 else ""
            mf1 = pm(r.MF1)
            body.append(f"{pr if i == 0 else ''} & {NAMES[r.model]} & {r.n_seeds} & {pm(r.Acc)} & {pm(r.WF1)} & "
                        f"{best}{mf1}{'}' if best else ''} & {pm(r.best_epoch) if r.best_epoch != '-' else '--'} \\\\")
        if k == 0:
            body.append("\\midrule")
    sig = s[(s["comparison"].str.startswith("model")) & (s["hboot_p_holm_sig"])]
    def fmt(a, b, d, lo, hi):   # bên thắng đứng trước, hiệu số dương
        if d < 0:
            a, b, d, lo, hi = b, a, -d, -hi, -lo
        return (f"{NAMES[a.split('/')[1]]} $>$ {NAMES[b.split('/')[1]]} ({a.split('/')[2]}, "
                f"$\\Delta$={d:+.2f}, 95\\% CI [{lo:.2f}, {hi:.2f}])")
    sig_txt = "; ".join(fmt(*v) for v in sig[["A", "B", "delta_MF1", "hboot_ci95_lo", "hboot_ci95_hi"]].values)
    n = int((s["comparison"].str.startswith("model")).sum())
    verb = "remains" if len(sig) == 1 else "remain"
    cap = ("Controlled comparison on the UIT-VSMEC test set (mean$\\pm$std over seeds; identical fine-tuning recipe "
           "for all PLMs; P0 = minimal cleaning, P1 = lexical normalisation and emoji-to-text). "
           f"Of {n} pairwise model comparisons, only {len(sig)} {verb} significant under a two-level (seed and "
           f"sentence) bootstrap with Holm correction: {sig_txt if len(sig) else 'none'}.")
    write(out, "tab_rq1", body, cap, "tab:rq1", "llrccc c")


def t_cost(tab, out):
    c = pd.read_csv(tab / "rq1_cost.csv")
    body = ["Model & Params & Train (min) & Inference (ms/sent.) & Peak VRAM (GB) \\\\", "\\midrule"]
    for r in c.itertuples():
        params = f"{r.n_params / 1e6:.1f}M" if r.n_params > 1e6 else f"{r.n_params / 1e3:.0f}K"
        body.append(f"{NAMES[r.model]} & {params} & {r.train_min_median:.2f} & {pm(r.infer_ms_per_sent)} & "
                    f"{pm(r.peak_vram_gb)} \\\\")
    write(out, "tab_cost", body, "Computational cost on one RTX 4090 (median training time per run; TF-IDF "
          "excludes the one-off hyper-parameter grid search).", "tab:cost", "lrrrr")


def t_replication(tab, out):
    b = pd.read_csv(tab / "b1_replication.csv")
    body = ["Model & Configuration & Valid MF1 & Test MF1 & Best ep. & Published MF1 \\\\", "\\midrule"]
    pub = {"ViSoBERT": "65.88~\\cite{nguyen2023visobert}",
           "PhoBERT-base": "65.44~\\cite{nguyen2022smtce}; 61.41~\\cite{nguyen2023visobert}"}
    short = {"bài gốc: batch 40, 10 epoch, lr 2e-5": "Original: bs 40, 10 ep.",
             "SMTCE: batch 8, 2 epoch": "SMTCE: bs 8, 2 ep.",
             "công thức chung: batch 16, 20 epoch, lr 2e-5": "Ours: bs 16, 20 ep."}
    for i, r in enumerate(b.itertuples()):
        cfg = short[r.cau_hinh]
        first = i % 2 == 0
        body.append(f"{r.model + ' (' + r.prep + ')' if first else ''} & {cfg} & {pm(r.valid_MF1)} & "
                    f"{pm(r.test_MF1)} & {pm(r.best_epoch)} & {pub[r.model] if first else ''} \\\\")
        if i == 1:
            body.append("\\midrule")
    write(out, "tab_replication", body, "Re-running the published fine-tuning configurations inside our pipeline "
          "(seeds 42--44; bs = batch size, ep.\\ = epochs; learning rate $2\\cdot10^{-5}$ in all runs, as SMTCE does "
          "not report one). Neither configuration recovers the published scores, and both trail our common recipe.",
          "tab:replication", "llcccl")


def t_robust(tab, out):
    g = pd.read_csv(tab / "gate_main.csv")
    cols = ["test", "diac50", "diac100", "teen30", "typo30", "avg_noisy"]
    hdr = ["Clean", "Diac.\\,50\\%", "Diac.\\,100\\%", "Teen.\\,30\\%", "Typo\\,30\\%", "Avg.\\ noisy"]
    meths = ["normal", "aug", "restore", "ens2", "gate", "gate_aug"]
    short = {"normal": "Standard", "aug": "Augmentation", "restore": "Restore$\\rightarrow$classify",
             "ens2": "2-model ensemble", "gate": "Accent-ratio gating (G0)", "gate_aug": "G0 + aug."}
    body = ["Model & System & " + " & ".join(hdr) + " \\\\", "\\midrule"]
    for k, m in enumerate(PLMS + ["lr"]):
        sub = g[(g["model"] == m) & (g["method"].isin(meths))].copy()
        sub["method"] = pd.Categorical(sub["method"], meths, ordered=True)
        sub = sub.sort_values("method")
        best = {c: sub[c].map(mean_of).max() for c in cols}
        for i, r in enumerate(sub.itertuples()):
            cells = []
            for c in cols:
                v = getattr(r, c)
                mu, sd = [x.strip() for x in str(v).split("±")]
                cell = f"\\textbf{{{mu}}}" if abs(mean_of(v) - best[c]) < 1e-9 else mu
                cells.append(f"{cell}$_{{\\pm {sd}}}$")
            body.append(f"{NAMES[m] if i == 0 else ''} & {short[r.method]} & " + " & ".join(cells) + " \\\\")
        if k < len(PLMS):
            body.append("\\midrule")
    write(out, "tab_robust", body, "Robustness to synthetic noise (test Macro-F1, mean$\\pm$std over 5 seeds, 3 for "
          "CafeBERT; best mean per model in bold). Diac.\\ = share of words with diacritics removed; Teen.\\ = "
          "teencode; Avg.\\ noisy = mean over all 12 noisy test sets. Restore$\\rightarrow$classify is identical across "
          "diacritic levels because every input is first stripped and then restored. The 2-model ensemble averages "
          "the standard models of seeds $i$ and $i{+}1$; for TF-IDF+LR it equals the standard model because logistic "
          "regression training is deterministic (std = 0).", "tab:robust", "llcccccc", size="\\footnotesize")


def t_gate_tests(tab, out):
    t = pd.read_csv(tab / "gate_tests.csv")
    pairs = [("gate", "normal"), ("gate", "restore"), ("gate", "ens2"), ("gate", "aug"), ("gate", "dictnorm"),
             ("gate", "avg"), ("gate_aug", "aug"), ("gate_rst", "normal"), ("gate", "gate_rst")]
    body = ["Comparison & \\multicolumn{2}{c}{Sentence bootstrap} & \\multicolumn{2}{c}{Two-level bootstrap} & $n$ \\\\",
            "& wins & losses & wins & losses & \\\\", "\\midrule"]
    for a, b in pairs:
        g = t[(t["A"] == a) & (t["B"] == b)]
        if g.empty:
            continue
        w1 = int(((g["sig_holm"]) & (g["delta_MF1"] > 0)).sum())
        l1 = int(((g["sig_holm"]) & (g["delta_MF1"] < 0)).sum())
        if "hsig_holm" in g:
            w2 = int(((g["hsig_holm"]) & (g["delta_MF1"] > 0)).sum())
            l2 = int(((g["hsig_holm"]) & (g["delta_MF1"] < 0)).sum())
        else:
            w2 = l2 = "--"
        body.append(f"{METH[a]} vs.\\ {METH[b]} & {w1} & {l1} & {w2} & {l2} & {len(g)} \\\\")
    write(out, "tab_gate_tests", body, f"Significant wins/losses (Holm-corrected over all {len(t)} comparisons, "
          "$\\alpha=0.05$) across 6 models and 7 test sets (clean, diacritic removal 25--100\\%, teencode 30\\%, "
          "typos 30\\%).", "tab:gatetests", "lccccc")


def t_restore(out):
    from .restore_annotation import LABELS, OUT, error_tokens, wilson
    rows, n_alpha = error_tokens()
    raw = 100 * (1 - len(rows) / n_alpha)
    df = pd.read_excel(OUT, sheet_name="chu_thich")
    lab = df["LOAI (dien)"].astype(str).str.strip()
    n = int(lab.isin(LABELS).sum())
    share = {L: wilson(int((lab == L).sum()), n) for L in LABELS}
    real = share["loi_that"][0] / 100
    corr = 100 * (1 - (1 - raw / 100) * real)
    yn = df["DOI_NGHIA_CAM_XUC (dien)"].astype(str).str.strip()
    emo = wilson(int((yn == "co").sum()), int(yn.isin(["co", "khong"]).sum()))
    f = lambda x: f"{x[0]:.1f} [{x[1]:.1f}, {x[2]:.1f}]"
    body = ["Quantity & Value \\\\", "\\midrule",
            "Alpha-word accuracy, Vietnamese, formal text~\\cite[Table 1]{naplava2021diacritics} & 98.53 \\\\",
            f"Alpha-word accuracy, VSMEC test, our restorer (raw; {n_alpha:,} words) & {raw:.2f} \\\\",
            f"\\quad corrected after manual error analysis & {corr:.2f} \\\\", "\\midrule",
            f"Sampled errors ($n={n}$): genuine error (\\%) & {f(share['loi_that'])} \\\\",
            f"\\quad plausible variant (\\%) & {f(share['bien_the_hop_ly'])} \\\\",
            f"\\quad correction of noisy gold (\\%) & {f(share['sua_loi_goc'])} \\\\",
            f"Errors that change the conveyed emotion (\\%) & {f(emo)} \\\\"]
    write(out, "tab_restore", body, "Diacritic restoration transfers poorly to social-media emotion text. Error "
          "categories follow N\\'aplava et al.~\\cite{naplava2021diacritics}; brackets give 95\\% Wilson intervals. "
          "Labels were "
          "assigned by an LLM and checked by the authors.", "tab:restore", "lr")


def t_gate_real(tab, out):
    """Kết quả âm tính: gating trên dữ liệu nhiễu thật (vigo_new_test) + thành phần nhóm câu bị gắn cờ."""
    import re
    g = pd.read_csv(tab / "gate_vigo.csv")
    cap_src = (tab / "gate_vigo.tex").read_text(encoding="utf-8")
    n_all, n_low = [int(x) for x in re.findall(r"(\d+) câu", cap_src)[:2]]
    lab = pd.read_csv(Path("review/gate_vigo_flagged_labels.csv"))["loai"].value_counts()
    sgn = lambda d: f"{d:+.2f}"
    body = [f"Model & Seeds & \\multicolumn{{3}}{{c}}{{All ({n_all:,} sent.)}} & "
            f"\\multicolumn{{3}}{{c}}{{Flagged, $w<0.5$ ({n_low} sent.)}} \\\\",
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
            " & & Standard & Gating & $\\Delta$ & Standard & Gating & $\\Delta$ \\\\", "\\midrule"]
    for m in ["cafebert", "xlmr", "phobert", "visobert", "lr", "svm"]:
        r = g[g["model"] == m].iloc[0]
        body.append(f"{NAMES[m]} & {r.n_seeds} & {pm(r.all_std)} & {pm(r.all_gate)} & {sgn(r.d_all)} & "
                    f"{pm(r.low_std)} & {pm(r.low_gate)} & {sgn(r.d_low)} \\\\")
    comp = {"that_su_bo_dau": "genuinely undiacritised", "ten_rieng_tag": "names or @-tags",
            "teencode_viet_tat": "teencode/abbreviations", "tu_von_khong_dau": "words that carry no diacritics",
            "tieng_anh": "English", "khac": "other"}
    parts = ", ".join(f"{int(lab.get(k, 0))} {v}" for k, v in comp.items())
    write(out, "tab_gate_real", body, "Diacritic-ratio gating on real social-media text (new part of "
          "ViGoEmotions, models trained on VSMEC, lenient MF1, mean$\\pm$std over seeds). The detector flags "
          f"{n_low} sentences with $w<0.5$; by manual inspection they are: {parts} (categories assigned by an LLM and "
          "checked by the authors).", "tab:gate_real", "lcrrrrrr")


def t_rq4(tab, out):
    m = pd.read_csv(tab / "rq4_main.csv")
    k = pd.read_csv(tab / "rq4_kappa.csv")
    body = ["Model & Prep. & VSMEC$\\rightarrow$ViGo MF1 & In-domain ceiling MF1 \\\\", "\\midrule"]
    for mod in ["cafebert", "xlmr", "phobert", "visobert", "lr", "svm"]:
        for pr in ["P0", "P1"]:
            x = m[(m["model"] == mod) & (m["prep"] == pr) & (m["setting"] == "VSMEC->ViGo")]
            c = m[(m["model"] == mod) & (m["prep"] == pr) & (m["setting"] == "in-domain ceiling")]
            if x.empty:
                continue
            body.append(f"{NAMES[mod] if pr == 'P0' else ''} & {pr} & {pm(x['MF1'].iloc[0])} & "
                        f"{pm(c['MF1'].iloc[0]) if not c.empty else '--'} \\\\")
    ka = k[k["mapping"] == "a"].iloc[0]
    cap = ("Cross-dataset transfer to the new part of ViGoEmotions after mapping its 28 labels to the 7 VSMEC "
           "classes (lenient scoring; mean$\\pm$std over 5 seeds, 3 for CafeBERT). The in-domain ceiling is trained "
           "on ViGoEmotions itself (3 seeds, P0 only). On the "
           f"{int(ka['n_used']):,} sentences shared by both corpora, the two label sets agree on "
           f"{ka['agree_rate_vsmec_in_groups']:.1f}\\% of items (Cohen's $\\kappa$ = {ka['kappa_lenient']:.2f} lenient, "
           f"{ka['kappa_strict(single-group rows)']:.2f} strict).")
    write(out, "tab_rq4", body, cap, "tab:rq4", "llcc")


def t_lr_epochs(tab, out):
    lr = pd.read_csv(tab / "b2_lr_sensitivity.csv")
    body = ["Model & lr & Valid MF1 & Test MF1 & Best ep. \\\\", "\\midrule"]
    for i, r in enumerate(lr.itertuples()):
        v = f"\\textbf{{{r.valid_MF1:.2f}}}" if r.best_lr_theo_valid else f"{r.valid_MF1:.2f}"
        body.append(f"{r.model if i % 3 == 0 else ''} & {r.lr} & {v} & {r.test_MF1:.2f} & {r.best_epoch} \\\\")
    write(out, "tab_lr", body, "Learning-rate sensitivity (P0, one seed). Bold: best on validation.", "tab:lr", "llccc")
    e = pd.read_csv(tab / "epoch_sensitivity.csv")
    sel = pd.read_csv(tab / "epoch_selection.csv")
    body = ["Model & Prep. & Valid 20 ep. & Valid 30 ep. & Test 20 ep. & Test 30 ep. \\\\", "\\midrule"]
    for r in e.itertuples():
        s = sel[(sel["model"] == r.model.lower().replace("-", "")) & (sel["prep"] == r.prep)]
        va = f"{s.iloc[0, 2]:.2f}" if not s.empty else "--"
        vb = f"{s.iloc[0, 3]:.2f}" if not s.empty else "--"
        body.append(f"{r.model} & {r.prep} & {va} & {vb} & {pm(r.test_MF1_20ep)} & {pm(r.test_MF1_30ep)} \\\\")
    tot = sel[sel["model"] == "TRUNG BÌNH"].iloc[0]
    write(out, "tab_epochs", body, f"Epoch budget. The rule fixed before the 30-epoch runs (adopt 30 epochs only if "
          f"mean validation MF1 improves by $\\geq$0.5) kept 20 epochs: {tot.iloc[2]:.2f} vs.\\ {tot.iloc[3]:.2f}. "
          "Test scores are shown only as a sensitivity check.", "tab:epochs", "llcccc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="manuscript/tables")
    a = ap.parse_args()
    tab = p(load_config(a.config), "results_dir") / "tables"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (t_rq1, t_cost, t_replication, t_robust, t_gate_tests, t_gate_real, t_rq4, t_lr_epochs):
        f(tab, out)
    t_restore(out)


if __name__ == "__main__":
    main()
