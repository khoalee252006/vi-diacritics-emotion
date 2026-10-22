"""Bảng và hình bổ sung cho bài, chạy trên laptop từ kết quả đã có (results/, results_e30/). Không cần GPU.

  perclass     F1 từng lớp (mean ± std qua seed) + ma trận nhầm lẫn (cộng qua seed, chuẩn hóa theo hàng), RQ1
  dataissues   chấm lại RQ1 sau khi bỏ câu test trùng với train hoặc có nhãn mâu thuẫn trong dữ liệu
  curves       hình valid MF1 theo epoch (RQ1, 20 và 30 epoch)
  epochs       bảng test MF1 20 so với 30 epoch (độ nhạy; bộ chính là 20 epoch, chọn theo valid)
  replication  bảng tái lập cấu hình bài gốc (b1_*) và độ nhạy learning rate (b2_*)
  diacfig      hình MF1 theo mức bỏ dấu cho các hệ (từ results/tables/gate_long.csv)

Chạy: python -m src.paper_extra all     (hoặc tên từng mục)
Ra:   results/tables/*.csv|.tex, results/figures/*.png|.pdf
"""
import argparse
import json
import re
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_vsmec
from .preprocess import base_clean
from .utils import enumerate_runs, is_done, labels, load_config, mean_std, p, run_dir, save_table

PLMS = ["cafebert", "xlmr", "phobert", "visobert"]
NAMES = {"cafebert": "CafeBERT", "xlmr": "XLM-R", "phobert": "PhoBERT", "visobert": "ViSoBERT", "lr": "TF-IDF+LR",
         "svm": "TF-IDF+SVM"}
# Bảng màu phân loại tham chiếu (dataviz/references/palette.md), dùng theo thứ tự cố định; kèm marker riêng.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LANG = "vi"   # "en": nhãn tiếng Anh và bỏ tiêu đề chung (bài báo dùng \caption của LaTeX)
TXT = {
    "vi": {"pred": "Nhãn dự đoán", "gold": "Nhãn vàng", "epoch": "Epoch", "valid": "Valid Macro-F1 (%)",
           "test": "Test Macro-F1 (%)", "prep": "Tiền xử lý", "loss": "Tỉ lệ từ bị bỏ dấu (%)",
           "t_conf": "Ma trận nhầm lẫn trên test (% theo hàng, cộng qua 3 seed)",
           "t_curve": "Valid Macro-F1 theo epoch (trung bình qua seed; dải = ±1 std của bản 20 epoch)",
           "t_diac": "Macro-F1 theo mức mất dấu (trung bình ± std qua seed, bộ 20 epoch)",
           "normal": "Model thường", "aug": "Augment hỗn hợp nhiễu", "restore": "Khôi phục dấu → phân loại",
           "gate": "Gating theo tỉ lệ dấu", "gate_aug": "Gating + augment"},
    "en": {"pred": "Predicted label", "gold": "Gold label", "epoch": "Epoch", "valid": "Validation Macro-F1 (%)",
           "test": "Test Macro-F1 (%)", "prep": "Preprocessing", "loss": "Words with diacritics removed (%)",
           "normal": "Standard model", "aug": "Mixed-noise augmentation", "restore": "Restore-then-classify",
           "gate": "Diacritic-aware gating", "gate_aug": "Gating + augmentation"},
}


def T(k):
    return TXT[LANG][k]


def _title(fig, k):
    if LANG == "vi":
        fig.suptitle(T(k), color=INK)


def _runs(cfg, exp, kind=None):
    for spec in enumerate_runs(cfg, exp, kind):
        d = run_dir(cfg, spec)
        if is_done(d):
            yield spec, d


def _metrics(d):
    with open(d / "metrics.json", encoding="utf-8") as f:
        return json.load(f)


def _pred(d, name="test"):
    return pd.read_csv(d / f"pred_{name}.csv", dtype={"id": str})


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "lines.linewidth": 1.5, "lines.markersize": 4.5, "savefig.dpi": 200,
                         "figure.constrained_layout.use": True})
    return plt


def _save(fig, out, stem):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight")
    print(f"[fig] {out / stem}.png / .pdf")


# ----------------------------------------------------------------------------- A3
def perclass(cfg, tab):
    from sklearn.metrics import confusion_matrix, f1_score
    labs = labels(cfg)
    per, conf = {}, {}
    for spec, d in _runs(cfg, "rq1"):
        df = _pred(d)
        f1 = 100 * f1_score(df["gold"], df["pred"], labels=labs, average=None, zero_division=0)
        per.setdefault((spec.model, spec.prep), []).append(f1)
        cm = confusion_matrix(df["gold"], df["pred"], labels=labs)
        conf[(spec.model, spec.prep)] = conf.get((spec.model, spec.prep), 0) + cm
    gold = _pred(next(_runs(cfg, "rq1"))[1])["gold"]
    rows = [{"model": "support (test)", "prep": "", **{c: int((gold == c).sum()) for c in labs}}]
    for (m, pr), v in sorted(per.items(), key=lambda x: (x[0][1], x[0][0])):
        v = np.array(v)
        rows.append({"model": NAMES.get(m, m), "prep": pr, **{c: mean_std(v[:, i]) for i, c in enumerate(labs)}})
    save_table(pd.DataFrame(rows), tab / "rq1_perclass", "RQ1: F1 từng lớp trên test (mean ± std qua seed).")
    crow = []
    for (m, pr), cm in sorted(conf.items()):
        pct = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
        for i, g in enumerate(labs):
            crow.append({"model": m, "prep": pr, "gold": g, **{f"pred_{c}": round(float(pct[i, j]), 1)
                                                               for j, c in enumerate(labs)}})
    save_table(pd.DataFrame(crow), tab / "rq1_confusion",
               "RQ1: ma trận nhầm lẫn (% theo hàng nhãn vàng), cộng qua các seed.")
    return conf


def confusion_fig(cfg, conf, figs):
    plt = _style()
    labs = labels(cfg)
    lncs = LANG == "en"   # khổ chữ LNCS ~12 cm = 6.4 in: vẽ đúng cỡ để chữ không bị thu nhỏ khi chèn vào bài
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3) if lncs else (9.2, 4.0))
    for ax, pr in zip(axes, ["P0", "P1"]):
        cm = conf[("cafebert", pr)]
        pct = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        ax.grid(False)
        for i in range(len(labs)):
            for j in range(len(labs)):
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=6.5 if lncs else 7,
                        color="white" if pct[i, j] > 55 else INK)
        if lncs:
            ax.tick_params(labelsize=7)
        ax.set_xticks(range(len(labs)), labs, rotation=45, ha="right")
        ax.set_yticks(range(len(labs)), labs)
        ax.set_xlabel(T("pred"))
        ax.set_title(f"CafeBERT, {pr}", color=INK)
    axes[0].set_ylabel(T("gold"))
    _title(fig, "t_conf")
    _save(fig, figs, "rq1_confusion_cafebert")


# ----------------------------------------------------------------------------- A4
def dataissues(cfg, tab):
    sp = {s: load_vsmec(cfg, s) for s in ["train", "valid", "test"]}
    # Khóa = chỉ chữ và số (bỏ dấu câu, emoji, khoảng trắng): bắt cả câu gần trùng, chỉ khác emoji/dấu câu.
    key = lambda t: re.sub(r"[^\w]+", "", base_clean(str(t)))
    for s, df in sp.items():
        df["key"] = df["text"].map(key)
    allrows = pd.concat([df.assign(split=s) for s, df in sp.items()])
    conflict = set(allrows.groupby("key")["label"].nunique().loc[lambda x: x > 1].index)
    train_keys = set(sp["train"]["key"])
    test = sp["test"]
    dup = set(test.loc[test["key"].isin(train_keys), "id"])
    conf_ids = set(test.loc[test["key"].isin(conflict), "id"])
    drop = dup | conf_ids
    print(f"[data] câu test trùng train: {len(dup)} | câu test có nhãn mâu thuẫn trong dữ liệu: {len(conf_ids)} "
          f"| bỏ tổng: {len(drop)} / {len(test)}")
    print(f"[data] số khóa có nhãn mâu thuẫn trên cả 3 split: {len(conflict)}")
    from sklearn.metrics import f1_score
    labs = labels(cfg)
    res = {}
    for spec, d in _runs(cfg, "rq1"):
        df = _pred(d)
        sub = df[~df["id"].isin(drop)]
        f = lambda x: 100 * f1_score(x["gold"], x["pred"], labels=labs, average="macro", zero_division=0)
        res.setdefault((spec.model, spec.prep), []).append((f(df), f(sub)))
    rows = []
    for (m, pr), v in res.items():
        full, sub = [a for a, _ in v], [b for _, b in v]
        rows.append({"model": NAMES.get(m, m), "prep": pr, "MF1_full": mean_std(full), "MF1_subset": mean_std(sub),
                     "delta": round(st.mean(sub) - st.mean(full), 2), "_f": st.mean(full), "_s": st.mean(sub)})
    t = pd.DataFrame(rows)
    t["rank_full"] = t.groupby("prep")["_f"].rank(ascending=False).astype(int)
    t["rank_subset"] = t.groupby("prep")["_s"].rank(ascending=False).astype(int)
    t = t.sort_values(["prep", "rank_full"]).drop(columns=["_f", "_s"])
    save_table(t, tab / "rq1_dataissues",
               f"RQ1: MF1 trên toàn bộ test so với test đã bỏ {len(drop)} câu trùng train / nhãn mâu thuẫn.")
    print(t.to_string(index=False))


# ----------------------------------------------------------------------------- B1
def _histories(cfg):
    out = {}
    for spec, d in _runs(cfg, "rq1", "plm"):
        out.setdefault((spec.model, spec.prep), []).append([h["valid_sel_mf1"] for h in _metrics(d)["history"]])
    return out


def curves(cfg, cfg30, figs):
    plt = _style()
    h20, h30 = _histories(cfg), (_histories(cfg30) if cfg30 else {})
    lncs = LANG == "en"
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4) if lncs else (9.2, 3.6), sharey=True)
    for ax, pr in zip(axes, ["P0", "P1"]):
        for k, m in enumerate(PLMS):
            if (m, pr) in h20:
                a = np.array(h20[(m, pr)])
                x = np.arange(1, a.shape[1] + 1)
                ax.plot(x, a.mean(0), color=SERIES[k], marker=MARKERS[k], markevery=4, label=f"{NAMES[m]} (20 ep)")
                ax.fill_between(x, a.mean(0) - a.std(0), a.mean(0) + a.std(0), color=SERIES[k], alpha=0.12, lw=0)
            if (m, pr) in h30:
                b = np.array(h30[(m, pr)])
                ax.plot(np.arange(1, b.shape[1] + 1), b.mean(0), color=SERIES[k], ls="--", lw=1.0,
                        label=f"{NAMES[m]} (30 ep)")
        ax.set_title(f"{T('prep')} {pr}", color=INK)
        ax.set_xlabel(T("epoch"))
    axes[0].set_ylabel(T("valid"))
    if lncs:
        axes[0].set_ylim(20, None)   # bỏ vùng epoch 1 rất thấp để phần đi ngang (điều cần đọc) rõ hơn
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="outside lower center", ncol=4, fontsize=7)
    else:
        axes[1].legend(loc="lower right", fontsize=7, ncol=2)
    _title(fig, "t_curve")
    _save(fig, figs, "valid_curves")


# ----------------------------------------------------------------------------- B2
def epochs(cfg, cfg30, tab):
    def test_mf1(c):
        r = {}
        for spec, d in _runs(c, "rq1", "plm"):
            r.setdefault((spec.model, spec.prep), []).append(_metrics(d)["metrics"]["test"]["mf1"])
        return r
    a, b = test_mf1(cfg), test_mf1(cfg30)
    rows = []
    for k in sorted(set(a) & set(b), key=lambda x: (x[1], x[0])):
        rows.append({"model": NAMES[k[0]], "prep": k[1], "test_MF1_20ep": mean_std(a[k]), "test_MF1_30ep": mean_std(b[k]),
                     "delta": round(st.mean(b[k]) - st.mean(a[k]), 2), "_a": st.mean(a[k]), "_b": st.mean(b[k])})
    t = pd.DataFrame(rows)
    t["rank_20ep"] = t.groupby("prep")["_a"].rank(ascending=False).astype(int)
    t["rank_30ep"] = t.groupby("prep")["_b"].rank(ascending=False).astype(int)
    t = t.drop(columns=["_a", "_b"])
    save_table(t, tab / "epoch_sensitivity",
               "Độ nhạy theo số epoch: test MF1 (RQ1, PLM). Bộ chính 20 epoch được chọn theo valid (epoch_selection).")
    print(t.to_string(index=False))


# ----------------------------------------------------------------------------- B3
def _get(cfg, exp, run):
    d = p(cfg, "results_dir") / exp / run
    if not is_done(d):
        return None
    m = _metrics(d)
    return m["metrics"]["test"]["mf1"], max(h["valid_sel_mf1"] for h in m["history"]), m["best_epoch"]


def replication(cfg, tab):
    seeds = [42, 43, 44]
    spec = [
        ("ViSoBERT", "P0", "b1_visobert_paper", "visobert", "bài gốc: batch 40, 10 epoch, lr 2e-5",
         "65.88 (Nguyen et al., 2023, B4, tr.6)"),
        ("PhoBERT-base", "P1", "b1_phobert_smtce", "phobert", "SMTCE: batch 8, 2 epoch",
         "65.44 (SMTCE, Nguyen et al., 2022, B5, tr.7); 61.41 (Nguyen et al., 2023, B4, tr.6)"),
    ]
    rows = []
    for name, pr, exp, m, desc, pub in spec:
        for lab, e in [(desc, exp), ("công thức chung: batch 16, 20 epoch, lr 2e-5", "rq1")]:
            v = [x for x in (_get(cfg, e, f"{m}_{pr}_none_s{s}") for s in seeds) if x]
            rows.append({"model": name, "prep": pr, "cau_hinh": lab, "n_seeds": len(v),
                         "valid_MF1": mean_std([x[1] for x in v]), "test_MF1": mean_std([x[0] for x in v]),
                         "best_epoch": mean_std([x[2] for x in v], 1), "so_cong_bo_MF1": pub})
    t = pd.DataFrame(rows)
    save_table(t, tab / "b1_replication", "Tái lập cấu hình train của bài gốc trong cùng pipeline (seed 42–44).")
    print(t.drop(columns="so_cong_bo_MF1").to_string(index=False))
    rows = []
    for m in PLMS:
        for lr, e in [("1e-5", "b2_lr1e5"), ("2e-5", "rq1"), ("3e-5", "b2_lr3e5")]:
            x = _get(cfg, e, f"{m}_P0_none_s42")
            if x:
                rows.append({"model": NAMES[m], "lr": lr, "valid_MF1": round(x[1], 2), "test_MF1": round(x[0], 2),
                             "best_epoch": x[2]})
    t = pd.DataFrame(rows)
    t["best_lr_theo_valid"] = t.groupby("model")["valid_MF1"].transform(lambda v: v == v.max())
    save_table(t, tab / "b2_lr_sensitivity", "Độ nhạy theo learning rate (P0, seed 42, 20 epoch). Chọn theo valid.")


# ----------------------------------------------------------------------------- B5
def diacfig(cfg, tab, figs):
    plt = _style()
    df = pd.read_csv(tab / "gate_long.csv")
    lv = {"test": 0, "diac25": 25, "diac50": 50, "diac75": 75, "diac100": 100}
    df = df[df["set"].isin(lv)].assign(level=lambda x: x["set"].map(lv))
    meths = [(m, T(m)) for m in ["normal", "aug", "restore", "gate", "gate_aug"]]
    # Bản tiếng Anh cho LNCS (khổ chữ hẹp, ~12 cm): lưới 2x2, chú thích chung bên dưới. Bản tiếng Việt: 1 hàng 4 ô.
    grid = LANG == "en"
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.4), sharey=True, sharex=True) if grid else \
        plt.subplots(1, 4, figsize=(11.5, 3.3), sharey=True)
    axes = np.ravel(axes)
    for ax, m in zip(axes, PLMS):
        for k, (meth, lab) in enumerate(meths):
            g = df[(df["model"] == m) & (df["method"] == meth)].groupby("level")["mf1"]
            if g.ngroups == 0:
                continue
            mu, sd = g.mean(), g.std().fillna(0)
            ax.errorbar(mu.index, mu.values, yerr=sd.values, color=SERIES[k], marker=MARKERS[k], capsize=2,
                        elinewidth=0.8, label=lab)
        ax.set_title(NAMES[m], color=INK)
        ax.set_xticks([0, 25, 50, 75, 100])
        if not grid or ax in axes[2:]:
            ax.set_xlabel(T("loss"))
    for ax in (axes[::2] if grid else axes[:1]):
        ax.set_ylabel(T("test"))
    if grid:
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="outside lower center", ncol=3, fontsize=8)
    else:
        axes[0].legend(loc="lower left", fontsize=7)
    _title(fig, "t_diac")
    _save(fig, figs, "mf1_vs_diacritic_loss")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all",
                    choices=["all", "perclass", "dataissues", "curves", "epochs", "replication", "diacfig"])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--config30", default="configs/e30.yaml")
    ap.add_argument("--lang", default="vi", choices=["vi", "en"])
    ap.add_argument("--figdir", help="thư mục ghi hình (mặc định <results_dir>/figures)")
    a = ap.parse_args()
    global LANG
    LANG = a.lang
    cfg = load_config(a.config)
    cfg30 = load_config(a.config30) if Path(a.config30).exists() else None
    tab = p(cfg, "results_dir") / "tables"
    figs = Path(a.figdir) if a.figdir else p(cfg, "results_dir") / "figures"
    w = a.what
    if w in ("all", "perclass"):
        confusion_fig(cfg, perclass(cfg, tab), figs)
    if w in ("all", "dataissues"):
        dataissues(cfg, tab)
    if w in ("all", "curves"):
        curves(cfg, cfg30, figs)
    if w in ("all", "epochs") and cfg30:
        epochs(cfg, cfg30, tab)
    if w in ("all", "replication"):
        replication(cfg, tab)
    if w in ("all", "diacfig"):
        diacfig(cfg, tab, figs)


if __name__ == "__main__":
    main()
