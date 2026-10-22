"""Thí nghiệm bổ sung theo nhận xét của thầy (review/17_thiet_ke_bo_sung_theo_nhan_xet.md), dùng checkpoint RQ1 P0, không train.

E5 `clear`: trên câu test có >= 1 từ class-indicative MƠ HỒ và >= 1 từ class-indicative KHÔNG mơ hồ (confus < 0.2, có dấu):
    k = min(hai số); amb_r = bỏ dấu k từ mơ hồ, clear_r = bỏ dấu k từ không mơ hồ (5 lần rút, seed 0).
    Phụ: trên câu có >= 1 từ không mơ hồ: clearall = bỏ dấu mọi từ không mơ hồ, crand_r = cùng số âm tiết ngẫu nhiên
    (không phải từ class-indicative).
E6 `occl`: mỗi PLM (checkpoint seed 42) chấm độ quan trọng theo occlusion trên TRAIN: p(gold | câu) − p(gold | câu bỏ
    một âm tiết có dấu). Gộp theo (từ, lớp) trên câu thuộc lớp đó, từ xuất hiện >= 5 lần; top-50 mỗi lớp cảm xúc.
    Rồi lặp lại can thiệp cấp câu của Phần 1 (src/intervention.py, thiết kế `multi`) với danh sách riêng của model.

Chạy: python -m src.intervention_rev clear-build | clear-infer | clear-analyze | occl-score | occl-build | occl-infer | occl-analyze
(PhoBERT cần JAVA_HOME cho VnCoreNLP.) Kết quả: results/ambiguity/intervention_rev/
"""
import argparse
import re

import numpy as np
import pandas as pd

from .ambiguity import CONFUS_MIN, MODELS, SYL, out_dir
from .data import load_vsmec
from .intervention import N_DRAWS, _resources, _strip_at
from .noise import strip_diacritics
from .preprocess import base_clean
from .utils import compute_metrics, get_device, labels, load_config, p

PLMS = ["cafebert", "xlmr", "phobert", "visobert"]
TOPK, MIN_COUNT = 50, 5


def od(cfg):
    d = out_dir(cfg) / "intervention_rev"
    d.mkdir(exist_ok=True)
    return d


def _write_sets(path, sets):
    path.mkdir(parents=True, exist_ok=True)
    for name, rows in sets.items():
        pd.DataFrame(rows).to_csv(path / f"{name}.csv", index=False, encoding="utf-8")


def _read_sets(path):
    return {f.stem: pd.read_csv(f, dtype={"id": str}, keep_default_na=False) for f in sorted(path.glob("*.csv"))}


# ============================================================================ E5
def clear_build(cfg):
    confus, _, emo_all, amb_emo = _resources(out_dir(cfg))
    clear = {w for w in emo_all if w in confus and confus[w] < CONFUS_MIN and strip_diacritics(w) != w}
    t = load_vsmec(cfg, "test")
    A = {"clean": [], **{f"amb_{r}": [] for r in range(N_DRAWS)}, **{f"clear_{r}": [] for r in range(N_DRAWS)}}
    B = {"clean": [], "clearall": [], **{f"crand_{r}": [] for r in range(N_DRAWS)}}
    rng = np.random.default_rng(0)
    for sid, txt, lab, y in zip(t["id"].astype(str), t["text"], t["label"], t["y"]):
        s = base_clean(txt)
        occ = [(m.start(), m.end(), m.group()) for m in SYL.finditer(s)]
        acc = [i for i, (_, _, w) in enumerate(occ) if strip_diacritics(w) != w]
        ae = [i for i in acc if occ[i][2] in amb_emo]
        ce = [i for i in acc if occ[i][2] in clear]
        rn = [i for i in acc if occ[i][2] not in emo_all]
        base = {"id": sid, "label": lab, "y": y}
        sp = lambda ix: _strip_at(s, [occ[i][:2] for i in ix])
        if ae and ce:
            k = min(len(ae), len(ce))
            A["clean"].append({**base, "text": s})
            for r in range(N_DRAWS):
                A[f"amb_{r}"].append({**base, "text": sp(rng.choice(ae, k, replace=False))})
                A[f"clear_{r}"].append({**base, "text": sp(rng.choice(ce, k, replace=False))})
        if ce and len(rn) >= len(ce):
            B["clean"].append({**base, "text": s})
            B["clearall"].append({**base, "text": sp(ce)})
            for r in range(N_DRAWS):
                B[f"crand_{r}"].append({**base, "text": sp(rng.choice(rn, len(ce), replace=False))})
    _write_sets(od(cfg) / "clear_vs_amb", A)
    _write_sets(od(cfg) / "clear_vs_rand", B)
    print(f"[E5] clear_vs_amb: {len(A['clean'])} câu; clear_vs_rand: {len(B['clean'])} câu; "
          f"{len(clear)} từ không mơ hồ có dấu, {len(amb_emo)} từ mơ hồ")


def _infer(cfg, set_dir, models, tag):
    """Chấm mọi tập trong set_dir bằng checkpoint RQ1 P0 của các model; ghi set_dir/<run>/pred_<tập>.csv."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .baseline_tfidf import load_model
    from .preprocess import prepare
    from .train_plm import evaluate_and_save
    from .utils import save_predictions
    sets = {k: {"df": v[["id", "text", "label", "y"]], "dictnorm": False, "kind": "single"} for k, v in _read_sets(set_dir).items()}
    labs = labels(cfg)
    R = p(cfg, "results_dir") / "rq1"
    device = get_device(cfg)
    for m in models:
        for s in MODELS[m]:
            run = f"{m}_P0_none_s{s}"
            dd = set_dir / run
            dd.mkdir(exist_ok=True)
            todo = {k: v for k, v in sets.items() if not (dd / f"pred_{k}.csv").exists()}
            if not todo:
                continue
            if m in ("lr", "svm"):
                model = load_model(R / run / "model.joblib")
                for name, es in todo.items():
                    sc = model.scores(prepare(es["df"]["text"].tolist(), "P0", cfg))
                    save_predictions(dd / f"pred_{name}.csv", es["df"]["id"].tolist(), es["df"]["label"].tolist(), sc.argmax(1), sc, labs)
            else:
                ck = R / run / "checkpoint"
                tok = AutoTokenizer.from_pretrained(ck)
                model = AutoModelForSequenceClassification.from_pretrained(ck, dtype=torch.float32).to(device)
                mcfg = cfg["models"][m]
                ccfg = {**cfg, "train": {**cfg["train"], **mcfg.get("train", {})}}
                evaluate_and_save(model, tok, todo, "P0", ccfg, device, False, mcfg.get("segment", False), dd)
                del model
                torch.cuda.empty_cache()
            print(f"[{tag}] {run}: {len(todo)} tập")


def _preds(set_dir, runs, name, labs):
    return np.array([[labs.index(q) for q in pd.read_csv(set_dir / r / f"pred_{name}.csv", dtype={"id": str})["pred"]] for r in runs])


def _gold(set_dir, run, labs):
    return np.array([labs.index(g) for g in pd.read_csv(set_dir / run / "pred_clean.csv", dtype={"id": str})["gold"]])


def clear_analyze(cfg):
    from .stats import hier_bootstrap, holm
    labs = labels(cfg)
    K = len(labs)
    out = []
    for sub, (na, nb) in {"clear_vs_amb": ("amb", "clear"), "clear_vs_rand": ("clearall", "crand")}.items():
        sd = od(cfg) / sub
        for m, seeds in MODELS.items():
            runs = [f"{m}_P0_none_s{s}" for s in seeds]
            gold = _gold(sd, runs[0], labs)
            draws = lambda nm: (np.vstack([_preds(sd, runs, f"{nm}_{r}", labs) for r in range(N_DRAWS)])
                                if (sd / runs[0] / f"pred_{nm}_0.csv").exists() else _preds(sd, runs, nm, labs))
            A, B, C = draws(na), draws(nb), _preds(sd, runs, "clean", labs)
            mf = lambda M: np.mean([compute_metrics(gold, row, K)["mf1"] for row in M])
            d, pv, lo, hi = hier_bootstrap(gold, A, B, K, 10000, 0)
            out.append({"test": sub, "model": m, "n_sent": len(gold), "clean": mf(C), na: mf(A), nb: mf(B),
                        "delta": d, "lo": lo, "hi": hi, "p": pv})
    r = pd.DataFrame(out)
    r["p_holm"] = np.nan
    for sub in r.test.unique():
        msk = r.test == sub
        r.loc[msk, "p_holm"] = holm(r.loc[msk, "p"].to_numpy())
    r.round(4).to_csv(od(cfg) / "E5_results.csv", index=False)
    print(r.round(3).to_string(index=False))
    a = r[(r.test == "clear_vs_amb") & r.model.isin(PLMS)]
    n_sig = int(((a.delta < 0) & (a.p_holm < 0.05)).sum())
    print(f"\nE5 (chốt trước): amb_k < clear_k có ý nghĩa ở {n_sig}/4 PLM -> "
          f"{'mơ hồ gây hại THÊM ngoài tính class-indicative' if n_sig >= 3 else 'KHÔNG đủ: viết lại cơ chế theo hướng dạng không dấu của từ khóa'}")


# ============================================================================ E6
def _delete_at(s, a, b):
    return re.sub(r"\s{2,}", " ", (s[:a] + s[b:])).strip()


def occl_build_train(cfg):
    """Biến thể bỏ một âm tiết có dấu trên TRAIN (dùng chung cho mọi model)."""
    confus, _, _, _ = _resources(out_dir(cfg))
    tr = load_vsmec(cfg, "train")
    clean, var = [], []
    for pos, (sid, txt, lab, y) in enumerate(zip(tr["id"].astype(str), tr["text"], tr["label"], tr["y"])):
        s = base_clean(txt)
        clean.append({"id": str(pos), "text": s, "label": lab, "y": y})
        for i, m_ in enumerate(SYL.finditer(s)):
            w = m_.group()
            if strip_diacritics(w) == w or w not in confus:
                continue
            var.append({"id": f"{pos}_{i}", "text": _delete_at(s, m_.start(), m_.end()), "label": lab, "y": y,
                        "pos": pos, "word": w})
    d = od(cfg) / "occl_train"
    d.mkdir(exist_ok=True)
    pd.DataFrame(clean).to_csv(d / "train_clean.csv", index=False, encoding="utf-8")
    pd.DataFrame(var).to_csv(d / "train_var.csv", index=False, encoding="utf-8")
    print(f"[E6] train: {len(clean)} câu, {len(var):,} biến thể")


def occl_score(cfg):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .train_plm import evaluate_and_save
    d = od(cfg) / "occl_train"
    if not (d / "train_var.csv").exists():
        occl_build_train(cfg)
    clean = pd.read_csv(d / "train_clean.csv", dtype={"id": str}, keep_default_na=False)
    var = pd.read_csv(d / "train_var.csv", dtype={"id": str}, keep_default_na=False)
    sets = {k: {"df": v[["id", "text", "label", "y"]], "dictnorm": False, "kind": "single"} for k, v in
            {"train_clean": clean, "train_var": var}.items()}
    device = get_device(cfg)
    for m in PLMS:
        run = f"{m}_P0_none_s42"
        dd = d / run
        dd.mkdir(exist_ok=True)
        todo = {k: v for k, v in sets.items() if not (dd / f"pred_{k}.csv").exists()}
        if not todo:
            continue
        ck = p(cfg, "results_dir") / "rq1" / run / "checkpoint"
        tok = AutoTokenizer.from_pretrained(ck)
        model = AutoModelForSequenceClassification.from_pretrained(ck, dtype=torch.float32).to(device)
        mcfg = cfg["models"][m]
        ccfg = {**cfg, "train": {**cfg["train"], **mcfg.get("train", {})}}
        evaluate_and_save(model, tok, todo, "P0", ccfg, device, False, mcfg.get("segment", False), dd)
        del model
        torch.cuda.empty_cache()
        print(f"[E6] occlusion {run} xong")


def occl_lists(cfg):
    labs = labels(cfg)
    confus, _, _, _ = _resources(out_dir(cfg))
    d = od(cfg) / "occl_train"
    var = pd.read_csv(d / "train_var.csv", dtype={"id": str}, keep_default_na=False)
    emo_classes = [c for c in labs if c != "Other"]
    rows = []
    for m in PLMS:
        dd = d / f"{m}_P0_none_s42"
        pc = pd.read_csv(dd / "pred_train_clean.csv", dtype={"id": str})
        pv = pd.read_csv(dd / "pred_train_var.csv", dtype={"id": str})
        gi = np.array([labs.index(g) for g in var["label"]])
        Pc = pc[[f"s_{l}" for l in labs]].to_numpy()[var["pos"].to_numpy(), gi]
        Pv = pv[[f"s_{l}" for l in labs]].to_numpy()[np.arange(len(var)), gi]
        x = var.assign(imp=Pc - Pv)
        for c in emo_classes:
            g = x[x.label == c].groupby("word")["imp"].agg(["mean", "size"])
            g = g[g["size"] >= MIN_COUNT].sort_values("mean", ascending=False).head(TOPK)
            for w, r in g.iterrows():
                rows.append({"model": m, "class": c, "word": w, "imp": r["mean"], "n": int(r["size"]), "confus": confus[w]})
    L = pd.DataFrame(rows)
    L.to_csv(od(cfg) / "E6_word_lists.csv", index=False, encoding="utf-8")
    print(L.groupby("model").agg(n_words=("word", "nunique"), amb_share=("confus", lambda s: float((s >= CONFUS_MIN).mean()))))
    return L


def occl_build(cfg):
    L = occl_lists(cfg)
    confus, _, _, _ = _resources(out_dir(cfg))
    t = load_vsmec(cfg, "test")
    for m in PLMS:
        emo_all = set(L[L.model == m]["word"])
        amb_emo = {w for w in emo_all if confus.get(w, -1) >= CONFUS_MIN}
        sets = {"clean": [], "amb_emo": [], **{f"rand_{r}": [] for r in range(N_DRAWS)}, **{f"ambother_{r}": [] for r in range(N_DRAWS)}}
        rng = np.random.default_rng(0)
        for sid, txt, lab, y in zip(t["id"].astype(str), t["text"], t["label"], t["y"]):
            s = base_clean(txt)
            occ = [(mm.start(), mm.end(), mm.group()) for mm in SYL.finditer(s)]
            acc = [i for i, (_, _, w) in enumerate(occ) if strip_diacritics(w) != w]
            ae = [i for i in acc if occ[i][2] in amb_emo]
            rn = [i for i in acc if occ[i][2] not in emo_all]
            an = [i for i in rn if confus.get(occ[i][2], -1) >= CONFUS_MIN]
            k = len(ae)
            if not (k >= 1 and len(rn) >= k and len(an) >= k):
                continue
            base = {"id": sid, "label": lab, "y": y}
            sets["clean"].append({**base, "text": s})
            sets["amb_emo"].append({**base, "text": _strip_at(s, [occ[i][:2] for i in ae])})
            for r in range(N_DRAWS):
                sets[f"rand_{r}"].append({**base, "text": _strip_at(s, [occ[i][:2] for i in rng.choice(rn, k, replace=False)])})
                sets[f"ambother_{r}"].append({**base, "text": _strip_at(s, [occ[i][:2] for i in rng.choice(an, k, replace=False)])})
        _write_sets(od(cfg) / f"occl_{m}", sets)
        print(f"[E6] {m}: {len(sets['clean'])} câu, {len(amb_emo)} từ mơ hồ / {len(emo_all)} từ")


def occl_infer(cfg):
    for m in PLMS:
        _infer(cfg, od(cfg) / f"occl_{m}", [m], "E6")


def occl_analyze(cfg):
    from .stats import hier_bootstrap, holm
    labs = labels(cfg)
    K = len(labs)
    out = []
    for m in PLMS:
        sd = od(cfg) / f"occl_{m}"
        runs = [f"{m}_P0_none_s{s}" for s in MODELS[m]]
        gold = _gold(sd, runs[0], labs)
        A = _preds(sd, runs, "amb_emo", labs)
        mf = lambda M: np.mean([compute_metrics(gold, row, K)["mf1"] for row in M])
        rec = {"model": m, "n_sent": len(gold), "clean": mf(_preds(sd, runs, "clean", labs)), "amb_emo": mf(A)}
        for nm in ["rand", "ambother"]:
            B = np.vstack([_preds(sd, runs, f"{nm}_{r}", labs) for r in range(N_DRAWS)])
            d, pv, lo, hi = hier_bootstrap(gold, A, B, K, 10000, 0)
            rec.update({nm: mf(B), f"d_vs_{nm}": d, f"lo_vs_{nm}": lo, f"hi_vs_{nm}": hi, f"p_vs_{nm}": pv})
        out.append(rec)
    r = pd.DataFrame(out)
    adj = holm(np.r_[r["p_vs_rand"], r["p_vs_ambother"]])
    r["p_holm_vs_rand"], r["p_holm_vs_ambother"] = adj[:len(r)], adj[len(r):]
    r.round(4).to_csv(od(cfg) / "E6_results.csv", index=False)
    print(r.round(3).to_string(index=False))
    n = int(((r.d_vs_rand < 0) & (r.p_holm_vs_rand < 0.05)).sum())
    print(f"\nE6 (chốt trước): vs ngẫu nhiên có ý nghĩa {n}/4 PLM -> {'ĐẠT' if n >= 3 else 'KHÔNG ĐẠT'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["clear-build", "clear-infer", "clear-analyze", "occl-score", "occl-build", "occl-infer",
                                     "occl-analyze", "all"])
    ap.add_argument("--config", default="configs/base.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    steps = {"clear-build": clear_build,
             "clear-infer": lambda c: [_infer(c, od(c) / sub, list(MODELS), "E5") for sub in ["clear_vs_amb", "clear_vs_rand"]],
             "clear-analyze": clear_analyze, "occl-score": occl_score, "occl-build": occl_build, "occl-infer": occl_infer,
             "occl-analyze": occl_analyze}
    if a.step == "all":
        for k in ["clear-build", "clear-infer", "clear-analyze", "occl-score", "occl-build", "occl-infer", "occl-analyze"]:
            steps[k](cfg)
    else:
        steps[a.step](cfg)


if __name__ == "__main__":
    main()
