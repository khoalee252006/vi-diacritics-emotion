"""RQ4 — mất dấu tự nhiên trên ViGoEmotions (thiết kế chốt trước: review/11_thiet_ke_rq4_tu_nhien.md).

  build   : tập N (nhãn cuối từ review/rq4_natural_candidates.xlsx), C (đối chứng có dấu ghép cặp), C_strip
  infer   : chấm bằng checkpoint RQ1 P0 (4 PLM), LR/SVM P0 và PS  -> results/natural/<run>/pred_<tập>.csv
  analyze : N1, N2, N3 -> results/natural/*.csv
Chạy: python -m src.natural build | infer | analyze
"""
import argparse
import unicodedata

import numpy as np
import pandas as pd

from .ambiguity import MODELS
from .data import add_groups, load_vigo, load_vsmec
from .gate_fusion import _probs, diac_ratio, gate_weights
from .noise import strip_diacritics
from .stats import holm
from .utils import canonical_group, get_device, labels, load_config, p

SPLIT = {"train": "train", "val": "valid", "test": "test"}
SETS = ["N", "C", "C_strip"]
N_BOOT = 10000


def out_dir(cfg):
    d = p(cfg, "results_dir") / "natural"
    d.mkdir(parents=True, exist_ok=True)
    return d


def vigo_all(cfg):
    labs = labels(cfg)
    fr = []
    for sp, lsp in SPLIT.items():
        x = add_groups(load_vigo(cfg, "new", lsp), cfg, "a").assign(split=sp)
        x["id"] = x["id"].astype(str)
        fr.append(x)
    v = pd.concat(fr, ignore_index=True)
    v = v[v["groups"].map(len) > 0].copy()
    v["canon"] = [canonical_group(set(g), vt, labs) for g, vt in zip(v["groups"], v["votes"])]
    return v


def build(cfg):
    lab = pd.read_excel("review/rq4_natural_candidates.xlsx", sheet_name="ung_vien", keep_default_na=False)
    team = lab["nhom_kiem_tra (để trống nếu đồng ý)"].astype(str).str.strip()
    lab["final"] = np.where(team != "", team, lab["claude_label_lan2"])
    lab["final"] = lab["final"].str.split(" ").str[0].str.rstrip(":")
    sc = pd.read_csv(out_dir(cfg).parent / "unfamiliar" / "vigo_natural_scores.csv", keep_default_na=False, dtype={"id": str})
    v = vigo_all(cfg).merge(sc[["split", "id", "miss", "acc", "n"]], on=["split", "id"], how="left")
    keyN = set(zip(lab.loc[lab["final"] == "mat_dau_that", "split"], lab.loc[lab["final"] == "mat_dau_that", "id"].astype(str)))
    isN = [(s, i) in keyN for s, i in zip(v["split"], v["id"])]
    N = v[isN].copy()
    print(f"[build] N: {len(N)} câu có nhóm nhãn (trên {len(keyN)} câu nhãn mat_dau_that)")
    pool = v[~np.array(isN) & (v["miss"] == 0) & (v["acc"] >= 3)].copy()
    rng = np.random.default_rng(0)
    used, rows = set(), []
    for _, r in N.iterrows():
        got = []
        for tol in (3, 6):
            cand = pool[(pool["canon"] == r["canon"]) & ((pool["n"] - r["n"]).abs() <= tol) & ~pool.index.isin(used)]
            need = 3 - len(got)
            if len(cand) >= need:
                pick = rng.choice(cand.index.to_numpy(), need, replace=False)
                got += list(pick)
                break
        used.update(got)
        rows += [{**pool.loc[g].to_dict(), "match_of": r["id"]} for g in got]
    C = pd.DataFrame(rows)
    print(f"[build] C: {len(C)} câu đối chứng (cần {3 * len(N)}); độ dài TB N {N.n.mean():.1f}, C {C.n.mean():.1f}")
    Cs = C.copy()
    Cs["text"] = Cs["text"].map(strip_diacritics)
    d = out_dir(cfg)
    for name, df in [("N", N), ("C", C), ("C_strip", Cs)]:
        df.to_pickle(d / f"set_{name}.pkl")
    print("[build] phân bố nhóm nhãn N:", N["canon"].value_counts().to_dict())


def load_sets(cfg):
    return {k: pd.read_pickle(out_dir(cfg) / f"set_{k}.pkl") for k in SETS}


def infer(cfg):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .baseline_tfidf import load_model
    from .preprocess import prepare
    from .train_plm import evaluate_and_save
    labs = labels(cfg)
    S = load_sets(cfg)
    R = p(cfg, "results_dir")
    device = get_device(cfg)
    es = {k: {"df": v.reset_index(drop=True), "dictnorm": False, "kind": "vigo"} for k, v in S.items()}
    for m in ["lr", "svm"]:
        for prep, exp in [("P0", "rq1"), ("PS", "strip")]:
            run = f"{m}_{prep}_none_s42"
            dd = out_dir(cfg) / run
            dd.mkdir(exist_ok=True)
            model = load_model(R / exp / run / "model.joblib")
            for k, e in es.items():
                sc = model.scores(prepare(e["df"]["text"].tolist(), prep, cfg))
                pd.DataFrame({"id": e["df"]["id"], **{f"s_{l}": sc[:, j] for j, l in enumerate(labs)},
                              "pred": [labs[i] for i in sc.argmax(1)]}).to_csv(dd / f"pred_{k}.csv", index=False)
    for m in ["cafebert", "xlmr", "phobert", "visobert"]:
        for s in MODELS[m]:
            run = f"{m}_P0_none_s{s}"
            dd = out_dir(cfg) / run
            dd.mkdir(exist_ok=True)
            todo = {k: e for k, e in es.items() if not (dd / f"pred_{k}.csv").exists()}
            if not todo:
                continue
            ck = R / "rq1" / run / "checkpoint"
            tok = AutoTokenizer.from_pretrained(ck)
            model = AutoModelForSequenceClassification.from_pretrained(ck, dtype=torch.float32).to(device)
            mcfg = cfg["models"][m]
            ccfg = {**cfg, "train": {**cfg["train"], **mcfg.get("train", {})}}
            evaluate_and_save(model, tok, todo, "P0", ccfg, device, False, mcfg.get("segment", False), dd)
            del model
            torch.cuda.empty_cache()
            print(f"[infer] {run}")


def preds(cfg, labs, run, k, probs=False):
    x = pd.read_csv(out_dir(cfg) / run / f"pred_{k}.csv", dtype={"id": str})
    if probs:
        return x[[f"s_{l}" for l in labs]].to_numpy(float)
    return np.array([labs.index(q) for q in x["pred"]])


def analyze(cfg):
    labs = labels(cfg)
    S = load_sets(cfg)
    gold = {k: [set(labs.index(g) for g in gs) for gs in v["groups"]] for k, v in S.items()}
    r_train = np.nanmean([diac_ratio(t) for t in load_vsmec(cfg, "train")["text"]])
    systems = {m: [f"{m}_P0_none_s{s}" for s in MODELS[m]] for m in ["cafebert", "xlmr", "phobert", "visobert"]}
    systems.update({"lr": ["lr_P0_none_s42"], "svm": ["svm_P0_none_s42"], "lr-PS": ["lr_PS_none_s42"], "svm-PS": ["svm_PS_none_s42"]})
    hit = {}   # (system, set) -> (seeds, n) đúng/sai lenient
    for sysn, runs in systems.items():
        for k in SETS:
            hit[(sysn, k)] = np.stack([np.array([pr in g for pr, g in zip(preds(cfg, labs, r, k), gold[k])]) for r in runs]).astype(float)
    for m in ["lr", "svm"]:
        for k in SETS:
            w = gate_weights(S[k]["text"].tolist(), r_train)[:, None]
            G = w * _probs(preds(cfg, labs, f"{m}_P0_none_s42", k, True), m) + (1 - w) * _probs(preds(cfg, labs, f"{m}_PS_none_s42", k, True), m)
            hit[(f"{m}-gate", k)] = np.array([[g_ in gg for g_, gg in zip(G.argmax(1), gold[k])]], float)
    rows = [{"system": sysn, **{f"acc_{k}": 100 * hit[(sysn, k)].mean() for k in SETS}} for sysn in dict.fromkeys(s for s, _ in hit)]
    tab = pd.DataFrame(rows)
    tab["gap_natural(C-N)"] = tab["acc_C"] - tab["acc_N"]
    tab["gap_synthetic(C-Cstrip)"] = tab["acc_C"] - tab["acc_C_strip"]
    rng = np.random.default_rng(0)

    def boot(a, b):   # a: (S,nN) N, b: (S,nC) C -> acc(N) - acc(C)
        obs = a.mean() - b.mean()
        bs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            sa, sb = rng.integers(0, len(a), len(a)), rng.integers(0, len(b), len(b))
            ia, ib = rng.integers(0, a.shape[1], a.shape[1]), rng.integers(0, b.shape[1], b.shape[1])
            bs[i] = a[sa][:, ia].mean() - b[sb][:, ib].mean()
        return 100 * obs, (1 + np.sum(np.abs(bs - obs) >= abs(obs))) / (N_BOOT + 1), *(100 * np.percentile(bs, [2.5, 97.5]))
    t1 = []
    for m in ["cafebert", "xlmr", "phobert", "visobert", "lr", "svm"]:
        d, pv, lo, hi = boot(hit[(m, "N")], hit[(m, "C")])
        t1.append({"test": "N1", "system": m, "delta_accN_minus_accC": d, "ci_lo": lo, "ci_hi": hi, "p": pv})
    t1 = pd.DataFrame(t1)
    t1["p_holm"] = holm(t1["p"].to_numpy())
    t3 = []
    for m in ["lr", "svm"]:   # (C-N) của PS − (C-N) của P0; âm = PS thu hẹp khoảng cách
        a = hit[(f"{m}-PS", "C")] - 0
        gN = lambda s: hit[(s, "N")][0]
        gC = lambda s: hit[(s, "C")][0]
        obs = (gC(f"{m}-PS").mean() - gN(f"{m}-PS").mean()) - (gC(m).mean() - gN(m).mean())
        bs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            iN, iC = rng.integers(0, len(gN(m)), len(gN(m))), rng.integers(0, len(gC(m)), len(gC(m)))
            bs[i] = (gC(f"{m}-PS")[iC].mean() - gN(f"{m}-PS")[iN].mean()) - (gC(m)[iC].mean() - gN(m)[iN].mean())
        t3.append({"test": "N3", "system": m, "delta_gapPS_minus_gapP0": 100 * obs, "ci_lo": 100 * np.percentile(bs, 2.5),
                   "ci_hi": 100 * np.percentile(bs, 97.5), "p": (1 + np.sum(np.abs(bs - obs) >= abs(obs))) / (N_BOOT + 1)})
    t3 = pd.DataFrame(t3)
    t3["p_holm"] = holm(t3["p"].to_numpy())
    d = out_dir(cfg)
    tab.round(2).to_csv(d / "rq4_natural_acc.csv", index=False)
    pd.concat([t1, t3]).round(4).to_csv(d / "rq4_natural_tests.csv", index=False)
    print(f"N = {len(S['N'])}, C = {len(S['C'])}\n")
    print(tab.round(2).to_string(index=False))
    print("\nN1 (acc N − acc C, Holm 6):\n", t1.round(3).to_string(index=False))
    print("\nN3 (khoảng cách PS − khoảng cách P0; âm = PS thu hẹp, Holm 2):\n", t3.round(3).to_string(index=False))
    k = int(((t1.p_holm < 0.05) & (t1.delta_accN_minus_accC < 0)).sum())
    print(f"\nKết luận N1 (>= 4/6 model): {k}/6 -> {'mất dấu tự nhiên GÂY HẠI' if k >= 4 else 'chưa đủ bằng chứng'}")


# ----------------------------------------------------------------------------- bộ lọc + kiểm định trên ViLexNorm
def detector(cfg):
    """Trả về hàm score(text) -> (miss, acc, n) đúng như bộ lọc đã dùng để chọn ứng viên RQ4."""
    import re
    from .ambiguity import SYL
    lex = pd.read_csv(p(cfg, "results_dir") / "ambiguity" / "lexicon.csv", keep_default_na=False)
    self_p = lex[lex.word == lex.key].set_index("key")["p_given_key"].to_dict()
    keys = set(lex.key)
    tot = lex.groupby("key")["count"].sum().to_dict()

    def score(t):
        t = unicodedata.normalize("NFC", str(t))   # văn bản dạng NFD (dấu tách rời) bị đọc như không dấu -> chuẩn hóa trước
        toks = [w for w in re.findall(r"\S+", t) if not w.startswith("@")]
        syl = [s.lower() for w in toks for s in SYL.findall(w) if not s[:1].isupper()]
        if len(syl) < 4:
            return 0, 0, len(syl)
        miss = sum(1 for s in syl if s == strip_diacritics(s) and s in keys and self_p.get(s, 0) < 0.2 and tot.get(s, 0) >= 20)
        acc = sum(1 for s in syl if s != strip_diacritics(s))
        return miss, acc, len(syl)
    return score


def gold_missing(orig, norm):
    """Đáp án từ ViLexNorm (người chuẩn hóa): gióng từ câu gốc với câu chuẩn hóa theo dạng không dấu; trong các âm tiết
    của câu chuẩn hóa CÓ dấu và gióng được 1-1, đếm bao nhiêu ở câu gốc bị viết không dấu (missing) / có dấu (present).
    Phần bị viết tắt/teencode (không gióng được) bỏ qua."""
    import difflib
    from .ambiguity import SYL
    a = [w.lower() for w in SYL.findall(unicodedata.normalize("NFC", str(orig)))]
    b = [w.lower() for w in SYL.findall(unicodedata.normalize("NFC", str(norm)))]
    sm = difflib.SequenceMatcher(a=[strip_diacritics(x) for x in a], b=[strip_diacritics(x) for x in b], autojunk=False)
    miss = pres = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for x, y in zip(a[i1:i2], b[j1:j2]):
            if y != strip_diacritics(y):
                if x == strip_diacritics(x):
                    miss += 1
                else:
                    pres += 1
    return miss, pres


def vilexnorm(cfg):
    """Kiểm định bộ lọc trên ViLexNorm. Đáp án dương = câu có >= 3 âm tiết lẽ ra có dấu (gióng được) và >= 50% trong
    số đó bị viết không dấu (cùng quy tắc gán RQ4). Ngưỡng chọn trên dev, báo cáo trên test."""
    score = detector(cfg)
    root = p(cfg, "results_dir").parent / "dataset" / "github" / "ViLexNorm" / "data"
    res = {}
    for sp in ["dev", "test"]:
        d = pd.read_csv(root / f"{sp}.csv", keep_default_na=False)
        g = [gold_missing(o, n) for o, n in zip(d["original"], d["normalized"])]
        sc = [score(o) for o in d["original"]]
        x = pd.DataFrame({"g_miss": [a for a, _ in g], "g_pres": [b for _, b in g],
                          "miss": [a for a, _, _ in sc], "acc": [b for _, b, _ in sc]})
        x["g_rate"] = x.g_miss / (x.g_miss + x.g_pres).clip(lower=1)
        x["gold"] = (x.g_miss + x.g_pres >= 3) & (x.g_rate >= 0.5)
        x["rate"] = x.miss / (x.miss + x.acc).clip(lower=1)
        res[sp] = x
        print(f"[{sp}] {len(x)} câu; đáp án dương (mất dấu thật): {int(x.gold.sum())} ({100 * x.gold.mean():.1f}%)")
    rows = []
    for mm in [2, 3, 4]:
        for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            r = {"min_miss": mm, "min_rate": th}
            for sp, x in res.items():
                flag = (x.miss >= mm) & (x.rate >= th)
                tp = int((flag & x.gold).sum())
                r[f"{sp}_flagged"] = int(flag.sum())
                r[f"{sp}_precision"] = 100 * tp / max(flag.sum(), 1)
                r[f"{sp}_recall"] = 100 * tp / max(x.gold.sum(), 1)
            rows.append(r)
    t = pd.DataFrame(rows)
    t.round(2).to_csv(out_dir(cfg) / "vilexnorm_detector.csv", index=False)
    print(t.round(1).to_string(index=False))
    cur = t[(t.min_miss == 3) & (t.min_rate == 0.6)].iloc[0]
    print(f"\nNgưỡng đang dùng cho RQ4 (miss>=3, rate>=0.6) trên TEST: precision {cur.test_precision:.1f}%, "
          f"recall {cur.test_recall:.1f}% ({int(cur.test_flagged)} câu bị gắn cờ)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "infer", "analyze", "vilexnorm"])
    a = ap.parse_args()
    globals()[a.step](load_config("configs/base.yaml"))


if __name__ == "__main__":
    main()
