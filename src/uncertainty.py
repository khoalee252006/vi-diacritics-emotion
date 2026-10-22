"""Đo độ bất định khi mất dấu — thực hiện ĐÚNG thiết kế chốt trước trong review/07_thiet_ke_bat_dinh.md.

H1 hiệu chỉnh (ECE 15 bin, Brier, NLL; thô và sau temperature scaling fit trên valid sạch của từng run).
H2 dự đoán có chọn lọc (MSP, entropy H[p̄], MI = H[p̄] − mean_s H[p_s]; AURC, AUROC).
H3/H4/H5: xem các hàm tương ứng.
Ghép theo VỊ TRÍ dòng (id test VSMEC bị trùng). SVM loại (không có xác suất).

Chạy: python -m src.uncertainty h1|h2|h3|h4|h5   -> results/uncertainty/
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .ambiguity import MODELS
from .data import load_vsmec
from .stats import holm
from .utils import labels, load_config, p

PLMS = ["cafebert", "xlmr", "phobert", "visobert"]
SETS = ["test", "diac25", "diac50", "diac75", "diac100"]
N_BOOT = 10000
EPS = 1e-6


def out_dir(cfg):
    d = p(cfg, "results_dir") / "uncertainty"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_probs(cfg, labs, model, set_name, exp="rq1", prep="P0"):
    """(S, N, K) xác suất theo seed, (N,) gold. Kiểm tra thứ tự dòng giống nhau giữa các seed."""
    R = p(cfg, "results_dir") / exp
    P, ids, gold = [], None, None
    for s in MODELS[model]:
        x = pd.read_csv(R / f"{model}_{prep}_none_s{s}" / f"pred_{set_name}.csv", dtype={"id": str})
        if ids is None:
            ids, gold = x["id"].tolist(), np.array([labs.index(g) for g in x["gold"]])
        assert x["id"].tolist() == ids
        q = np.clip(x[[f"s_{l}" for l in labs]].to_numpy(float), EPS, None)
        P.append(q / q.sum(1, keepdims=True))
    return np.stack(P), gold


def temp_scale(P, T):
    lg = np.log(P) / T
    lg -= lg.max(-1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(-1, keepdims=True)


def fit_T(P, y):
    nll = lambda T: -np.log(temp_scale(P, T)[np.arange(len(y)), y]).mean()
    return minimize_scalar(nll, bounds=(0.05, 50), method="bounded").x


def ece(P, y, bins=15):
    conf, pred = P.max(1), P.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            tot += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return 100 * tot


def brier(P, y):
    Y = np.eye(P.shape[1])[y]
    return ((P - Y) ** 2).sum(1).mean()


def nll(P, y):
    return -np.log(P[np.arange(len(y)), y]).mean()


# ----------------------------------------------------------------------------- H1
def h1(cfg):
    labs = labels(cfg)
    rows, boots = [], []
    rng = np.random.default_rng(0)
    for m in PLMS + ["lr"]:
        Pv, yv = load_probs(cfg, labs, m, "valid")
        Ts = [fit_T(Pv[i], yv) for i in range(len(Pv))]
        cal = {}
        for sname in SETS:
            P, y = load_probs(cfg, labs, m, sname)
            for ver in ["raw", "ts"]:
                Q = P if ver == "raw" else np.stack([temp_scale(P[i], Ts[i]) for i in range(len(P))])
                cal[(sname, ver)] = (Q, y)
                rows.append({"model": m, "set": sname, "version": ver, "T_mean": np.mean(Ts),
                             "ECE": np.mean([ece(q, y) for q in Q]), "Brier": np.mean([brier(q, y) for q in Q]),
                             "NLL": np.mean([nll(q, y) for q in Q]),
                             "conf_on_wrong": np.mean([q.max(1)[q.argmax(1) != y].mean() for q in Q]),
                             "acc": np.mean([100 * (q.argmax(1) == y).mean() for q in Q])})
        if m == "lr":
            continue
        for ver in ["raw", "ts"]:   # ΔECE (bỏ dấu 100% − sạch), bootstrap hai tầng: câu dùng chung + seed
            (Qc, y), (Qd, _) = cal[("test", ver)], cal[("diac100", ver)]
            S, N = Qc.shape[:2]
            obs = np.mean([ece(Qd[i], y) - ece(Qc[i], y) for i in range(S)])
            bs = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx, ss = rng.integers(0, N, N), rng.integers(0, S, S)
                bs[b] = np.mean([ece(Qd[i][idx], y[idx]) - ece(Qc[i][idx], y[idx]) for i in ss])
            pv = (1 + np.sum(np.abs(bs - obs) >= abs(obs))) / (N_BOOT + 1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            boots.append({"model": m, "version": ver, "dECE": obs, "ci_lo": lo, "ci_hi": hi, "p": pv})
    t = pd.DataFrame(rows)
    tt = pd.DataFrame(boots)
    tt["p_holm"] = holm(tt["p"].to_numpy())
    tt["sig"] = tt["p_holm"] < 0.05
    d = out_dir(cfg)
    t.round(4).to_csv(d / "h1_calibration.csv", index=False)
    tt.round(4).to_csv(d / "h1_tests.csv", index=False)
    print(t.pivot_table(index=["model", "version"], columns="set", values="ECE").round(2)[SETS].to_string())
    print("\nĐộ tin cậy TB trên câu SAI (thô):")
    print(t[t.version == "raw"].pivot_table(index="model", columns="set", values="conf_on_wrong").round(3)[SETS].to_string())
    print("\nT (fit trên valid sạch):", t.groupby("model").T_mean.first().round(2).to_dict())
    print("\nH1: ΔECE (bỏ dấu 100% − sạch), Holm 8:\n", tt.round(3).to_string(index=False))


# ----------------------------------------------------------------------------- H2
def signals(P):
    """P: (S, N, K). Trả về p̄ và các tín hiệu bất định (càng lớn càng bất định)."""
    pb = P.mean(0)
    H = lambda q: -(q * np.log(q)).sum(-1)
    ent = H(pb)
    return pb, {"MSP": 1 - pb.max(1), "entropy": ent, "MI": ent - H(P).mean(0)}


def aurc(u, err):
    o = np.argsort(u, kind="stable")
    e = err[o].astype(float)
    return 100 * (np.cumsum(e) / np.arange(1, len(e) + 1)).mean()


def auroc(u, err):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(err, u) if 0 < err.sum() < len(err) else np.nan


def h2(cfg):
    labs = labels(cfg)
    rows, tests = [], []
    rng = np.random.default_rng(0)
    pairs = [("MI", "entropy"), ("MI", "MSP"), ("entropy", "MSP")]
    for m in PLMS + ["lr"]:
        for sname in SETS:
            P, y = load_probs(cfg, labs, m, sname)
            pb, sig = signals(P)
            err = (pb.argmax(1) != y).astype(int)
            for k, u in sig.items():
                if m == "lr" and k == "MI":
                    continue
                rows.append({"model": m, "set": sname, "signal": k, "err_rate": 100 * err.mean(),
                             "AURC": aurc(u, err), "AUROC": auroc(u, err)})
            rows.append({"model": m, "set": sname, "signal": "random", "err_rate": 100 * err.mean(),
                         "AURC": 100 * err.mean(), "AUROC": 0.5})
            if sname != "diac100" or m == "lr":
                continue
            N = len(y)
            idxs = rng.integers(0, N, (N_BOOT, N))
            for a, b in pairs:
                obs = aurc(sig[a], err) - aurc(sig[b], err)
                bs = np.array([aurc(sig[a][i], err[i]) - aurc(sig[b][i], err[i]) for i in idxs])
                pv = (1 + np.sum(np.abs(bs - obs) >= abs(obs))) / (N_BOOT + 1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                tests.append({"model": m, "a": a, "b": b, "dAURC": obs, "ci_lo": lo, "ci_hi": hi, "p": pv})
    t = pd.DataFrame(rows)
    tt = pd.DataFrame(tests)
    tt["p_holm"] = holm(tt["p"].to_numpy())
    tt["sig"] = tt["p_holm"] < 0.05
    d = out_dir(cfg)
    t.round(4).to_csv(d / "h2_selective.csv", index=False)
    tt.round(4).to_csv(d / "h2_tests.csv", index=False)
    print("AUROC (phân biệt câu sai), sạch vs bỏ dấu 100%:")
    print(t[t.set.isin(["test", "diac100"])].pivot_table(index=["model", "signal"], columns="set", values="AUROC").round(3).to_string())
    print("\nAURC (thấp = tốt) trên bỏ dấu 100%:")
    print(t[t.set == "diac100"].pivot_table(index="model", columns="signal", values="AURC").round(2).to_string())
    print("\nH2: ΔAURC (âm = tín hiệu a tốt hơn b), Holm 12:\n", tt.round(3).to_string(index=False))


# ----------------------------------------------------------------------------- H3 (dùng tập can thiệp của Phần 1)
IV = None


def iv_dir(cfg):
    return p(cfg, "results_dir") / "ambiguity" / "intervention"


def iv_probs(cfg, labs, model, name):
    """(S, N, K) xác suất trên tập can thiệp `name` (pred_<name>.csv), thứ tự dòng như file."""
    P, ids = [], None
    for s in MODELS[model]:
        x = pd.read_csv(iv_dir(cfg) / f"{model}_P0_none_s{s}" / f"pred_{name}.csv", dtype={"id": str})
        if ids is None:
            ids = x["id"].tolist()
        assert x["id"].tolist() == ids
        q = np.clip(x[[f"s_{l}" for l in labs]].to_numpy(float), EPS, None)
        P.append(q / q.sum(1, keepdims=True))
    return np.stack(P), ids


def ent(q):
    return -(q * np.log(q)).sum(-1)


def single_deltas(cfg, labs, model, inst):
    """ΔH, ΔMI, Δp(gold) cho từng biến thể một âm tiết so với câu sạch tương ứng (p̄ theo seed)."""
    Pc, _ = iv_probs(cfg, labs, model, "single_clean")
    Pv, ids = iv_probs(cfg, labs, model, "single")
    assert ids == inst["id"].tolist()
    Pc = Pc[:, inst["sent_pos"].to_numpy()]
    _, sc = signals(Pc)
    _, sv = signals(Pv)
    gi = np.array([labs.index(g) for g in inst["label"]])
    ix = np.arange(len(inst))
    return pd.DataFrame({"dH": sv["entropy"] - sc["entropy"], "dMI": sv["MI"] - sc["MI"],
                         "dp": 100 * (Pc.mean(0)[ix, gi] - Pv.mean(0)[ix, gi])})


def h3(cfg):
    import statsmodels.formula.api as smf
    labs = labels(cfg)
    od = iv_dir(cfg)
    inst = pd.read_csv(od / "single_instances.csv", dtype={"id": str, "sent_id": str})
    lex = pd.read_csv(p(cfg, "results_dir") / "ambiguity" / "lexicon.csv", keep_default_na=False).set_index("word")["count"]
    inst["log_freq"] = np.log(inst["word"].map(lex))
    rows, cells, tests = [], [], []
    for m in PLMS:
        x = pd.concat([inst, single_deltas(cfg, labs, m, inst)], axis=1)
        for (e, a), g in x.groupby(["emo", "amb"]):
            cells.append({"model": m, "emo": e, "amb": a, "n": len(g), "dH": g.dH.mean(), "dMI": g.dMI.mean()})
        for yv in ["dH", "dMI"]:
            fit = smf.ols(f"{yv} ~ emo * amb + collision + n_syl + rel_pos + log_freq", x).fit(
                cov_type="cluster", cov_kwds={"groups": x["sent_pos"].to_numpy()})
            ci = fit.conf_int()
            r = {"level": "syllable", "model": m, "outcome": yv, "term": "emo:amb", "estimate": fit.params["emo:amb"],
                 "ci_lo": ci.loc["emo:amb", 0], "ci_hi": ci.loc["emo:amb", 1], "p": fit.pvalues["emo:amb"]}
            (tests if yv == "dH" else rows).append(r)   # dH là phép so chính (Holm); dMI là phụ
    # cấp câu: H(amb_emo) - H(đối chứng), p̄ theo seed; đối chứng = TB 5 lần rút
    rng = np.random.default_rng(0)
    for m in PLMS:
        Pa, ids = iv_probs(cfg, labs, m, "multi_amb_emo")
        Ha = signals(Pa)[1]["entropy"]
        for ctl in ["rand", "ambother"]:
            Hc = np.mean([signals(iv_probs(cfg, labs, m, f"multi_{ctl}_{i}")[0])[1]["entropy"] for i in range(5)], 0)
            d = Ha - Hc
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
            pv = (1 + np.sum(np.abs(bs - d.mean()) >= abs(d.mean()))) / (N_BOOT + 1)
            tests.append({"level": "sentence", "model": m, "outcome": "dH", "term": f"amb_emo-{ctl}",
                          "estimate": d.mean(), "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5), "p": pv})
    tt = pd.DataFrame(tests)
    tt["p_holm"] = holm(tt["p"].to_numpy())
    tt["sig"] = tt["p_holm"] < 0.05
    d = out_dir(cfg)
    tt.round(5).to_csv(d / "h3_tests.csv", index=False)
    pd.DataFrame(rows).round(5).to_csv(d / "h3_dmi_secondary.csv", index=False)
    pd.DataFrame(cells).round(5).to_csv(d / "h3_cells.csv", index=False)
    print("ΔH theo ô 2x2 (emo, amb):")
    print(pd.DataFrame(cells).pivot_table(index="model", columns=["emo", "amb"], values="dH").round(4).to_string())
    print("\nH3 (Holm 12):\n", tt.round(4).to_string(index=False))
    print("\nPhụ, ΔMI emo:amb:\n", pd.DataFrame(rows).round(4).to_string(index=False))


# ----------------------------------------------------------------------------- H4 (bất định của từng từ)
ADD_K = 0.1   # làm trơn add-k, chốt trước, không dò


def context_confus(cfg, inst):
    """I-1: p(w | dạng không dấu k, âm tiết trước L, âm tiết sau R) theo naive Bayes trên corpus train:
    p(w'|k,L,R) ∝ p(w'|k) · p(L|w') · p(R|w'), p(L|w') = (c(L,w') + k) / (c(w') + k·V). Trả về confus và entropy theo ngữ cảnh."""
    from collections import Counter
    from .ambiguity import SYL, corpus
    from .preprocess import base_clean
    seqs = [SYL.findall(t) for t in corpus(cfg)]
    uni, left, right = Counter(), Counter(), Counter()
    for q in seqs:
        uni.update(q)
        for a, b in zip(q, q[1:]):
            right[(a, b)] += 1   # (w, R)
            left[(a, b)] += 1    # (L, w) — cùng bảng bigram
    V = len(uni)
    lex = pd.read_csv(p(cfg, "results_dir") / "ambiguity" / "lexicon.csv", keep_default_na=False)
    cands = lex.groupby("key").apply(lambda g: list(zip(g["word"], g["p_given_key"])), include_groups=False).to_dict()
    test = [SYL.findall(base_clean(t)) for t in load_vsmec(cfg, "test")["text"]]
    conf, ent_ = [], []
    for sp, iid, w, k in zip(inst["sent_pos"], inst["id"], inst["word"], inst["key"]):
        i = int(iid.rsplit("_", 1)[1])
        q = test[sp]
        assert q[i] == w
        L = q[i - 1] if i > 0 else None
        R = q[i + 1] if i + 1 < len(q) else None
        sc = []
        for c, pk in cands[k]:
            v = np.log(pk)
            if L is not None:
                v += np.log((left[(L, c)] + ADD_K) / (uni[c] + ADD_K * V))
            if R is not None:
                v += np.log((right[(c, R)] + ADD_K) / (uni[c] + ADD_K * V))
            sc.append((c, v))
        lv = np.array([v for _, v in sc])
        pr = np.exp(lv - lv.max())
        pr /= pr.sum()
        pw = pr[[c for c, _ in sc].index(w)]
        conf.append(1 - pw)
        ent_.append(-(pr * np.log2(np.clip(pr, 1e-12, None))).sum())
    return np.array(conf), np.array(ent_)


def h4(cfg):
    from scipy.stats import spearmanr
    labs = labels(cfg)
    od = iv_dir(cfg)
    inst = pd.read_csv(od / "single_instances.csv", dtype={"id": str, "sent_id": str})
    lex = pd.read_csv(p(cfg, "results_dir") / "ambiguity" / "lexicon.csv", keep_default_na=False).set_index("word")
    inst["key"] = inst["word"].map(lex["key"])
    inst["I0_confus"] = inst["confus"]
    inst["I0_entropy"] = inst["word"].map(lex["key_entropy"])
    inst["I1_confus"], inst["I1_entropy"] = context_confus(cfg, inst)
    f2 = out_dir(cfg) / "h4_I2.csv"
    ctx = ["I1_confus"]
    if f2.exists():
        i2 = pd.read_csv(f2, dtype={"id": str})
        assert i2["id"].tolist() == inst["id"].tolist()
        inst["I2_confus"], inst["I2_entropy"] = i2["I2_confus"].to_numpy(), i2["I2_entropy"].to_numpy()
        ctx.append("I2_confus")
        print(f"[4a] I-2 confus TB: từ cảm xúc {inst[inst.emo == 1].I2_confus.mean():.3f}, toàn bộ {inst.I2_confus.mean():.3f}")
    emo = inst[inst.emo == 1]
    red = 1 - emo["I1_confus"].mean() / emo["I0_confus"].mean()
    print(f"[4a] confus TB trên từ cảm xúc: I-0 = {emo.I0_confus.mean():.3f}, I-1 = {emo.I1_confus.mean():.3f} "
          f"(giảm {100 * red:.1f}%); quy tắc chốt trước: chạy I-2 nếu giảm >= 20% -> {'CÓ' if red >= 0.2 else 'KHÔNG'}")
    print(f"     toàn bộ âm tiết: I-0 = {inst.I0_confus.mean():.3f}, I-1 = {inst.I1_confus.mean():.3f}")
    d = out_dir(cfg)
    words, tests = [], []
    rng = np.random.default_rng(0)
    for m in PLMS:
        x = pd.concat([inst, single_deltas(cfg, labs, m, inst)], axis=1)
        g = x.groupby("word").agg(n=("dH", "size"), emo=("emo", "max"), I0=("I0_confus", "mean"), I1=("I1_confus", "mean"), **({"I2": ("I2_confus", "mean")} if "I2_confus" in x else {}),
                                  dH=("dH", "mean"), dp=("dp", "mean"), dMI=("dMI", "mean")).reset_index()
        words.append(g[g.n >= 5].assign(model=m))
        sp = x["sent_pos"].to_numpy()
        groups = {s: np.where(sp == s)[0] for s in np.unique(sp)}
        keys = np.array(list(groups))
        for yv in ["dH", "dp"]:
            rho = lambda ix, c: spearmanr(x[c].to_numpy()[ix], x[yv].to_numpy()[ix])[0]
            allix = np.arange(len(x))
            for cc in ctx:
                obs = rho(allix, cc) - rho(allix, "I0_confus")
                bs = []
                for _ in range(2000):
                    ix = np.concatenate([groups[s] for s in rng.choice(keys, len(keys))])
                    bs.append(rho(ix, cc) - rho(ix, "I0_confus"))
                bs = np.array(bs)
                pv = (1 + np.sum(np.abs(bs - obs) >= abs(obs))) / (len(bs) + 1)
                tests.append({"model": m, "outcome": yv, "context": cc, "rho_I0": rho(allix, "I0_confus"),
                              "rho_ctx": rho(allix, cc), "d_rho": obs, "ci_lo": np.percentile(bs, 2.5),
                              "ci_hi": np.percentile(bs, 97.5), "p": pv, "primary": yv == "dH"})
    tt = pd.DataFrame(tests)
    prim = tt["primary"].to_numpy()
    tt["p_holm"] = np.nan
    tt.loc[prim, "p_holm"] = holm(tt.loc[prim, "p"].to_numpy())
    tt["sig"] = tt["p_holm"] < 0.05
    W = pd.concat(words)
    inst.round(4).to_csv(d / "h4_instances.csv", index=False)
    W.round(4).to_csv(d / "h4_word_uncertainty.csv", index=False)
    tt.round(4).to_csv(d / "h4_tests.csv", index=False)
    print("\nH4c (chính: dH, Holm trên 4 PLM x số mức ngữ cảnh; phụ: dp, mô tả):\n", tt.round(3).to_string(index=False))
    top = W[W.emo == 1].groupby("word")[["dp", "dH", "I0", "I1"] + (["I2"] if "I2_confus" in inst else [])].mean().sort_values("dp", ascending=False).head(15)
    print("\nTừ cảm xúc gây mất tự tin vào nhãn đúng nhiều nhất (TB 4 PLM):\n", top.round(3).to_string())


def h4r(cfg):
    """I-2: xác suất từng dạng có dấu theo model khôi phục dấu (peterhung/vietnamese-accent-marker-xlm-roberta) tại đúng
    vị trí âm tiết bị bỏ dấu trong câu biến thể. Xấp xỉ (ghi ở nhật ký thiết kế): p(c) ∝ Σ_subword Σ_{tag t: áp t lên
    dạng không dấu ra c} p(t); dạng không đổi (c = k) nhận cả xác suất các tag không áp được; chuẩn hóa trên các ứng viên
    trong lexicon. -> results/uncertainty/h4_I2.csv"""
    import torch
    from .ambiguity import SYL
    from .noise import strip_diacritics
    from .restore_diacritics import load_restorer
    from .utils import get_device
    device = get_device(cfg)
    tok, model, tags, _ = load_restorer(device)
    inst = pd.read_csv(iv_dir(cfg) / "single_instances.csv", dtype={"id": str, "sent_id": str}, keep_default_na=False)
    lex = pd.read_csv(p(cfg, "results_dir") / "ambiguity" / "lexicon.csv", keep_default_na=False)
    keyof = lex.set_index("word")["key"]
    cands = lex.groupby("key")["word"].apply(list).to_dict()
    split = [t.split("-") for t in tags]
    out = []
    B = 64
    for st in range(0, len(inst), B):
        chunk = inst.iloc[st:st + B]
        words, tgt = [], []
        for text, iid, w in zip(chunk["text"], chunk["id"], chunk["word"]):
            i = int(iid.rsplit("_", 1)[1])
            occ = list(SYL.finditer(text))
            a, b = occ[i].start(), occ[i].end()
            assert text[a:b] == strip_diacritics(w), (text, w)
            ws, pos, j = text.split(), 0, None
            for jj, wtok in enumerate(ws):
                k0 = text.index(wtok, pos)
                if k0 <= a < k0 + len(wtok):
                    j, off = jj, a - k0
                    break
                pos = k0 + len(wtok)
            words.append(ws)
            tgt.append((j, off, b - a))
        enc = tok(words, is_split_into_words=True, truncation=True, max_length=512, padding=True, return_tensors="pt")
        with torch.no_grad():
            pr = torch.softmax(model(**{k: v.to(device) for k, v in enc.items()}).logits.float(), -1).cpu().numpy()
        for bi, ((j, off, ln), w) in enumerate(zip(tgt, chunk["word"])):
            k = keyof[w]
            cs = cands[k]
            seg = words[bi][j][off:off + ln]
            mass = dict.fromkeys(cs, 0.0)
            for posi, wid in enumerate(enc.word_ids(bi)):
                if wid != j:
                    continue
                for t, (raw, vowel) in enumerate(split):
                    c = seg.replace(raw, vowel, 1) if raw and raw in seg else seg
                    if c in mass:
                        mass[c] += pr[bi, posi, t]
            z = sum(mass.values())
            pw = mass[w] / z if z > 0 else np.nan
            q = np.array([v / z for v in mass.values()]) if z > 0 else np.array([np.nan])
            out.append({"id": chunk["id"].iloc[bi], "I2_confus": 1 - pw,
                        "I2_entropy": float(-(q * np.log2(np.clip(q, 1e-12, None))).sum())})
    o = pd.DataFrame(out)
    o.to_csv(out_dir(cfg) / "h4_I2.csv", index=False)
    print(f"[I-2] {len(o)} biến thể; NaN: {int(o.I2_confus.isna().sum())}; confus TB = {o.I2_confus.mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["h1", "h2", "h3", "h4", "h4r", "h5"])
    a = ap.parse_args()
    cfg = load_config("configs/base.yaml")
    globals()[a.step](cfg)


if __name__ == "__main__":
    main()
