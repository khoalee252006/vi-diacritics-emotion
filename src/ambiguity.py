"""Phần 1 của hướng mới: vì sao mất dấu làm hỏng nhận diện cảm xúc — độ mơ hồ từ vựng khi bỏ dấu.

Bước 1 `lexicon`: đếm âm tiết có dấu trong văn bản có dấu (VSMEC train + phần mới của ViGoEmotions train), gom theo
          dạng không dấu. Với mỗi dạng không dấu k: các dạng có dấu w và p(w|k); entropy H(k) (bit) theo tần suất.
          Với mỗi dạng có dấu w: confus(w) = 1 - p(w|k) = xác suất đoán sai w nếu chỉ thấy dạng không dấu và chọn
          theo tần suất (đây là "độ mơ hồ" có trọng số, khác với chỉ đếm số dạng).
          Dạng hiếm (< MIN_COUNT lần) bị bỏ để không tính lỗi gõ là một dạng.
Bước 2 `emotion`: từ cảm xúc = unigram có hệ số dương lớn nhất cho từng lớp trong TF-IDF + LR (P0; TF-IDF tất định nên
          seed nào cũng như nhau).
Bước 3 `pairs`: giao hai kết quả: từ cảm xúc có confus cao, kèm lớp, hệ số, tần suất và các dạng tranh chấp.

Chạy: python -m src.ambiguity all   -> results/ambiguity/{lexicon,keys,emotion_words,ambiguous_emotion_words}.csv
Chỉ dùng tập train (không dùng valid/test) để tính thống kê từ vựng.
"""
import argparse
import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .baseline_tfidf import load_model
from .data import load_vigo, load_vsmec
from .noise import strip_diacritics
from .preprocess import base_clean
from .utils import labels, load_config, p

SYL = re.compile(r"[^\W\d_]+")      # âm tiết: chỉ chữ cái
MIN_COUNT = 3                       # dạng có dấu xuất hiện < 3 lần coi như lỗi gõ, bỏ
TOP_K = 50                          # số từ cảm xúc mỗi lớp
CONFUS_MIN = 0.2                    # ngưỡng "mơ hồ" cho bảng cặp từ


def out_dir(cfg):
    d = p(cfg, "results_dir") / "ambiguity"
    d.mkdir(parents=True, exist_ok=True)
    return d


def corpus(cfg):
    texts = load_vsmec(cfg, "train")["text"].tolist() + load_vigo(cfg, "new", "train")["text"].astype(str).tolist()
    return [base_clean(t) for t in texts]


def build_lexicon(cfg):
    cnt = Counter(s for t in corpus(cfg) for s in SYL.findall(t))
    by_key = defaultdict(dict)
    for w, c in cnt.items():
        if c >= MIN_COUNT:
            by_key[strip_diacritics(w)][w] = c
    rows, keys = [], []
    for k, forms in by_key.items():
        tot = sum(forms.values())
        ps = {w: c / tot for w, c in forms.items()}
        H = -sum(q * math.log2(q) for q in ps.values())
        top = sorted(forms, key=forms.get, reverse=True)
        keys.append({"key": k, "n_forms": len(forms), "entropy_bits": round(H, 3), "total": tot,
                     "forms": " | ".join(f"{w}:{forms[w]}" for w in top[:6])})
        for w in forms:
            rows.append({"word": w, "key": k, "count": forms[w], "p_given_key": round(ps[w], 4),
                         "confus": round(1 - ps[w], 4), "key_entropy": round(H, 3), "key_n_forms": len(forms),
                         "has_diacritic": w != k,
                         "competitors": " | ".join(f"{x}:{forms[x]}" for x in top if x != w)[:120]})
    lex = pd.DataFrame(rows).sort_values("count", ascending=False)
    kd = pd.DataFrame(keys).sort_values("total", ascending=False)
    return lex, kd


def emotion_words(cfg, run="lr_P0_none_s42"):
    m = load_model(p(cfg, "results_dir") / "rq1" / run / "model.joblib")
    vocab = m.vec.get_feature_names_out()
    uni = np.array([" " not in v and SYL.fullmatch(v) is not None for v in vocab])
    coef = m.clf.coef_
    classes = list(m.clf.classes_)
    labs = labels(cfg)
    rows = []
    for ci, c in enumerate(classes):
        name = labs[c] if isinstance(c, (int, np.integer)) else str(c)
        w = np.where(uni, coef[ci], -np.inf)
        for rank, j in enumerate(np.argsort(-w)[:TOP_K], 1):
            rows.append({"class": name, "rank": rank, "word": vocab[j], "coef": round(float(coef[ci, j]), 4)})
    return pd.DataFrame(rows)


MODELS = {"lr": [42], "svm": [42], "phobert": [42, 43, 44, 45, 46], "xlmr": [42, 43, 44, 45, 46],
          "visobert": [42, 43, 44, 45, 46], "cafebert": [42, 43, 44]}   # TF-IDF tất định: 1 seed


def sentence_features(cfg, d):
    """Đặc trưng mơ hồ của từng câu test (văn bản sạch, P0)."""
    lex = pd.read_csv(d / "lexicon.csv", keep_default_na=False).set_index("word")
    emo = pd.read_csv(d / "emotion_words_with_ambiguity.csv").query("`class` != \"Other\"")  # Other không mang cảm xúc
    amb_emo = set(emo.loc[emo["confus"] >= CONFUS_MIN, "word"])
    emo_all = set(emo["word"])
    conf = lex["confus"].to_dict()
    t = load_vsmec(cfg, "test")
    rows = []
    for pos, txt in enumerate(t["text"]):   # khóa = vị trí dòng: cột id của test VSMEC bị trùng (693 dòng, 608 id)
        syl = SYL.findall(base_clean(txt))
        other = [conf[s] for s in syl if s in conf and s not in emo_all]
        rows.append({"pos": pos, "n_syl": len(syl), "n_emo": sum(s in emo_all for s in syl),
                     "n_amb_emo": sum(s in amb_emo for s in syl),
                     "mean_confus_other": float(np.mean(other)) if other else 0.0})
    return pd.DataFrame(rows)


def read_pred(f, test_ids):
    """Đọc file dự đoán test VSMEC, kiểm tra đúng thứ tự dòng của test rồi gắn khóa vị trí `pos`."""
    x = pd.read_csv(f, dtype={"id": str})
    assert x["id"].tolist() == test_ids, f"thứ tự dòng khác test: {f}"
    return x.assign(pos=range(len(x)))


def errors(cfg, d):
    """Bước 4: câu chứa nhiều từ cảm xúc mơ hồ có bị đổi từ đúng sang sai khi bỏ dấu 100% nhiều hơn không?
    Logit: flip ~ n_amb_emo + n_emo_khong_mo_ho + n_syl + mean_confus_other, chỉ trên các cặp (câu, seed) đúng ở test sạch;
    sai số chuẩn cụm theo câu. Kèm phân tích theo lớp."""
    import statsmodels.formula.api as smf
    from scipy.stats import spearmanr
    from sklearn.metrics import f1_score
    feats = sentence_features(cfg, d)
    feats["n_emo_clear"] = feats["n_emo"] - feats["n_amb_emo"]
    R = p(cfg, "results_dir") / "rq1"
    labs = labels(cfg)
    emo = pd.read_csv(d / "emotion_words_with_ambiguity.csv")
    cls_conf = {c: np.average(g.confus.fillna(0), weights=g["count"].fillna(1)) for c, g in emo.groupby("class")}
    test_ids = load_vsmec(cfg, "test")["id"].astype(str).tolist()
    res, cls_rows = [], []
    for m, seeds in MODELS.items():
        frames = []
        for s in seeds:
            a = read_pred(R / f"{m}_P0_none_s{s}" / "pred_test.csv", test_ids)
            b = read_pred(R / f"{m}_P0_none_s{s}" / "pred_diac100.csv", test_ids)
            f = pd.DataFrame({"pos": a["pos"], "gold": a["gold"], "ok_clean": a["gold"] == a["pred"],
                              "ok_diac": b["gold"] == b["pred"], "seed": s})
            frames.append(f)
            for c in labs:
                cls_rows.append({"model": m, "seed": s, "class": c,
                                 "f1_clean": 100 * f1_score(a["gold"] == c, a["pred"] == c),
                                 "f1_diac100": 100 * f1_score(b["gold"] == c, b["pred"] == c)})
        df = pd.concat(frames).merge(feats, on="pos", validate="many_to_one")
        x = df[df["ok_clean"]].copy()
        x["flip"] = (~x["ok_diac"]).astype(int)
        fit = smf.logit("flip ~ n_amb_emo + n_emo_clear + n_syl + mean_confus_other", x).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": x["pos"].to_numpy()})
        ci = fit.conf_int()
        for v in ["n_amb_emo", "n_emo_clear", "n_syl", "mean_confus_other"]:
            res.append({"model": m, "term": v, "odds_ratio": round(float(np.exp(fit.params[v])), 3),
                        "or_ci_lo": round(float(np.exp(ci.loc[v, 0])), 3), "or_ci_hi": round(float(np.exp(ci.loc[v, 1])), 3),
                        "p": float(fit.pvalues[v]), "n_pairs": len(x), "flip_rate": round(x["flip"].mean(), 3)})
    res = pd.DataFrame(res)
    res.to_csv(d / "flip_logit.csv", index=False)
    cr = pd.DataFrame(cls_rows).groupby(["model", "class"])[["f1_clean", "f1_diac100"]].mean().reset_index()
    cr["drop"] = cr["f1_clean"] - cr["f1_diac100"]
    cr["class_emo_confus"] = cr["class"].map(cls_conf)
    cr.round(3).to_csv(d / "class_drop.csv", index=False)
    print(res[res.term.isin(["n_amb_emo", "n_emo_clear"])].to_string(index=False))
    print("\nSpearman (độ mơ hồ từ cảm xúc của lớp ~ mức giảm F1 của lớp), 7 lớp:")
    for m, g in cr.groupby("model", sort=False):
        r, pv = spearmanr(g["class_emo_confus"], g["drop"])
        print(f"  {m:9s} rho = {r:+.2f} (p = {pv:.2f})")


def ps_errors(cfg, d):
    """Model không dấu (PS, train trên văn bản đã bỏ hết dấu) phải tự phân giải "dau" = đau/đâu nên phải nhạy với
    độ mơ hồ khôi phục (confus). Trên test SẠCH: câu P0 đúng mà PS sai có chứa nhiều từ cảm xúc mơ hồ hơn không?
    Logit: ps_wrong ~ n_amb_emo + n_emo_clear + n_syl + mean_confus_other, chỉ trên (câu, seed) mà P0 đúng, cụm theo câu."""
    import statsmodels.formula.api as smf
    feats = sentence_features(cfg, d)
    feats["n_emo_clear"] = feats["n_emo"] - feats["n_amb_emo"]
    R = p(cfg, "results_dir")
    test_ids = load_vsmec(cfg, "test")["id"].astype(str).tolist()
    res = []
    for m, seeds in MODELS.items():
        frames = []
        for s in seeds:
            a = read_pred(R / "rq1" / f"{m}_P0_none_s{s}" / "pred_test.csv", test_ids)
            b = read_pred(R / "strip" / f"{m}_PS_none_s{s}" / "pred_test.csv", test_ids)
            frames.append(pd.DataFrame({"pos": a["pos"], "ok_p0": (a["gold"] == a["pred"]).values,
                                        "ps_wrong": (b["gold"] != b["pred"]).astype(int).values}))
        x = pd.concat(frames).merge(feats, on="pos", validate="many_to_one")
        x = x[x["ok_p0"]]
        fit = smf.logit("ps_wrong ~ n_amb_emo + n_emo_clear + n_syl + mean_confus_other", x).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": x["pos"].to_numpy()})
        ci = fit.conf_int()
        for v in ["n_amb_emo", "n_emo_clear"]:
            res.append({"model": m, "term": v, "odds_ratio": round(float(np.exp(fit.params[v])), 3),
                        "or_ci_lo": round(float(np.exp(ci.loc[v, 0])), 3), "or_ci_hi": round(float(np.exp(ci.loc[v, 1])), 3),
                        "p": float(fit.pvalues[v]), "n_pairs": len(x), "ps_wrong_rate": round(x["ps_wrong"].mean(), 3)})
    res = pd.DataFrame(res)
    res.to_csv(d / "ps_errors_logit.csv", index=False)
    print(res.to_string(index=False))


def _junctions(seg):
    """Chuỗi đã tách từ ("chiến_thắng là") -> (số âm tiết, danh sách nhãn nối: 1 = cùng từ, 0 = khác từ)."""
    out = []
    for tok in seg.split():
        parts = tok.split("_")
        out += [1] * (len(parts) - 1) + [0]
    return out[:-1]


def seg_change(cfg, d):
    """VnCoreNLP tách từ khác đi bao nhiêu khi bỏ hết dấu, và điều đó có giải thích lỗi RIÊNG của PhoBERT không.
    Đối chứng giả: các model không dùng VnCoreNLP (XLM-R, ViSoBERT, CafeBERT, LR, SVM) không được bị ảnh hưởng."""
    import statsmodels.formula.api as smf
    from .preprocess import get_segmenter
    nz = pd.read_csv(p(cfg, "results_dir") / "noise" / "test_diac100.csv", dtype={"id": str})
    test_ids = load_vsmec(cfg, "test")["id"].astype(str).tolist()
    assert nz["id"].tolist() == test_ids
    seg = get_segmenter(cfg)
    clean = [base_clean(t) for t in nz["text_clean"].fillna("")]
    strip = [base_clean(t) for t in nz["text"].fillna("")]
    sc, ss = seg.segment(clean), seg.segment(strip)
    rows = []
    for pos, (a, b) in enumerate(zip(sc, ss)):
        ja, jb = _junctions(a), _junctions(b)
        if len(ja) != len(jb) or not ja:
            rows.append({"pos": pos, "seg_ok": False, "n_junc": 0, "n_changed": 0})
            continue
        rows.append({"pos": pos, "seg_ok": True, "n_junc": len(ja), "n_changed": int(sum(x != y for x, y in zip(ja, jb))),
                     "n_multi_clean": a.count("_")})
    sg = pd.DataFrame(rows)
    sg["seg_changed"] = (sg["n_changed"] > 0).astype(int)
    ok = sg[sg.seg_ok]
    print(f"[seg] so khớp được {len(ok)}/{len(sg)} câu; câu bị đổi cách tách từ: {ok.seg_changed.mean():.1%}; "
          f"tỉ lệ chỗ nối bị đổi: {ok.n_changed.sum() / ok.n_junc.sum():.1%}")
    feats = sentence_features(cfg, d)
    R = p(cfg, "results_dir") / "rq1"
    res = []
    for m, seeds in MODELS.items():
        frames = []
        for s in seeds:
            a = read_pred(R / f"{m}_P0_none_s{s}" / "pred_test.csv", test_ids)
            b = read_pred(R / f"{m}_P0_none_s{s}" / "pred_diac100.csv", test_ids)
            frames.append(pd.DataFrame({"pos": a["pos"], "ok_clean": (a["gold"] == a["pred"]).values,
                                        "flip": (b["gold"] != b["pred"]).astype(int).values}))
        x = pd.concat(frames).merge(feats, on="pos", validate="many_to_one").merge(sg[sg.seg_ok], on="pos")
        x = x[x["ok_clean"]]
        fit = smf.logit("flip ~ seg_changed + n_amb_emo + n_syl + mean_confus_other", x).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": x["pos"].to_numpy()})
        ci = fit.conf_int()
        res.append({"model": m, "uses_vncorenlp": m == "phobert",
                    "or_seg_changed": round(float(np.exp(fit.params["seg_changed"])), 3),
                    "ci_lo": round(float(np.exp(ci.loc["seg_changed", 0])), 3),
                    "ci_hi": round(float(np.exp(ci.loc["seg_changed", 1])), 3), "p": float(fit.pvalues["seg_changed"]),
                    "n_pairs": len(x)})
    sg.to_csv(d / "seg_change.csv", index=False)
    res = pd.DataFrame(res)
    res.to_csv(d / "seg_change_logit.csv", index=False)
    print(res.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["all", "errors", "ps", "seg"])
    ap.add_argument("--config", default="configs/base.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    d = out_dir(cfg)
    if a.step == "errors":
        return errors(cfg, d)
    if a.step == "ps":
        return ps_errors(cfg, d)
    if a.step == "seg":
        return seg_change(cfg, d)

    lex, keys = build_lexicon(cfg)
    lex.to_csv(d / "lexicon.csv", index=False, encoding="utf-8")
    keys.to_csv(d / "keys.csv", index=False, encoding="utf-8")
    tok = lex["count"].sum()
    w_conf = (lex["count"] * lex["confus"]).sum() / tok
    print(f"[lexicon] {len(lex):,} dạng có dấu, {len(keys):,} dạng không dấu; token-weighted confus = {w_conf:.3f}; "
          f"dạng không dấu có >=2 dạng: {(keys.n_forms >= 2).mean():.1%} (chiếm {keys[keys.n_forms >= 2].total.sum() / tok:.1%} token)")

    emo = emotion_words(cfg)
    emo.to_csv(d / "emotion_words.csv", index=False, encoding="utf-8")

    j = emo.merge(lex, on="word", how="left")
    j["in_lexicon"] = j["count"].notna()
    j.to_csv(d / "emotion_words_with_ambiguity.csv", index=False, encoding="utf-8")
    amb = j[j["confus"] >= CONFUS_MIN].sort_values(["class", "coef"], ascending=[True, False])
    amb[["class", "rank", "word", "coef", "key", "count", "p_given_key", "confus", "key_entropy", "competitors"]] \
        .to_csv(d / "ambiguous_emotion_words.csv", index=False, encoding="utf-8")

    other = lex[~lex["word"].isin(emo["word"])]
    s = j[j.in_lexicon]
    print(f"[emotion] {len(emo)} từ cảm xúc ({emo['class'].nunique()} lớp); có trong lexicon: {s.shape[0]}")
    print(f"  confus TB (theo token): từ cảm xúc = {np.average(s.confus, weights=s['count']):.3f}; "
          f"từ khác = {np.average(other.confus, weights=other['count']):.3f}")
    print(f"  từ cảm xúc có confus >= {CONFUS_MIN}: {len(amb)}")
    for c, g in j[j.in_lexicon].groupby("class"):
        print(f"  {c:10s} confus TB = {np.average(g.confus, weights=g['count']):.3f}  "
              f"mơ hồ(>= {CONFUS_MIN}) = {(g.confus >= CONFUS_MIN).sum():2d}/{len(g)}")


if __name__ == "__main__":
    main()
