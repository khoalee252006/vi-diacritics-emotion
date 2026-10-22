"""Phần 1, thí nghiệm can thiệp: bỏ dấu CÓ CHỌN LỌC trên test UIT-VSMEC rồi chấm bằng model RQ1 P0 đã train
(không train thêm). Mục tiêu: tách tác động của "từ cảm xúc" và "độ mơ hồ khi bỏ dấu" ra khỏi số lượng từ bị bỏ dấu.

Hai khái niệm mơ hồ (lấy từ results/ambiguity/lexicon.csv, chỉ tính trên tập train):
  confus(w)    = 1 - p(w | dạng không dấu k)  : khó khôi phục w từ k (quan trọng với model không dấu, LLM, bộ khôi phục).
  collision(w) = p(k | k)                        : bản thân dạng không dấu k là một từ viết như vậy và hay gặp
                                                   (vd. mày -> "may"); model train trên văn bản có dấu sẽ đọc thành từ khác.

Thiết kế 1 `single` (chính): mỗi biến thể chỉ bỏ dấu MỘT âm tiết (có dấu, có trong lexicon) của một câu test.
  Nhân tố 2x2: emo (âm tiết thuộc 350 từ cảm xúc) x amb (confus >= 0.2). Đo dp = p_sạch(gold) - p_biến_thể(gold)
  và tỉ lệ đổi đúng -> sai. Hồi quy dp ~ emo*amb + collision + n_syl + rel_pos, sai số chuẩn cụm theo câu.
Thiết kế 2 `multi` (phụ, cấp câu, đo MF1): trên các câu có k >= 1 từ cảm xúc mơ hồ có dấu,
  amb_emo   = bỏ dấu đúng k từ đó;
  rand_r    = bỏ dấu k âm tiết có dấu KHÔNG phải từ cảm xúc, chọn ngẫu nhiên (5 lần rút);
  ambother_r= bỏ dấu k âm tiết có dấu không phải từ cảm xúc nhưng mơ hồ (confus >= 0.2) (5 lần rút).
  Cùng tập câu, cùng số âm tiết bị bỏ dấu ở mọi điều kiện.

Lưu ý diễn giải: 350 từ cảm xúc lấy từ hệ số TF-IDF+LR, nên với chính LR kết quả "emo" có phần do cách chọn từ
(vòng lặp); bằng chứng chính là các PLM.

Chạy: python -m src.intervention build | infer | analyze      (infer cần GPU; PhoBERT cần Java cho VnCoreNLP)
"""
import argparse

import numpy as np
import pandas as pd

from .ambiguity import CONFUS_MIN, MODELS, SYL, out_dir
from .data import load_vsmec
from .noise import strip_diacritics
from .preprocess import base_clean, prepare
from .utils import compute_metrics, get_device, labels, load_config, p, save_predictions

N_DRAWS = 5


def _resources(d):
    lex = pd.read_csv(d / "lexicon.csv", keep_default_na=False)
    confus = lex.set_index("word")["confus"].to_dict()
    collision = lex[lex["word"] == lex["key"]].set_index("key")["p_given_key"].to_dict()
    emo = pd.read_csv(d / "emotion_words_with_ambiguity.csv").query("`class` != \"Other\"")  # Other không mang cảm xúc
    return confus, collision, set(emo["word"]), set(emo.loc[emo["confus"] >= CONFUS_MIN, "word"])


def _strip_at(s, spans):
    out, last = [], 0
    for a, b in sorted(spans):
        out += [s[last:a], strip_diacritics(s[a:b])]
        last = b
    return "".join(out + [s[last:]])


def build(cfg):
    d = out_dir(cfg)
    confus, collision, emo_all, amb_emo = _resources(d)
    t = load_vsmec(cfg, "test")
    single, multi_rows, sets = [], [], {"clean": [], "amb_emo": []}
    for r in range(N_DRAWS):
        sets[f"rand_{r}"], sets[f"ambother_{r}"] = [], []
    rng = np.random.default_rng(0)
    for pos, (sid, txt, lab, y) in enumerate(zip(t["id"].astype(str), t["text"], t["label"], t["y"])):
        s = base_clean(txt)
        occ = [(m.start(), m.end(), m.group()) for m in SYL.finditer(s)]
        n = len(occ)
        acc = [i for i, (_, _, w) in enumerate(occ) if strip_diacritics(w) != w]
        for i in acc:
            a, b, w = occ[i]
            if w not in confus:
                continue
            k = strip_diacritics(w)
            single.append({"id": f"{sid}_{i}", "sent_id": sid, "sent_pos": pos, "text": _strip_at(s, [(a, b)]), "label": lab, "y": y,
                           "word": w, "emo": int(w in emo_all), "amb": int(confus[w] >= CONFUS_MIN),
                           "confus": confus[w], "collision": collision.get(k, 0.0), "n_syl": n,
                           "rel_pos": i / max(n - 1, 1)})
        ae = [i for i in acc if occ[i][2] in amb_emo]
        rn = [i for i in acc if occ[i][2] not in emo_all]
        an = [i for i in rn if confus.get(occ[i][2], -1) >= CONFUS_MIN]
        k = len(ae)
        ok = k >= 1 and len(rn) >= k and len(an) >= k
        multi_rows.append({"sent_id": sid, "sent_pos": pos, "k": k, "n_rand_pool": len(rn), "n_ambother_pool": len(an), "in_subset": ok})
        if not ok:
            continue
        base = {"id": sid, "label": lab, "y": y}
        sets["clean"].append({**base, "text": s})
        sets["amb_emo"].append({**base, "text": _strip_at(s, [occ[i][:2] for i in ae])})
        for r in range(N_DRAWS):
            sets[f"rand_{r}"].append({**base, "text": _strip_at(s, [occ[i][:2] for i in rng.choice(rn, k, replace=False)])})
            sets[f"ambother_{r}"].append({**base, "text": _strip_at(s, [occ[i][:2] for i in rng.choice(an, k, replace=False)])})
    od = d / "intervention"
    od.mkdir(exist_ok=True)
    sg = pd.DataFrame(single)
    sg.to_csv(od / "single_instances.csv", index=False, encoding="utf-8")
    pd.DataFrame(multi_rows).to_csv(od / "multi_subset.csv", index=False)
    for name, rows in sets.items():
        pd.DataFrame(rows).to_csv(od / f"multi_{name}.csv", index=False, encoding="utf-8")
    m = pd.DataFrame(multi_rows)
    print(f"[single] {len(sg):,} biến thể từ {sg.sent_id.nunique()} câu; ô 2x2 (emo, amb):")
    print(sg.groupby(["emo", "amb"]).size().rename("n").to_string())
    print(f"[multi] tập con {int(m.in_subset.sum())}/{len(m)} câu; k TB = {m[m.in_subset].k.mean():.2f}; "
          f"câu có k>=1 nhưng thiếu âm tiết đối chứng: {int(((m.k >= 1) & ~m.in_subset).sum())}")


def _eval_sets(cfg, od):
    sets = {"single": pd.read_csv(od / "single_instances.csv", dtype={"id": str, "sent_id": str})}
    for f in sorted(od.glob("multi_*.csv")):
        if f.stem != "multi_subset":
            sets[f.stem] = pd.read_csv(f, dtype={"id": str})
    sets["single_clean"] = load_vsmec(cfg, "test").assign(
        id=lambda x: x["id"].astype(str), text=lambda x: x["text"].map(base_clean))
    return {k: {"df": v[["id", "text", "label", "y"]], "dictnorm": False, "kind": "single"} for k, v in sets.items()}


def infer(cfg):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .baseline_tfidf import load_model
    from .train_plm import evaluate_and_save
    od = out_dir(cfg) / "intervention"
    sets = _eval_sets(cfg, od)
    labs = labels(cfg)
    R = p(cfg, "results_dir") / "rq1"
    device = get_device(cfg)
    for m, seeds in MODELS.items():
        for s in seeds:
            run = f"{m}_P0_none_s{s}"
            dd = od / run
            dd.mkdir(exist_ok=True)
            todo = {k: v for k, v in sets.items() if not (dd / f"pred_{k}.csv").exists()}
            if not todo:
                continue
            if m in ("lr", "svm"):
                model = load_model(R / run / "model.joblib")
                for name, es in todo.items():
                    sc = model.scores(prepare(es["df"]["text"].tolist(), "P0", cfg))
                    save_predictions(dd / f"pred_{name}.csv", es["df"]["id"].tolist(), es["df"]["label"].tolist(),
                                     sc.argmax(1), sc, labs)
            else:
                ck = R / run / "checkpoint"
                tok = AutoTokenizer.from_pretrained(ck)
                model = AutoModelForSequenceClassification.from_pretrained(ck, torch_dtype=torch.float32).to(device)
                mcfg = cfg["models"][m]
                ccfg = {**cfg, "train": {**cfg["train"], **mcfg.get("train", {})}}
                evaluate_and_save(model, tok, todo, "P0", ccfg, device, False, mcfg.get("segment", False), dd)
                del model
                torch.cuda.empty_cache()
            print(f"[infer] {run}: {len(todo)} tập")


def _load(od, run, name):
    return pd.read_csv(od / run / f"pred_{name}.csv", dtype={"id": str})


def analyze(cfg):
    import statsmodels.formula.api as smf
    from .stats import hier_bootstrap, holm
    od = out_dir(cfg) / "intervention"
    labs = labels(cfg)
    K = len(labs)
    inst = pd.read_csv(od / "single_instances.csv", dtype={"id": str, "sent_id": str})
    test_ids = load_vsmec(cfg, "test")["id"].astype(str).tolist()
    # ---------------- thiết kế 1: một âm tiết
    rows, cells = [], []
    for m, seeds in MODELS.items():
        if m == "svm":          # điểm SVM không phải xác suất -> không tính dp
            continue
        dp, flip = [], []
        for s in seeds:
            run = f"{m}_P0_none_s{s}"
            # Ghép theo VỊ TRÍ dòng: id của test VSMEC bị trùng (693 dòng, 608 id).
            c = _load(od, run, "single_clean")
            v = _load(od, run, "single")
            assert c["id"].tolist() == test_ids and v["id"].tolist() == inst["id"].tolist()
            pc = c.iloc[inst["sent_pos"].to_numpy()].reset_index(drop=True)
            gi = [labs.index(g) for g in inst["label"]]
            P_clean = pc[[f"s_{l}" for l in labs]].to_numpy()[np.arange(len(inst)), gi]
            P_var = v[[f"s_{l}" for l in labs]].to_numpy()[np.arange(len(inst)), gi]
            dp.append(P_clean - P_var)
            ok_c = (pc["pred"].to_numpy() == inst["label"].to_numpy())
            ok_v = (v["pred"].to_numpy() == inst["label"].to_numpy())
            flip.append(np.where(ok_c, (~ok_v).astype(float), np.nan))
        x = inst.copy()
        x["dp"] = np.mean(dp, 0) * 100
        x["flip"] = np.nanmean(np.vstack(flip), 0)
        g = x.groupby(["emo", "amb"]).agg(n=("dp", "size"), dp_mean=("dp", "mean"), flip_rate=("flip", "mean"))
        for (e, a), r in g.iterrows():
            cells.append({"model": m, "emo": e, "amb": a, "n": int(r.n), "dp_mean": round(r.dp_mean, 3),
                          "flip_rate": round(r.flip_rate, 4)})
        fit = smf.ols("dp ~ emo * amb + collision + n_syl + rel_pos", x).fit(
            cov_type="cluster", cov_kwds={"groups": x["sent_pos"].to_numpy()})
        ci = fit.conf_int()
        for term in ["emo", "amb", "emo:amb", "collision"]:
            rows.append({"model": m, "term": term, "coef_pp": round(fit.params[term], 3),
                         "ci_lo": round(ci.loc[term, 0], 3), "ci_hi": round(ci.loc[term, 1], 3), "p": fit.pvalues[term]})
    reg = pd.DataFrame(rows)
    reg["p_holm"] = holm(reg["p"].to_numpy())
    reg.to_csv(od / "single_regression.csv", index=False)
    pd.DataFrame(cells).to_csv(od / "single_cells.csv", index=False)
    print("[single] mức giảm p(gold) (điểm %) theo ô 2x2, trung bình theo seed:")
    print(pd.DataFrame(cells).pivot_table(index="model", columns=["emo", "amb"], values="dp_mean").round(2).to_string())
    print("\n[single] hồi quy dp (điểm %), cụm theo câu:")
    print(reg.round(4).to_string(index=False))
    # ---------------- thiết kế 2: cấp câu
    out = []
    for m, seeds in MODELS.items():
        runs = [f"{m}_P0_none_s{s}" for s in seeds]
        gold = np.array([labs.index(g) for g in _load(od, runs[0], "multi_clean")["gold"]])
        P = lambda name: np.array([[labs.index(q) for q in _load(od, r, name)["pred"]] for r in runs])
        A = P("multi_amb_emo")
        B1 = np.vstack([P(f"multi_rand_{i}") for i in range(N_DRAWS)])
        B2 = np.vstack([P(f"multi_ambother_{i}") for i in range(N_DRAWS)])
        C = P("multi_clean")
        mf = lambda M: np.mean([compute_metrics(gold, row, K)["mf1"] for row in M])
        rec = {"model": m, "n_sent": len(gold), "clean": mf(C), "amb_emo": mf(A), "rand": mf(B1), "ambother": mf(B2)}
        for nm, B in [("rand", B1), ("ambother", B2)]:
            dlt, pv, lo, hi = hier_bootstrap(gold, A, B, K, 10000, 0)
            rec.update({f"d_vs_{nm}": dlt, f"p_vs_{nm}": pv, f"lo_vs_{nm}": lo, f"hi_vs_{nm}": hi})
        out.append(rec)
    mt = pd.DataFrame(out)
    adj = holm(np.r_[mt["p_vs_rand"], mt["p_vs_ambother"]])
    mt["p_holm_vs_rand"], mt["p_holm_vs_ambother"] = adj[:len(mt)], adj[len(mt):]
    mt.round(4).to_csv(od / "multi_results.csv", index=False)
    print("\n[multi] MF1 trên tập con (âm = bỏ dấu từ cảm xúc mơ hồ hại hơn đối chứng):")
    print(mt.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "infer", "analyze"])
    ap.add_argument("--config", default="configs/base.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"build": build, "infer": infer, "analyze": analyze}[a.step](cfg)


if __name__ == "__main__":
    main()
