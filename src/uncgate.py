"""H5 (review/07_thiet_ke_bat_dinh.md): gating THEO BẤT ĐỊNH, so với gating theo tỉ lệ dấu.

  p = (1 − g)·p_std + g·p_free,  g = σ((u − τ)/s),  u = bất định của model thường (MSP hoặc entropy, cùng seed).
  Tín hiệu, τ, s chọn trên VALID (sạch + bỏ dấu 25/50/75/100%, trọng số đều), riêng cho từng PLM, dùng chung mọi seed.
  Test chấm MỘT lần.

p_free = model không dấu (PS, results/strip). PS bỏ hết dấu ở đầu vào nên dự đoán của nó KHÔNG phụ thuộc mức bỏ dấu
(đã kiểm: pred_diac* trùng pred_test); vì vậy với valid nhiễu dùng thẳng pred_valid.csv của PS.

Bước:
  prep : sinh valid bỏ dấu 25/50/75/100% (cùng bộ sinh nhiễu với test, khóa "valid_<tập>|<id>") và chấm bằng checkpoint
         RQ1 P0 (GPU laptop) -> results/uncertainty/valid_noisy/<run>/pred_<tập>.csv
  run  : chọn tham số trên valid, chấm test (sạch, bỏ dấu 25..100%, ViGoEmotions mới) và kiểm định
         -> results/uncertainty/h5_*.csv
Chạy: python -m src.uncgate prep | run
"""
import argparse
import itertools

import numpy as np
import pandas as pd

from .ambiguity import MODELS
from .data import add_groups, load_vigo, load_vsmec
from .gate_fusion import _probs, diac_ratio, gate_weights
from .stats import hier_bootstrap, holm, macro_f1_batch
from .uncertainty import EPS, N_BOOT, PLMS, out_dir
from .utils import compute_metrics, labels, lenient_targets, load_config, p

LEVELS = [0.25, 0.5, 0.75, 1.0]
VSETS = ["valid", "valid_diac25", "valid_diac50", "valid_diac75", "valid_diac100"]
TSETS = ["test", "diac25", "diac50", "diac75", "diac100"]


def vdir(cfg):
    d = out_dir(cfg) / "valid_noisy"
    d.mkdir(parents=True, exist_ok=True)
    return d


def prep(cfg):
    from .noise import Noiser
    from .preprocess import base_clean
    v = load_vsmec(cfg, "valid")
    clean = [base_clean(t, cfg["preprocess"].get("lowercase", True)) for t in v["text"]]
    nz = Noiser(cfg)
    sets = {}
    for lv in LEVELS:
        name = f"valid_diac{int(lv * 100)}"
        rows = [{"id": str(r["id"]), "label": r["label"], "y": r["y"],
                 "text": nz.apply(c, "diac", lv, f"{name}|{r['id']}|{pos}")[0]} for pos, ((_, r), c) in enumerate(zip(v.iterrows(), clean))]
        df = pd.DataFrame(rows)
        df.to_csv(vdir(cfg) / f"{name}.csv", index=False, encoding="utf-8")
        sets[name] = {"df": df, "dictnorm": False, "kind": "single"}
        print(f"[prep] {name}: câu bị đổi {sum(a != b for a, b in zip(df.text, clean))}/{len(df)}")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .train_plm import evaluate_and_save
    from .utils import get_device
    device = get_device(cfg)
    R = p(cfg, "results_dir") / "rq1"
    for m in PLMS:
        for s in MODELS[m]:
            run = f"{m}_P0_none_s{s}"
            dd = vdir(cfg) / run
            dd.mkdir(exist_ok=True)
            todo = {k: x for k, x in sets.items() if not (dd / f"pred_{k}.csv").exists()}
            if not todo:
                continue
            ck = R / run / "checkpoint"
            tok = AutoTokenizer.from_pretrained(ck)
            model = AutoModelForSequenceClassification.from_pretrained(ck, dtype=torch.float32).to(device)
            mcfg = cfg["models"][m]
            ccfg = {**cfg, "train": {**cfg["train"], **mcfg.get("train", {})}}
            evaluate_and_save(model, tok, todo, "P0", ccfg, device, False, mcfg.get("segment", False), dd)
            del model
            torch.cuda.empty_cache()
            print(f"[prep] {run}: {len(todo)} tập")


# ----------------------------------------------------------------------------- dữ liệu cho gating
def _read(f, labs, ids=None):
    x = pd.read_csv(f, dtype={"id": str})
    if ids is not None:
        assert x["id"].tolist() == ids, f
    return x


def system_probs(cfg, labs, m, split):
    """Trả về dict tập -> (A: (S,N,K) model thường, B: (S,N,K) PS, texts cho w, gold (N,) hoặc None)."""
    R = p(cfg, "results_dir")
    cols = [f"s_{l}" for l in labs]
    out = {}
    names = VSETS if split == "valid" else TSETS + ["vigo"]
    for name in names:
        A, B, ids = [], [], None
        for s in MODELS[m]:
            if split == "valid":
                fa = (R / "rq1" / f"{m}_P0_none_s{s}" / "pred_valid.csv" if name == "valid"
                      else vdir(cfg) / f"{m}_P0_none_s{s}" / f"pred_{name}.csv")
                fb = R / "strip" / f"{m}_PS_none_s{s}" / "pred_valid.csv"
            else:
                fn = "pred_vigo_new_test.csv" if name == "vigo" else f"pred_{name}.csv"
                fa = R / "rq1" / f"{m}_P0_none_s{s}" / fn
                fb = R / "strip" / f"{m}_PS_none_s{s}" / fn
            a = _read(fa, labs, ids)
            ids = a["id"].tolist()
            b = _read(fb, labs, ids)
            A.append(_probs(a[cols].to_numpy(float), m))
            B.append(_probs(b[cols].to_numpy(float), m))
        out[name] = (np.stack(A), np.stack(B), ids, a)
    return out


def texts_for(cfg, split, name):
    from .preprocess import base_clean
    if split == "valid":
        if name == "valid":
            return [base_clean(t) for t in load_vsmec(cfg, "valid")["text"]]
        return pd.read_csv(vdir(cfg) / f"{name}.csv")["text"].fillna("").tolist()
    if name == "test":
        return [base_clean(t) for t in load_vsmec(cfg, "test")["text"]]
    if name == "vigo":
        return load_vigo(cfg, "new", "test")["text"].astype(str).tolist()
    return pd.read_csv(p(cfg, "results_dir") / "noise" / f"test_{name}.csv")["text"].fillna("").tolist()


def unc(P, kind):
    Q = np.clip(P, EPS, None)
    return 1 - Q.max(-1) if kind == "MSP" else -(Q * np.log(Q)).sum(-1)


def uncgate(A, B, kind, tau, s):
    g = 1 / (1 + np.exp(-(unc(A, kind) - tau) / s))
    return (1 - g)[..., None] * A + g[..., None] * B


def mf1_mean(gold, P, K):
    pred = P.argmax(-1)
    return macro_f1_batch(np.broadcast_to(gold, pred.shape), pred, K).mean()


# ----------------------------------------------------------------------------- run
def run(cfg):
    labs = labels(cfg)
    K = len(labs)
    r_train = np.nanmean([diac_ratio(t) for t in load_vsmec(cfg, "train")["text"]])
    gv = np.array(load_vsmec(cfg, "valid")["y"])
    gt = np.array(load_vsmec(cfg, "test")["y"])
    vt = add_groups(load_vigo(cfg, "new", "test"), cfg, "a")
    keep = (vt["groups"].map(len) > 0).to_numpy()
    wv = {n: gate_weights(texts_for(cfg, "valid", n), r_train) for n in VSETS}
    wt = {n: gate_weights(texts_for(cfg, "test", n), r_train) for n in TSETS + ["vigo"]}
    low = keep & (wt["vigo"] < 0.5)
    sel, preds = [], {}
    for m in PLMS:
        V = system_probs(cfg, labs, m, "valid")
        allu = {k: np.concatenate([unc(V[n][0], k).ravel() for n in VSETS]) for k in ["MSP", "entropy"]}
        best = None
        for kind in ["MSP", "entropy"]:
            taus = np.quantile(allu[kind], np.linspace(0.05, 0.95, 19))
            rng_ = allu[kind].max() - allu[kind].min()
            for tau, sf in itertools.product(taus, [0.01, 0.03, 0.1, 0.3]):
                sc = np.mean([mf1_mean(gv, uncgate(V[n][0], V[n][1], kind, tau, sf * rng_), K) for n in VSETS])
                if best is None or sc > best[0]:
                    best = (sc, kind, tau, sf * rng_)
        ratio_v = np.mean([mf1_mean(gv, wv[n][:, None] * V[n][0] + (1 - wv[n][:, None]) * V[n][1], K) for n in VSETS])
        std_v = np.mean([mf1_mean(gv, V[n][0], K) for n in VSETS])
        sel.append({"model": m, "signal": best[1], "tau": best[2], "s": best[3], "valid_mf1_uncgate": best[0],
                    "valid_mf1_ratiogate": ratio_v, "valid_mf1_standard": std_v})
        T = system_probs(cfg, labs, m, "test")
        for n, (A, B, ids, _) in T.items():
            w = wt[n][:, None]
            preds[(m, n)] = {"standard": A.argmax(-1), "PS": B.argmax(-1), "avg": (0.5 * A + 0.5 * B).argmax(-1),
                             "ratiogate": (w * A + (1 - w) * B).argmax(-1),
                             "uncgate": uncgate(A, B, best[1], best[2], best[3]).argmax(-1)}
    sel = pd.DataFrame(sel)
    # --------- MF1 trên test; ViGo chấm lenient (gold hiệu dụng phụ thuộc dự đoán)
    def eff_gold(pred_row, mask):
        sub = vt[mask]
        e = lenient_targets([labs[i] for i in pred_row[mask]], sub["groups"].map(set).tolist(), sub["votes"].tolist(), labs)
        return np.array([labs.index(x) for x in e])

    def score(m, n, sysn, mask=None):
        Pd = preds[(m, n)][sysn]
        if n != "vigo":
            return np.mean([compute_metrics(gt, r, K)["mf1"] for r in Pd])
        return np.mean([compute_metrics(eff_gold(r, mask), r[mask], K)["mf1"] for r in Pd])

    rows = []
    for m in PLMS:
        for n in TSETS + ["vigo", "vigo_flagged"]:
            nn, mask = ("vigo", keep) if n == "vigo" else (("vigo", low) if n == "vigo_flagged" else (n, None))
            rows.append({"model": m, "set": n, **{k: score(m, nn, k, mask) for k in preds[(m, nn)]}})
    res = pd.DataFrame(rows)
    # --------- kiểm định: uncgate vs ratiogate, uncgate vs standard; 4 PLM x 6 tập (5 test + ViGo toàn tập) = 48
    rng = np.random.default_rng(0)
    tests = []
    for m in PLMS:
        for n in TSETS + ["vigo"]:
            for other in ["ratiogate", "standard"]:
                A, B = preds[(m, n)]["uncgate"], preds[(m, n)][other]
                if n != "vigo":
                    d, pv, lo, hi = hier_bootstrap(gt, A, B, K, N_BOOT, 0)
                else:   # lenient: gold hiệu dụng riêng cho từng hệ/seed; bootstrap câu + seed
                    idx_keep = np.where(keep)[0]
                    GA = np.stack([eff_gold(r, keep) for r in A])
                    GB = np.stack([eff_gold(r, keep) for r in B])
                    a_, b_ = A[:, keep], B[:, keep]
                    f = lambda G, Pp: macro_f1_batch(G, Pp, K)
                    d = f(GA, a_).mean() - f(GB, b_).mean()
                    bs = []
                    N = len(idx_keep)
                    for _ in range(N_BOOT // 5):
                        ix = rng.integers(0, N, N)
                        sa, sb = rng.integers(0, len(A), len(A)), rng.integers(0, len(B), len(B))
                        bs.append(f(GA[sa][:, ix], a_[sa][:, ix]).mean() - f(GB[sb][:, ix], b_[sb][:, ix]).mean())
                    bs = np.array(bs)
                    pv = (1 + np.sum(np.abs(bs - d) >= abs(d))) / (len(bs) + 1)
                    lo, hi = np.percentile(bs, [2.5, 97.5])
                tests.append({"model": m, "set": n, "vs": other, "delta": d, "ci_lo": lo, "ci_hi": hi, "p": pv})
    tt = pd.DataFrame(tests)
    tt["p_holm"] = holm(tt["p"].to_numpy())
    tt["sig"] = tt["p_holm"] < 0.05
    d = out_dir(cfg)
    sel.round(5).to_csv(d / "h5_selection.csv", index=False)
    res.round(3).to_csv(d / "h5_mf1.csv", index=False)
    tt.round(4).to_csv(d / "h5_tests.csv", index=False)
    print("Chọn trên valid:\n", sel.round(4).to_string(index=False))
    print("\nMF1 trên test (ViGo: lenient; vigo_flagged = 70 câu w<0.5):")
    print(res.round(2).to_string(index=False))
    print("\nKiểm định (delta = uncgate − đối chứng), Holm 48:")
    print(tt.pivot_table(index=["model", "set"], columns="vs", values="delta").round(2).to_string())
    print(tt[tt.sig].round(3).to_string(index=False))
    # tiêu chí chốt trước
    c1 = not ((tt.vs == "standard") & (tt.set == "test") & tt.sig & (tt.delta < 0)).any()
    c2 = not ((tt.vs == "ratiogate") & tt.set.isin(["diac75", "diac100"]) & tt.sig & (tt.delta < 0)).any()
    fl = res[res.set == "vigo_flagged"].set_index("model")
    vg = res[res.set == "vigo"].set_index("model")
    c3 = bool((fl.uncgate >= fl.standard - 1e-9).all() and (vg.uncgate >= vg.standard - 1e-9).all())
    print(f"\nTiêu chí: (1) không thua model thường trên câu sạch: {c1}; (2) không thua gating tỉ lệ dấu ở 75–100%: {c2}; "
          f"(3) ViGo: nhóm 70 câu và toàn tập >= model thường (mọi PLM): {c3}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "run"])
    a = ap.parse_args()
    globals()[a.step](load_config("configs/base.yaml"))


if __name__ == "__main__":
    main()
