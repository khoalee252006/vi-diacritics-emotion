"""Phần 2: so LLM (results/llm/<model>/responses.jsonl) với các model fine-tune RQ1 P0 trên test sạch, bỏ dấu 50%, 100%.

Các phép so sánh (chốt trước khi xem kết quả, Holm trên TOÀN BỘ):
  với mỗi cấu hình LLM (model x shots) và mỗi PLM: (1) MF1 sạch, (2) MF1 bỏ dấu 100%,
  (3) mức giảm = (sạch - bỏ dấu 100%) của LLM trừ mức giảm của PLM (âm = LLM bền hơn).
Bootstrap hai tầng: lấy mẫu lại câu (dùng chung cho mọi điều kiện) và seed của PLM; LLM chạy 1 lần (temperature 0).
Ghép theo VỊ TRÍ dòng (id test VSMEC bị trùng). INVALID của LLM = -1 (luôn tính là sai).

Chạy: python -m src.llm_compare   -> results/llm/compare_mf1.csv, compare_tests.csv
"""
import json

import numpy as np
import pandas as pd

from .ambiguity import MODELS
from .data import load_vsmec
from .stats import holm, macro_f1_batch
from .utils import labels, load_config, p

SETS = {"clean": "pred_test.csv", "diac50": "pred_diac50.csv", "diac100": "pred_diac100.csv"}
PLMS = ["cafebert", "xlmr", "phobert", "visobert"]
N_BOOT = 10000


def plm_preds(cfg, labs, test_ids):
    R = p(cfg, "results_dir") / "rq1"
    out = {}
    for m, seeds in MODELS.items():
        out[m] = {}
        for s_name, f in SETS.items():
            rows = []
            for s in seeds:
                x = pd.read_csv(R / f"{m}_P0_none_s{s}" / f, dtype={"id": str})
                assert x["id"].tolist() == test_ids
                rows.append([labs.index(v) for v in x["pred"]])
            out[m][s_name] = np.array(rows)
    return out


def llm_preds(cfg, labs, n):
    out = {}
    for d in sorted((p(cfg, "results_dir") / "llm").iterdir()):
        f = d / "responses.jsonl"
        if not f.is_dir() and f.exists():
            r = pd.DataFrame([json.loads(x) for x in open(f, encoding="utf-8")])
            if "pos" not in r:
                print(f"[bỏ qua] {d.name}: bản ghi cũ chưa có pos, chạy lại llm_eval để chuyển đổi")
                continue
            r = r.drop_duplicates(["set", "shots", "pos"])
            for k, g in r.groupby("shots"):
                sets = {}
                for s_name in SETS:
                    gg = g[g["set"] == s_name].sort_values("pos")
                    if len(gg) != n or gg["pos"].tolist() != list(range(n)):
                        break
                    sets[s_name] = np.array([[labs.index(v) if v in labs else -1 for v in gg["pred"]]])
                else:
                    out[f"{d.name} ({k}-shot)"] = sets
                    continue
                print(f"[bỏ qua] {d.name} {k}-shot: chưa đủ {n} câu ở mọi tập")
    return out


def boot(gold, A, B, K, seed=0):
    """Trả về (giá trị quan sát, p hai phía, CI 95%) cho ba đại lượng: clean, diac100, did."""
    rng = np.random.default_rng(seed)
    N = len(gold)

    def stats(idx, ia, ib):
        g = gold[idx]
        fa = {s: macro_f1_batch(np.broadcast_to(g, (len(ia), N)), A[s][ia][:, idx], K).mean() for s in ("clean", "diac100")}
        fb = {s: macro_f1_batch(np.broadcast_to(g, (len(ib), N)), B[s][ib][:, idx], K).mean() for s in ("clean", "diac100")}
        return np.array([fa["clean"] - fb["clean"], fa["diac100"] - fb["diac100"],
                         (fa["clean"] - fa["diac100"]) - (fb["clean"] - fb["diac100"])])

    obs = stats(np.arange(N), np.arange(len(A["clean"])), np.arange(len(B["clean"])))
    bs = np.array([stats(rng.integers(0, N, N), rng.integers(0, len(A["clean"]), len(A["clean"])),
                         rng.integers(0, len(B["clean"]), len(B["clean"]))) for _ in range(N_BOOT)])
    pv = (1 + (np.abs(bs - obs) >= np.abs(obs)).sum(0)) / (N_BOOT + 1)
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return obs, pv, lo, hi


def main():
    cfg = load_config("configs/base.yaml")
    labs = labels(cfg)
    K = len(labs)
    t = load_vsmec(cfg, "test")
    test_ids = t["id"].astype(str).tolist()
    gold = np.array([labs.index(v) for v in t["label"]])
    plm = plm_preds(cfg, labs, test_ids)
    llm = llm_preds(cfg, labs, len(gold))
    rows = []
    for name, sets in [*plm.items(), *llm.items()]:
        rec = {"system": name, "kind": "LLM" if name in llm else "fine-tuned", "n_runs": len(sets["clean"])}
        for s_name, M in sets.items():
            v = macro_f1_batch(np.broadcast_to(gold, M.shape), M, K)
            rec[s_name] = round(v.mean(), 2)
            rec[f"{s_name}_std"] = round(v.std(ddof=1), 2) if len(v) > 1 else 0.0
        rec["drop50"] = round(rec["clean"] - rec["diac50"], 2)
        rec["drop100"] = round(rec["clean"] - rec["diac100"], 2)
        rows.append(rec)
    tab = pd.DataFrame(rows)
    out = p(cfg, "results_dir") / "llm"
    tab.to_csv(out / "compare_mf1.csv", index=False)
    print(tab[["system", "kind", "n_runs", "clean", "diac50", "diac100", "drop50", "drop100"]].to_string(index=False))
    tests = []
    for ln, A in llm.items():
        for m in PLMS:
            obs, pv, lo, hi = boot(gold, A, plm[m], K)
            for i, q in enumerate(["clean", "diac100", "drop_diff"]):
                tests.append({"llm": ln, "plm": m, "quantity": q, "delta": round(obs[i], 2), "p": pv[i],
                              "ci_lo": round(lo[i], 2), "ci_hi": round(hi[i], 2)})
    if tests:
        tt = pd.DataFrame(tests)
        tt["p_holm"] = holm(tt["p"].to_numpy())
        tt["sig"] = tt["p_holm"] < 0.05
        tt.to_csv(out / "compare_tests.csv", index=False)
        print("\n(delta = LLM - PLM; drop_diff âm = LLM giảm ít hơn, tức bền hơn)")
        print(tt.pivot_table(index=["llm", "plm"], columns="quantity", values="delta").round(2).to_string())
        print("\nSố phép so có ý nghĩa sau Holm:", int(tt.sig.sum()), "/", len(tt))
        for q, g in tt.groupby("quantity"):
            pos, neg = int((g.sig & (g.delta > 0)).sum()), int((g.sig & (g.delta < 0)).sum())
            msg = f"{neg} LLM bền hơn / {pos} LLM kém bền hơn" if q == "drop_diff" else f"{pos} LLM cao hơn / {neg} LLM thấp hơn"
            print(f"  {q:9s}: {msg} (trên {len(g)} phép so)")


if __name__ == "__main__":
    main()
