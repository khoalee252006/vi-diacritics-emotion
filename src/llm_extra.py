"""Phần 2, bước 4 (bổ sung): các phép kiểm định CHỐT TRƯỚC khi chạy (2026-10-14), Holm trên toàn bộ họ dưới đây.

(A) LLM so với model fine-tune KHÔNG DẤU (PS: train/valid/test đều bỏ hết dấu; results/strip) trên test bỏ dấu 100%.
    Ở bỏ dấu 100% mọi câu có r = 0 -> w = 0, nên gating trùng với PS: (A) cũng chính là "LLM so với gating".
    12 cấu hình LLM (6 model x 0/7-shot) x 6 hệ PS (lr, svm, phobert, xlmr, visobert, cafebert) = 72 phép so.
(B) LLM so với GATING (p = w*p_P0 + (1-w)*p_PS, w = min(1, r/r_train)) trên test bỏ dấu 50% (nơi gating khác PS):
    12 x 4 PLM = 48 phép so.
(C) Arcee-VyLinh so với Qwen2.5-3B-Instruct (cùng kiến trúc, cùng cỡ; khác ở tinh chỉnh tiếng Việt):
    0/7-shot x {sạch, bỏ dấu 100%, mức giảm} = 6 phép so.
Tổng 126 phép so, Holm chung. Bootstrap hai tầng (câu dùng chung; seed của hệ fine-tune), 10,000 vòng.
Ghép theo VỊ TRÍ dòng (id test VSMEC bị trùng).

Chạy: python -m src.llm_extra   -> results/llm/extra_tests.csv
"""
import numpy as np
import pandas as pd

from .ambiguity import MODELS
from .data import load_vsmec
from .gate_fusion import _probs, diac_ratio, gate_weights
from .llm_compare import N_BOOT, boot, llm_preds
from .stats import hier_bootstrap, holm, macro_f1_batch
from .utils import labels, load_config, p

PLMS = ["cafebert", "xlmr", "phobert", "visobert"]


def _read(f, test_ids, labs):
    x = pd.read_csv(f, dtype={"id": str})
    assert x["id"].tolist() == test_ids, f
    return x


def ps_preds(cfg, labs, test_ids):
    R = p(cfg, "results_dir") / "strip"
    return {m: np.array([[labs.index(v) for v in _read(R / f"{m}_PS_none_s{s}" / "pred_test.csv", test_ids, labs)["pred"]]
                         for s in seeds]) for m, seeds in MODELS.items()}


def gate_preds_diac50(cfg, labs, test_ids):
    """Gating trên test bỏ dấu 50%, đúng như src/gate_fusion.py: P0 và PS cùng chấm pred_diac50; w tính trên văn bản
    bỏ dấu 50%."""
    R = p(cfg, "results_dir")
    r_train = np.nanmean([diac_ratio(t) for t in load_vsmec(cfg, "train")["text"]])
    nz = pd.read_csv(R / "noise" / "test_diac50.csv", dtype={"id": str})
    assert nz["id"].tolist() == test_ids
    w = gate_weights(nz["text"].fillna("").tolist(), r_train)[:, None]
    cols = [f"s_{l}" for l in labs]
    out = {}
    for m in PLMS:
        rows = []
        for s in MODELS[m]:
            a = _probs(_read(R / "rq1" / f"{m}_P0_none_s{s}" / "pred_diac50.csv", test_ids, labs)[cols].to_numpy(float), m)
            b = _probs(_read(R / "strip" / f"{m}_PS_none_s{s}" / "pred_diac50.csv", test_ids, labs)[cols].to_numpy(float), m)
            rows.append((w * a + (1 - w) * b).argmax(1))
        out[m] = np.array(rows)
    return out


def main():
    cfg = load_config("configs/base.yaml")
    labs = labels(cfg)
    K = len(labs)
    t = load_vsmec(cfg, "test")
    test_ids = t["id"].astype(str).tolist()
    gold = np.array([labs.index(v) for v in t["label"]])
    llm = llm_preds(cfg, labs, len(gold))
    ps = ps_preds(cfg, labs, test_ids)
    gate = gate_preds_diac50(cfg, labs, test_ids)
    mf = lambda M: macro_f1_batch(np.broadcast_to(gold, M.shape), M, K).mean()
    print("MF1 tham chiếu: PS (bỏ dấu 100%):", {m: round(mf(v), 2) for m, v in ps.items()})
    print("                gating (bỏ dấu 50%):", {m: round(mf(v), 2) for m, v in gate.items()})
    rows = []
    for ln, A in llm.items():
        for m, B in ps.items():
            d, pv, lo, hi = hier_bootstrap(gold, A["diac100"], B, K, N_BOOT, 0)
            rows.append({"family": "A_llm_vs_PS@diac100", "a": ln, "b": f"{m}-PS", "delta": d, "p": pv, "ci_lo": lo, "ci_hi": hi})
        for m, B in gate.items():
            d, pv, lo, hi = hier_bootstrap(gold, A["diac50"], B, K, N_BOOT, 0)
            rows.append({"family": "B_llm_vs_gate@diac50", "a": ln, "b": f"{m}-gate", "delta": d, "p": pv, "ci_lo": lo, "ci_hi": hi})
    vy = {k: v for k, v in llm.items() if k.startswith("arcee-ai_Arcee-VyLinh")}
    qw = {k: v for k, v in llm.items() if k.startswith("Qwen_Qwen2.5-3B-Instruct")}
    for k in ("0-shot", "7-shot"):
        a, b = next(v for n, v in vy.items() if k in n), next(v for n, v in qw.items() if k in n)
        obs, pv, lo, hi = boot(gold, a, b, K)
        for i, q in enumerate(["clean", "diac100", "drop_diff"]):
            rows.append({"family": "C_VyLinh_vs_Qwen3B", "a": f"VyLinh ({k})", "b": f"Qwen2.5-3B ({k})", "quantity": q,
                         "delta": obs[i], "p": pv[i], "ci_lo": lo[i], "ci_hi": hi[i]})
    tt = pd.DataFrame(rows)
    tt["p_holm"] = holm(tt["p"].to_numpy())
    tt["sig"] = tt["p_holm"] < 0.05
    tt.round(4).to_csv(p(cfg, "results_dir") / "llm" / "extra_tests.csv", index=False)
    print(f"\nTổng {len(tt)} phép so (Holm chung).")
    for fam, g in tt.groupby("family"):
        print(f"\n== {fam}: {int((g.sig & (g.delta > 0)).sum())} a cao hơn / {int((g.sig & (g.delta < 0)).sum())} a thấp hơn / {len(g)}")
        if fam != "C_VyLinh_vs_Qwen3B":
            print(g.pivot_table(index="a", columns="b", values="delta").round(2).to_string())
        else:
            print(g[["a", "quantity", "delta", "ci_lo", "ci_hi", "p_holm"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
