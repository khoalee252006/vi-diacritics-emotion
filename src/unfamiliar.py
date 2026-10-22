"""Giả thuyết "dạng lạ" (review/09_thiet_ke_dang_la.md), D1 quan sát trên 7,074 biến thể một âm tiết.

U1 = log(1 + số lần dạng không dấu xuất hiện như một âm tiết trong VSMEC train)  (quen thuộc)
U2 = số subword(dạng không dấu) − số subword(dạng có dấu) theo tokenizer của từng model
Mô hình: dp ~ U1 + U2 + confus + emo + n_syl + rel_pos + log_freq, cụm theo câu; Holm 8 (4 PLM x U1, U2).

Chạy: python -m src.unfamiliar d1   -> results/unfamiliar/
"""
import argparse
from collections import Counter

import numpy as np
import pandas as pd

from .ambiguity import SYL
from .data import load_vsmec
from .noise import strip_diacritics
from .preprocess import base_clean
from .stats import holm
from .uncertainty import PLMS, iv_dir, single_deltas
from .utils import labels, load_config, p


def out_dir(cfg):
    d = p(cfg, "results_dir") / "unfamiliar"
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_counts(cfg):
    c = Counter()
    for t in load_vsmec(cfg, "train")["text"]:
        c.update(SYL.findall(base_clean(t)))
    return c


def d1(cfg):
    import statsmodels.formula.api as smf
    from transformers import AutoTokenizer
    labs = labels(cfg)
    inst = pd.read_csv(iv_dir(cfg) / "single_instances.csv", dtype={"id": str, "sent_id": str}, keep_default_na=False)
    lex = pd.read_csv(p(cfg, "results_dir") / "ambiguity" / "lexicon.csv", keep_default_na=False).set_index("word")
    inst["log_freq"] = np.log(inst["word"].map(lex["count"]).astype(float))
    cnt = train_counts(cfg)
    inst["k"] = inst["word"].map(strip_diacritics)
    inst["U1"] = np.log1p(inst["k"].map(lambda k: cnt.get(k, 0)))
    print(f"[D1] dạng không dấu chưa từng gặp trong train: {(inst.U1 == 0).mean():.1%} biến thể "
          f"(từ cảm xúc: {(inst[inst.emo == 1].U1 == 0).mean():.1%})")
    R = p(cfg, "results_dir") / "rq1"
    rows, frames = [], []
    for m in PLMS:
        tok = AutoTokenizer.from_pretrained(R / f"{m}_P0_none_s42" / "checkpoint")
        npc = {}
        n = lambda w: npc.setdefault(w, len(tok.tokenize(w)))
        x = pd.concat([inst, single_deltas(cfg, labs, m, inst)], axis=1)
        x["U2"] = [n(k) - n(w) for k, w in zip(x["k"], x["word"])]
        frames.append(x.assign(model=m))
        fit = smf.ols("dp ~ U1 + U2 + confus + emo + n_syl + rel_pos + log_freq", x).fit(
            cov_type="cluster", cov_kwds={"groups": x["sent_pos"].to_numpy()})
        ci = fit.conf_int()
        for term in ["U1", "U2", "confus"]:
            rows.append({"model": m, "term": term, "coef": fit.params[term], "ci_lo": ci.loc[term, 0],
                         "ci_hi": ci.loc[term, 1], "t": fit.tvalues[term], "p": fit.pvalues[term],
                         "primary": term in ("U1", "U2")})
        print(f"  {m:9s} U2>0 ở {(x.U2 > 0).mean():.1%} biến thể; U2 TB {x.U2.mean():+.2f}")
    t = pd.DataFrame(rows)
    pr = t["primary"].to_numpy()
    t["p_holm"] = np.nan
    t.loc[pr, "p_holm"] = holm(t.loc[pr, "p"].to_numpy())
    t["sig"] = t["p_holm"] < 0.05
    d = out_dir(cfg)
    t.round(5).to_csv(d / "d1_regression.csv", index=False)
    pd.concat(frames).round(4).to_csv(d / "d1_instances.csv", index=False)
    print("\nD1 (dự đoán G: U1 < 0, U2 > 0; Holm 8 cho U1, U2; confus là mô tả):")
    print(t.round(4).to_string(index=False))
    ok = int(((t.term == "U1") & t.sig & (t.coef < 0)).sum())
    print(f"\nTiêu chí D1 (U1 < 0 có ý nghĩa ở >= 3/4 PLM): {ok}/4 -> {'ỦNG HỘ' if ok >= 3 else 'KHÔNG ủng hộ'}")


def targeted_copy(texts, W):
    """D2: bản sao của câu với mọi âm tiết thuộc W bị bỏ dấu (các âm tiết khác giữ nguyên)."""
    W = set(W)
    return [SYL.sub(lambda m: strip_diacritics(m.group()) if m.group() in W else m.group(), base_clean(t)) for t in texts]


def load_targets(cfg, name):
    f = p(cfg, "results_dir") / "unfamiliar" / f"W_{name}.txt"
    return [w for w in f.read_text(encoding="utf-8").split("\n") if w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["d1"])
    a = ap.parse_args()
    globals()[a.step](load_config("configs/base.yaml"))


if __name__ == "__main__":
    main()
