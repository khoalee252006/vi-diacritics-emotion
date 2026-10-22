"""Baseline "khôi phục dấu rồi mới phân loại" (hướng của Le-Hong et al., 2021, KBS 233, tr.13–14), bước 1: khôi phục dấu.

Dùng model khôi phục dấu tiếng Việt có sẵn trên Hugging Face: peterhung/vietnamese-accent-marker-xlm-roberta
(XLM-R Large, token classification, 528 tag, Apache-2.0). Cách áp tag theo đúng model card: gộp subword thành từ,
áp tag đầu tiên làm đổi từ đó.

Mỗi câu: base_clean -> bỏ hết dấu -> khôi phục dấu. Áp cho MỌI câu (kể cả test sạch), vì lúc dùng thật không biết
câu nào bị mất dấu; nhờ đó đo được cả cái giá trên test sạch.

Chạy (trên máy có GPU, vd. laptop):
  python -m src.restore_diacritics --config configs/base.yaml --noise-dir <thư mục test_*.csv> --out <thư mục ra>
Ra: <out>/test_<tên>.csv (id, label, y, text = câu đã khôi phục, text_in = câu đầu vào), cho "test" và mọi tập nhiễu.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .data import load_vsmec
from .noise import strip_diacritics
from .preprocess import base_clean
from .utils import load_config

REPO = "peterhung/vietnamese-accent-marker-xlm-roberta"


def load_restorer(device):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(REPO).to(device).eval()
    if device.type == "cuda":
        model = model.half()
    with open(hf_hub_download(REPO, "selected_tags_names.txt"), encoding="utf-8") as f:
        tags = [x.strip() for x in f if x.strip()]
    assert len(tags) == 528, len(tags)
    return tok, model, tags, torch


def apply_tags(word, tag_ids, tags):
    for t in tag_ids:                    # tag đầu tiên làm đổi từ (theo model card)
        raw, vowel = tags[t].split("-")
        if raw and raw in word:
            return word.replace(raw, vowel)
    return word


def restore(sentences, tok, model, tags, torch, device, batch_size=64):
    out = []
    for i in range(0, len(sentences), batch_size):
        words = [s.split() or [""] for s in sentences[i:i + batch_size]]
        enc = tok(words, is_split_into_words=True, truncation=True, max_length=512, padding=True, return_tensors="pt")
        with torch.no_grad():
            pred = model(**{k: v.to(device) for k, v in enc.items()}).logits.argmax(-1).cpu().numpy()
        for b, ws in enumerate(words):
            per_word = {}
            for pos, wid in enumerate(enc.word_ids(b)):
                if wid is not None:
                    per_word.setdefault(wid, []).append(int(pred[b, pos]))
            out.append(" ".join(apply_tags(w, per_word.get(j, []), tags) for j, w in enumerate(ws)).strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--noise-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model, tags, torch = load_restorer(device)

    nd, out = Path(a.noise_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(nd / "manifest.json", encoding="utf-8") as f:
        names = [x["name"] for x in json.load(f)["sets"]]
    sets = {"test": load_vsmec(cfg, "test")[["id", "label", "y", "text"]]}
    for n in names:
        sets[n] = pd.read_csv(nd / f"test_{n}.csv", dtype={"id": str})[["id", "label", "y", "text"]]

    lc = cfg["preprocess"].get("lowercase", True)
    for name, df in sets.items():
        inp = [base_clean(str(t), lc) for t in df["text"].fillna("")]
        rst = restore([strip_diacritics(t) for t in inp], tok, model, tags, torch, device)
        res = df.assign(text_in=inp, text=rst)
        res.to_csv(out / f"test_{name}.csv", index=False)
        print(f"[restore] {name:<13} {len(res)} câu | vd: {inp[0][:50]!r} -> {rst[0][:50]!r}")


if __name__ == "__main__":
    main()
