"""Sinh nhiễu có kiểm soát cho RQ3. Mặc định chỉ áp lên tập test VSMEC; hàm `noise_text` cũng được
dùng để augment tập train (rq3_aug, rq3_matrix).

Đơn vị là token tách theo khoảng trắng (VSMEC _nor_811 đã tách sẵn dấu câu). Số token bị đổi trong
mỗi câu là k = floor(tỉ lệ × mẫu số + 0.5), chọn ngẫu nhiên không lặp. Nếu không đủ token đủ điều kiện
thì đổi hết số đủ điều kiện; tỉ lệ thực tế được ghi vào `noise_stats.csv`.

1. Bỏ dấu (diac25/50/75/100). Mẫu số = số từ CÓ dấu trong câu, giống ViSoBERT (Nguyen et al., 2023,
   B8, tr.14: "diacritics of total words that included diacritics in each comment"). Bỏ dấu = bỏ mọi
   dấu thanh và dấu mũ/móc/trăng (NFD rồi xóa ký tự kết hợp), đ -> d.

2. Teencode (teen10/20/30). Mẫu số = tổng số token. Thay NGƯỢC: dựng từ điển đảo
   "dạng chuẩn -> các dạng không chuẩn" từ teencode4.txt và vi-nsw-dict.json (đã lọc như preprocess:
   bỏ khóa NSW là âm tiết chuẩn; chỉ giữ khóa có ít nhất một chữ cái). Dạng chuẩn dài 1–2 token. Các vị
   trí khớp được chọn ngẫu nhiên, không chồng nhau; cụm 2 token tính là 2 token bị đổi. Dạng không chuẩn
   được chọn đều trong các dạng có sẵn.

3. Lỗi gõ (typo10/20/30). Mẫu số = tổng số token. Token đủ điều kiện = có ít nhất một chữ cái. Với mỗi
   token được chọn, áp MỘT trong hai luật (chọn đều trong các luật áp được):
   (a) Telex không bật bộ gõ: từ được viết lại thành chuỗi phím Telex thô:
         â->aa, ê->ee, ô->oo, ă->aw, ơ->ow, ư->uw, đ->dd; dấu thanh thành phím ở CUỐI từ:
         sắc->s, huyền->f, hỏi->r, ngã->x, nặng->j.  Vd. "người" -> "nguwowif", "không" -> "khoong".
       Chỉ áp được khi từ có ký tự tiếng Việt có dấu.
   (b) Phím kề: một ký tự a–z (không dấu) trong từ được thay bằng một phím kề trên bàn phím QWERTY
       (bảng QWERTY_NEIGHBORS: phím trái/phải cùng hàng và các phím cùng vị trí ±1 ở hàng trên/dưới).
       Vd. "bạn" -> "vạn". Chỉ áp được khi từ có ký tự a–z.

4. Emoji (emoji_keep / emoji_text / emoji_remove). keep = test sạch (không sinh file riêng);
   text = emoji và emoticon -> chữ theo emojis.json; remove = xóa emoji (dải Unicode emoji) và các
   emoticon có trong emojis.json.

Mỗi câu dùng một bộ sinh ngẫu nhiên riêng, seed = "<noise.seed>|<tên tập>|<id câu>", nên kết quả
không phụ thuộc thứ tự câu hay PYTHONHASHSEED.

Chạy: python -m src.noise --config configs/base.yaml
"""
import json
import random
import re
import unicodedata

import pandas as pd

from .preprocess import base_clean, emoji_to_text, load_resources, remove_emoji
from .utils import load_config, p

TONE_KEYS = {"́": "s", "̀": "f", "̉": "r", "̃": "x", "̣": "j"}
SHAPE_MARKS = {"̂", "̆", "̛"}  # mũ, trăng, móc
QWERTY_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]


def _build_neighbors():
    pos = {c: (r, i) for r, row in enumerate(QWERTY_ROWS) for i, c in enumerate(row)}
    nb = {}
    for c, (r, i) in pos.items():
        cand = []
        for dr in (-1, 0, 1):
            rr = r + dr
            if 0 <= rr < len(QWERTY_ROWS):
                for di in (-1, 0, 1):
                    ii = i + di
                    if 0 <= ii < len(QWERTY_ROWS[rr]) and (dr, di) != (0, 0):
                        cand.append(QWERTY_ROWS[rr][ii])
        nb[c] = sorted(set(cand) - {c})
    return nb


QWERTY_NEIGHBORS = _build_neighbors()


# ----------------------------------------------------------------------------- primitives
def strip_diacritics(word):
    s = unicodedata.normalize("NFD", word)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s.replace("đ", "d").replace("Đ", "D"))


def has_diacritic(word):
    return strip_diacritics(word) != word


def to_telex(word):
    out, tone = [], ""
    for ch in word:
        if ch in "đĐ":
            out.append("dd" if ch == "đ" else "DD")
            continue
        d = unicodedata.normalize("NFD", ch)
        base, marks = d[0], d[1:]
        piece = base
        for mk in marks:
            if mk in TONE_KEYS:
                tone = TONE_KEYS[mk]
            elif mk == "̂":
                piece = base + base.lower()        # â->aa, ê->ee, ô->oo
            elif mk in ("̆", "̛"):
                piece = base + "w"                 # ă->aw, ơ->ow, ư->uw
        out.append(piece)
    return "".join(out) + tone


def adjacent_key(word, rng):
    idx = [i for i, ch in enumerate(word) if ch.lower() in QWERTY_NEIGHBORS]
    if not idx:
        return None
    i = rng.choice(idx)
    ch = word[i]
    rep = rng.choice(QWERTY_NEIGHBORS[ch.lower()])
    rep = rep.upper() if ch.isupper() else rep
    return word[:i] + rep + word[i + 1:]


def _k(rate, n):
    return int(rate * n + 0.5)


# ----------------------------------------------------------------------------- reverse teencode dict
class ReverseDict:
    def __init__(self, res):
        rev = {}
        pairs = [(k, v) for k, v in res.teen_raw.items()]
        pairs += [(k, s) for k, vs in res.nsw_raw.items() for s in vs]
        for nsw, std in pairs:
            std = std.lower().strip()
            if not re.search(r"[^\W\d_]", nsw) or not std or nsw == std:
                continue
            if 1 <= len(std.split()) <= 2:
                rev.setdefault(std, set()).add(nsw)
        self.rev = {k: sorted(v) for k, v in rev.items()}


# ----------------------------------------------------------------------------- noise functions
def noise_diac(text, rate, rng, **_):
    toks = text.split()
    cand = [i for i, t in enumerate(toks) if has_diacritic(t)]
    chosen = rng.sample(cand, min(len(cand), _k(rate, len(cand))))
    for i in chosen:
        toks[i] = strip_diacritics(toks[i])
    return " ".join(toks), len(chosen), len(cand)


def noise_teen(text, rate, rng, rev=None, **_):
    toks = text.split()
    k = _k(rate, len(toks))
    spans = []
    for i in range(len(toks)):
        for n in (2, 1):
            if i + n <= len(toks) and " ".join(toks[i:i + n]).lower() in rev.rev:
                spans.append((i, n))
    rng.shuffle(spans)
    used, chosen, changed = set(), [], 0
    for i, n in spans:
        if changed >= k:
            break
        cells = set(range(i, i + n))
        if cells & used or changed + n > k:
            continue
        used |= cells
        chosen.append((i, n))
        changed += n
    for i, n in sorted(chosen, reverse=True):
        std = " ".join(toks[i:i + n]).lower()
        toks[i:i + n] = [rng.choice(rev.rev[std])]
    return " ".join(toks), changed, len(text.split())


def noise_typo(text, rate, rng, **_):
    toks = text.split()
    cand = [i for i, t in enumerate(toks) if re.search(r"[^\W\d_]", t)]
    chosen = rng.sample(cand, min(len(cand), _k(rate, len(toks))))
    changed = 0
    for i in chosen:
        w = toks[i]
        rules = []
        if has_diacritic(w):
            rules.append("telex")
        if any(ch.lower() in QWERTY_NEIGHBORS for ch in w):
            rules.append("adjacent")
        if not rules:
            continue
        rule = rng.choice(rules)
        new = to_telex(w) if rule == "telex" else adjacent_key(w, rng)
        if new and new != w:
            toks[i] = new
            changed += 1
    return " ".join(toks), changed, len(toks)


def noise_emoji(text, level, rng, res=None, **_):
    new = emoji_to_text(text, res) if level == "text" else remove_emoji(text, res)
    return new, int(new != text), 1


NOISE_FUNCS = {"diac": noise_diac, "teen": noise_teen, "typo": noise_typo, "emoji": noise_emoji}


class Noiser:
    def __init__(self, cfg):
        self.cfg = cfg
        self.res = load_resources(cfg)
        self.rev = ReverseDict(self.res)
        self.levels = cfg["noise"]["levels"]   # dict type -> list mức

    def apply(self, text, ntype, level, key):
        rng = random.Random(f"{self.cfg['noise']['seed']}|{key}")
        return NOISE_FUNCS[ntype](text, level, rng, rev=self.rev, res=self.res)

    def augment(self, texts, ids, mode, seed):
        """Augment train. mode = 'mix': mỗi câu chọn đều 1 loại nhiễu rồi chọn đều 1 mức của loại đó
        (theo cách trộn nhiễu của Belinkov & Bisk, 2018, tr.7–8). mode = tên 1 loại: chỉ dùng loại đó."""
        types = list(self.levels) if mode == "mix" else [mode]
        out, meta = [], []
        for t, i in zip(texts, ids):
            r = random.Random(f"aug|{seed}|{mode}|{i}")
            nt = r.choice(types)
            lv = r.choice(self.levels[nt])
            new, ch, _ = self.apply(t, nt, lv, f"aug|{seed}|{mode}|{nt}|{lv}|{i}")
            out.append(new)
            meta.append((nt, lv, ch))
        return out, meta


def set_name(ntype, level):
    if ntype == "emoji":
        return f"emoji_{level}"
    return f"{ntype}{int(round(level * 100))}"


def generate_test_sets(cfg, force=False):
    from .data import load_vsmec
    out_dir = p(cfg, "noise_dir")
    if (out_dir / "manifest.json").exists() and not force:
        print(f"[noise] đã có {out_dir / 'manifest.json'}, bỏ qua (dùng --force để sinh lại)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    test = load_vsmec(cfg, "test")
    clean = [base_clean(t, cfg["preprocess"].get("lowercase", True)) for t in test["text"]]
    nz = Noiser(cfg)
    manifest, stats = [], []
    for ntype, levels in nz.levels.items():
        for lv in levels:
            name = set_name(ntype, lv)
            rows, ch_sum, den_sum, n_aff = [], 0, 0, 0
            for (_, r), c in zip(test.iterrows(), clean):
                new, ch, den = nz.apply(c, ntype, lv, f"{name}|{r['id']}")
                rows.append({"id": r["id"], "label": r["label"], "y": r["y"], "text": new, "text_clean": c,
                             "n_changed": ch})
                ch_sum += ch
                den_sum += den
                n_aff += int(new != c)
            pd.DataFrame(rows).to_csv(out_dir / f"test_{name}.csv", index=False)
            manifest.append({"name": name, "type": ntype, "level": lv})
            stats.append({"set": name, "type": ntype, "level": lv, "n_sent": len(rows),
                          "sent_changed": n_aff, "units_changed": ch_sum, "units_denominator": den_sum,
                          "realized_rate": ch_sum / den_sum if den_sum else 0.0})
            print(f"[noise] {name:<14} câu bị đổi {n_aff}/{len(rows)}  tỉ lệ thực tế {stats[-1]['realized_rate']:.3f}")
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"seed": cfg["noise"]["seed"], "sets": manifest}, f, ensure_ascii=False, indent=2)
    pd.DataFrame(stats).to_csv(out_dir / "noise_stats.csv", index=False)
    print(f"[noise] đã lưu {len(manifest)} tập vào {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--force", action="store_true", help="sinh lại kể cả khi đã có (các run cũ sẽ không còn khớp)")
    a = ap.parse_args()
    generate_test_sets(load_config(a.config), force=a.force)
