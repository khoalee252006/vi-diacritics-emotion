"""Tiền xử lý văn bản: P0 / P1, chuẩn hóa bằng từ điển, tách từ VnCoreNLP cho PhoBERT.

Bước chung cho mọi chế độ: Unicode NFC, (tùy config) viết thường, gộp khoảng trắng.
UIT-VSMEC bản _nor_811 vốn đã viết thường, nên P0 trên VSMEC chính là văn bản gốc.

P0: giữ nguyên văn bản (chỉ bước chung).
P1: (1) emoji/emoticon -> chữ theo `emojis.json` (khớp chuỗi dài nhất trước; ký tự cuối của emoticon được
        phép lặp, vd. "=)))))" khớp khóa "=))"; bỏ U+FE0F và mã màu da khi so khớp);
    (2) chuẩn hóa teencode theo `teencode4.txt` (ưu tiên) và `vi-nsw-dict.json` (lấy dạng chuẩn đầu tiên).

Chuẩn hóa từ điển (`normalize_words`) cũng là "cách bù 1" của RQ3 khi áp lên model P0.
  - Khớp tham lam tối đa 3 token liền nhau, so khớp phần lõi của token (bỏ dấu câu bao quanh), không phân biệt hoa thường.
  - Loại khỏi từ điển NSW các khóa một âm tiết vốn là âm tiết chuẩn (xuất hiện trong vế chuẩn của
    hai từ điển), vd. "là", "gì", "mà", "chiến". Nếu không loại, chuẩn hóa mù sẽ sửa sai từ đúng
    ("chiến" -> "chuyến"). Hàm `load_resources` in ra số khóa bị loại.

Tách từ PhoBERT: VnCoreNLP RDRSegmenter qua `py_vncorenlp` (cần Java). pyjnius làm hỏng ký tự ngoài BMP
(emoji như 😂), nên chuỗi được cắt quanh các ký tự đó, chỉ phần chữ được đưa vào VnCoreNLP, emoji giữ
nguyên thành token riêng. Kết quả tách từ được cache ra file JSON để chạy lại không phải tách lại.
"""
import json
import os
import re
import threading
import unicodedata
import urllib.request
from functools import lru_cache
from pathlib import Path

from .utils import p

VS16 = "️"
SKIN = re.compile("[\U0001F3FB-\U0001F3FF]")
NON_BMP = re.compile("([\U00010000-\U0010FFFF]+)")
# Dải ký tự emoji dùng cho nhiễu "xóa emoji" (ngoài các emoticon trong emojis.json).
EMOJI_CHARS = re.compile("[\U0001F000-\U0001FAFF☀-➿⌀-⏿⬀-⯿️‍\U0001F3FB-\U0001F3FF]+")
LETTER = re.compile(r"[^\W\d_]")
CORE = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)


# ----------------------------------------------------------------------------- resources
class Resources:
    def __init__(self, emoji_map, emoji_re, word_map, max_ngram, patterns, nsw_raw, teen_raw, removed_nsw):
        self.emoji_map = emoji_map
        self.emoji_re = emoji_re
        self.word_map = word_map
        self.max_ngram = max_ngram
        self.patterns = patterns
        self.nsw_raw = nsw_raw          # dict NSW -> [chuẩn], sau khi lọc
        self.teen_raw = teen_raw        # dict teencode -> chuẩn
        self.removed_nsw = removed_nsw


def _strip_emoji_mods(s):
    return SKIN.sub("", s.replace(VS16, ""))


@lru_cache(maxsize=4)
def _load_resources_cached(emoji_path, teen_path, nsw_path, patterns_path):
    with open(emoji_path, encoding="utf-8") as f:
        emo = json.load(f)
    emoji_map = {}
    for k, v in emo.items():
        k2 = _strip_emoji_mods(unicodedata.normalize("NFC", k)).lower()
        if k2.strip():
            emoji_map.setdefault(k2, v)
    keys = sorted(emoji_map, key=len, reverse=True)
    # Emoticon ASCII được phép lặp ký tự cuối: "=))" khớp cả "=)))))". Emoji Unicode lặp ("😂😂") vẫn đổi từng cái.
    emoji_re = re.compile("|".join(f"(?:{re.escape(k)}{re.escape(k[-1])}*)" if k.isascii() else re.escape(k)
                                   for k in keys))

    teen = {}
    with open(teen_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) == 2 and parts[0].strip():
                teen.setdefault(unicodedata.normalize("NFC", parts[0].strip().lower()),
                                unicodedata.normalize("NFC", parts[1].strip()))
    with open(nsw_path, encoding="utf-8") as f:
        nsw = {unicodedata.normalize("NFC", k.strip().lower()): [unicodedata.normalize("NFC", x) for x in v]
               for k, v in json.load(f).items() if k.strip() and v}
    standard = {w for v in nsw.values() for s in v for w in s.lower().split()}
    standard |= {w for v in teen.values() for w in v.lower().split()}
    removed = sorted(k for k in nsw if " " not in k and k in standard)
    nsw = {k: v for k, v in nsw.items() if k not in set(removed)}

    word_map = {k: v[0] for k, v in nsw.items()}
    word_map.update(teen)  # teencode4 ưu tiên
    max_ngram = min(3, max(len(k.split()) for k in word_map))

    patterns = []
    if patterns_path:
        with open(patterns_path, encoding="utf-8") as f:
            patterns = [(re.compile(k), v) for k, v in json.load(f).items()]
    return Resources(emoji_map, emoji_re, word_map, max_ngram, patterns, nsw, teen, removed)


def load_resources(cfg, verbose=False):
    pp = p(cfg, "patterns") if cfg["preprocess"].get("use_patterns", False) else None
    r = _load_resources_cached(str(p(cfg, "emoji_dict")), str(p(cfg, "teencode_dict")),
                               str(p(cfg, "nsw_dict")), str(pp) if pp else None)
    if verbose:
        print(f"[preprocess] emoji keys={len(r.emoji_map)} word_map={len(r.word_map)} "
              f"(teencode={len(r.teen_raw)}, nsw={len(r.nsw_raw)}, nsw bị loại vì là âm tiết chuẩn={len(r.removed_nsw)})")
    return r


# ----------------------------------------------------------------------------- basic ops
def base_clean(text, lowercase=True):
    text = "" if text is None or (isinstance(text, float)) else str(text)
    text = unicodedata.normalize("NFC", text)
    if lowercase:
        text = text.lower()
    return " ".join(text.split())


def apply_patterns(text, res):
    for rx, rep in res.patterns:
        text = rx.sub(rep, text)
    return text


def emoji_to_text(text, res):
    text = _strip_emoji_mods(text)
    text = res.emoji_re.sub(lambda m: f" {res.emoji_map[_emoji_key(m.group(0), res)]} ", text)
    return " ".join(text.split())


def _emoji_key(matched, res):
    """Đưa chuỗi khớp (có thể có ký tự cuối lặp) về khóa trong từ điển."""
    k = matched
    while k not in res.emoji_map and len(k) > 1 and k[-1] == k[-2]:
        k = k[:-1]
    return k


def remove_emoji(text, res):
    text = _strip_emoji_mods(text)
    text = res.emoji_re.sub(" ", text)
    text = EMOJI_CHARS.sub(" ", text)
    return " ".join(text.split())


def normalize_words(text, res):
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        hit = False
        for n in range(min(res.max_ngram, len(toks) - i), 0, -1):
            chunk = toks[i:i + n]
            m_first, m_last = CORE.match(chunk[0]), CORE.match(chunk[-1])
            if n == 1:
                pre, core, suf = m_first.group(1), m_first.group(2), m_first.group(3)
            else:
                pre, suf = m_first.group(1), m_last.group(3)
                inner = [chunk[0][len(pre):]] + chunk[1:-1] + [chunk[-1][:len(chunk[-1]) - len(suf)]]
                core = " ".join(inner)
            key = core.lower()
            if key and key in res.word_map:
                out.append(pre + res.word_map[key] + suf)
                i += n
                hit = True
                break
        if not hit:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def p1_transform(text, res, use_patterns=False):
    if use_patterns:
        text = apply_patterns(text, res)
    text = emoji_to_text(text, res)
    return normalize_words(text, res)


# ----------------------------------------------------------------------------- VnCoreNLP
VNCORENLP_FILES = ["VnCoreNLP-1.2.jar", "models/wordsegmenter/vi-vocab", "models/wordsegmenter/wordsegmenter.rdr"]


def download_vncorenlp(save_dir, base_url):
    save_dir = Path(save_dir)
    (save_dir / "models/wordsegmenter").mkdir(parents=True, exist_ok=True)
    for f in VNCORENLP_FILES:
        dst = save_dir / f
        if not dst.exists() or dst.stat().st_size == 0:
            print(f"[vncorenlp] tải {f}")
            urllib.request.urlretrieve(base_url + f, dst)
    return save_dir


class Segmenter:
    _model = None
    _lock = threading.Lock()

    def __init__(self, cfg):
        self.cfg = cfg
        # Biến môi trường VNCORENLP_DIR ghi đè config: Java trên Windows không nạp được classpath có ký tự
        # non-ASCII (vd. thư mục có dấu), nên khi chạy local cần trỏ tới một thư mục ASCII.
        self.dir = Path(os.environ.get("VNCORENLP_DIR") or p(cfg, "vncorenlp_dir"))
        self.cache_path = p(cfg, "cache_dir") / "vncorenlp_wseg.json"
        self.cache = {}
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                self.cache = json.load(f)

    def _get_model(self):
        if Segmenter._model is None:
            download_vncorenlp(self.dir, self.cfg["preprocess"]["vncorenlp_url"])
            import py_vncorenlp
            cwd = os.getcwd()  # py_vncorenlp đổi thư mục làm việc khi khởi tạo
            try:
                Segmenter._model = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(self.dir.resolve()),
                                                          max_heap_size=self.cfg["preprocess"].get("java_heap", "-Xmx2g"))
            finally:
                os.chdir(cwd)
        return Segmenter._model

    def _vn(self, text):
        model = self._get_model()
        parts = []
        for chunk in NON_BMP.split(text):
            if not chunk.strip():
                continue
            if NON_BMP.fullmatch(chunk):
                parts.append(chunk)
            else:
                parts.extend(model.word_segment(chunk))
        return " ".join(parts).split()

    def _segment_one(self, text):
        """Giữ nguyên token đã nối sẵn (có "_", vd. emoji đã đổi thành chữ), token không có chữ cái
        (emoticon, số, dấu câu) và emoticon có trong emojis.json; phần còn lại đưa vào VnCoreNLP."""
        emo = load_resources(self.cfg).emoji_map
        out, buf = [], []
        for t in text.split():
            if "_" in t or not LETTER.search(t) or t in emo:
                if buf:
                    out.extend(self._vn(" ".join(buf)))
                    buf = []
                out.append(t)
            else:
                buf.append(t)
        if buf:
            out.extend(self._vn(" ".join(buf)))
        return " ".join(out)

    def segment(self, texts):
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if new:
            with Segmenter._lock:
                for t in new:
                    self.cache[t] = self._segment_one(t)
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.cache_path.with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, ensure_ascii=False)
                os.replace(tmp, self.cache_path)
        return [self.cache[t] for t in texts]


_SEGMENTER = None


def get_segmenter(cfg):
    global _SEGMENTER
    if _SEGMENTER is None:
        _SEGMENTER = Segmenter(cfg)
    return _SEGMENTER


# ----------------------------------------------------------------------------- main entry
def prepare(texts, mode, cfg, segment=False, dictnorm=False):
    """Áp tiền xử lý cho một danh sách câu.

    mode: "P0" | "P1" | "PS". dictnorm=True: chuẩn hóa từ điển lúc test (cách bù 1 của RQ3); với P1 đã có sẵn.
    "PS": như P0 rồi bỏ TOÀN BỘ dấu (dạng chuẩn không dấu). Câu mất dấu ở mọi mức đều quy về cùng một dạng;
          dùng cho model "không dấu" của phương pháp gating theo tỉ lệ dấu. Cùng hàm strip_diacritics với nhiễu diac.
    segment=True: tách từ VnCoreNLP (chỉ dùng cho PhoBERT), làm sau cùng.
    """
    pc = cfg["preprocess"]
    res = load_resources(cfg)
    out = [base_clean(t, pc.get("lowercase", True)) for t in texts]
    if mode == "P1":
        out = [p1_transform(t, res, pc.get("use_patterns", False)) for t in out]
    elif mode == "P0":
        if dictnorm:
            out = [normalize_words(t, res) for t in out]
    elif mode == "PS":
        from .noise import strip_diacritics  # import muộn: noise.py import preprocess
        out = [strip_diacritics(t) for t in out]
    else:
        raise ValueError(mode)
    if segment:
        out = get_segmenter(cfg).segment(out)
    return out


if __name__ == "__main__":
    import argparse
    import sys
    from .utils import load_config
    ap = argparse.ArgumentParser(description="Thử tiền xử lý trên vài câu")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--segment", action="store_true")
    ap.add_argument("texts", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config)
    load_resources(cfg, verbose=True)
    texts = a.texts or ["ko bít nói j luôn =))))) 😂😂", "chiến thắng là của mình 👍", "mik thik bạn lắm nha :v"]
    for mode in ["P0", "P1"]:
        for t, o in zip(texts, prepare(texts, mode, cfg, segment=a.segment)):
            sys.stdout.write(f"{mode} | {t} -> {o}\n")
