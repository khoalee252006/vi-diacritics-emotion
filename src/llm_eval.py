"""Phần 2: LLM có bền với mất dấu không? Phân loại 7 nhãn UIT-VSMEC bằng LLM qua API kiểu OpenAI-compatible
(OpenAI, Gemini endpoint tương thích OpenAI, OpenRouter, hoặc server cục bộ vLLM/Ollama cho model mở).

Thiết kế (cố định trước khi chạy):
  - Tập: test sạch (P0), bỏ dấu 50%, bỏ dấu 100% — đúng các file results/noise/test_*.csv đã dùng cho PLM, cùng id.
  - Prompt GIỐNG HỆT nhau ở mọi điều kiện; KHÔNG gợi ý rằng câu có thể thiếu dấu (để đo độ bền thật).
  - zero-shot và 7-shot (1 ví dụ/lớp, rút cố định từ VSMEC train, seed 0, giữ nguyên dấu ở mọi điều kiện).
  - temperature 0, chỉ trả về một nhãn. Lấy nhãn ĐẦU TIÊN khớp trong câu trả lời; không khớp -> INVALID (tính là sai).
  - Mọi phản hồi được lưu cache (jsonl) để chạy tiếp không tốn thêm tiền và để kiểm tra lại.

Chạy (key đọc từ biến môi trường, không ghi vào file):
  python -m src.llm_eval run --base-url https://api.openai.com/v1 --model gpt-4o-mini --key-env OPENAI_API_KEY --limit 20
  python -m src.llm_eval report
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import threading
import time
from pathlib import Path

import pandas as pd
import requests

from .data import load_vsmec
from .preprocess import base_clean
from .utils import compute_metrics, labels, load_config, p

DEFS = {
    "Enjoyment": "joy, happiness, love, admiration, amusement",
    "Sadness": "sadness, disappointment, regret, pity, loneliness",
    "Fear": "fear, worry, anxiety, being scared",
    "Anger": "anger, irritation, hostility toward someone or something",
    "Disgust": "disgust, contempt, strong dislike, mockery",
    "Surprise": "surprise, astonishment, disbelief",
    "Other": "no emotion, or an emotion not covered by the six categories above",
}
SETS = {"clean": None, "diac50": "test_diac50.csv", "diac100": "test_diac100.csv"}


def system_prompt(labs):
    lines = "\n".join(f"- {l}: {DEFS[l]}" for l in labs)
    return ("You classify the emotion expressed in Vietnamese social-media comments.\n"
            f"Choose exactly one label:\n{lines}\n"
            "Reply with the label name only, nothing else.")


def fewshot(cfg, labs, seed=0):
    tr = load_vsmec(cfg, "train")
    tr = tr[tr["text"].str.split().str.len().between(5, 25)]
    rng = random.Random(seed)
    ex = []
    for l in labs:
        rows = tr[tr["label"] == l]
        r = rows.iloc[rng.randrange(len(rows))]
        ex.append((base_clean(r["text"]), l, str(r["id"])))
    return ex


def load_set(cfg, name):
    """Cột `pos` = vị trí dòng trong test VSMEC, là KHÓA duy nhất (cột id bị trùng: 693 dòng, 608 id)."""
    test_ids = load_vsmec(cfg, "test")["id"].astype(str).tolist()
    if name == "clean":
        t = load_vsmec(cfg, "test")
        df = pd.DataFrame({"id": t["id"].astype(str), "text": t["text"].map(base_clean), "label": t["label"]})
    else:
        df = pd.read_csv(p(cfg, "results_dir") / "noise" / SETS[name], dtype={"id": str})
        df = df[["id", "text", "label"]].assign(text=lambda x: x["text"].fillna("").map(base_clean))
    assert df["id"].tolist() == test_ids, name
    return df.assign(pos=range(len(df)))


def migrate(cfg, d):
    """Bản ghi cũ (khóa theo id, chưa có pos): gán pos cho id duy nhất; bỏ bản ghi có id trùng để gọi lại."""
    f = d / "responses.jsonl"
    rows = [json.loads(x) for x in open(f, encoding="utf-8")]
    if all("pos" in r for r in rows):
        return
    test_ids = load_vsmec(cfg, "test")["id"].astype(str).tolist()
    cnt = pd.Series(test_ids).value_counts()
    pos_of = {i: k for k, i in enumerate(test_ids) if cnt[i] == 1}
    keep = [{**r, "pos": pos_of[r["id"]]} if "pos" not in r else r for r in rows if "pos" in r or r["id"] in pos_of]
    f.rename(d / "responses_v1_idkey.jsonl")
    with open(f, "w", encoding="utf-8") as g:
        for r in keep:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[migrate] {d.name}: giữ {len(keep)}/{len(rows)} bản ghi; bỏ {len(rows) - len(keep)} bản ghi có id trùng (sẽ gọi lại)")


def parse(text, labs):
    """Nhãn đầu tiên khớp. Model in phần suy nghĩ vào câu trả lời (Gemma: <thought>...</thought>) -> chỉ đọc phần SAU
    thẻ đóng cuối cùng; có thẻ mở mà không có thẻ đóng (bị cắt) -> INVALID."""
    text = text or ""
    if "<thought>" in text and "</thought>" not in text:
        return "INVALID"
    text = text.rsplit("</thought>", 1)[-1]
    hits = [(m.start(), l) for l in labs for m in [re.search(rf"\b{l}\b", text or "", re.I)] if m]
    return min(hits)[1] if hits else "INVALID"


class Client:
    def __init__(self, base_url, model, key_env, max_tokens=8, extra=None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        key = os.environ.get(key_env) if key_env else None
        self.headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
        self.max_tokens = max_tokens
        self.extra = extra or {}

    def chat(self, messages, retries=10):
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": self.max_tokens, **self.extra}
        for i in range(retries):
            try:
                r = requests.post(self.url, headers=self.headers, json=body, timeout=300)
                if r.status_code == 200:
                    j = r.json()
                    ch = j["choices"][0]
                    usage = {**(j.get("usage") or {}), "finish_reason": ch.get("finish_reason")}
                    return ch.get("message", {}).get("content"), usage, j.get("model", self.model)
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(2 ** i, 60))
                    continue
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            except requests.RequestException:
                time.sleep(min(2 ** i, 60))
        raise RuntimeError("hết lượt thử lại")


def run(cfg, a):
    labs = labels(cfg)
    out = p(cfg, "results_dir") / "llm" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    cache_f = out / "responses.jsonl"
    done = set()
    if cache_f.exists():
        migrate(cfg, out)
        for line in open(cache_f, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["set"], r["shots"], r["pos"]))
    ex = fewshot(cfg, labs)
    if a.strip_examples:   # E4 (review/17): ví dụ few-shot bỏ hết dấu (cùng câu, cùng nhãn)
        from .noise import strip_diacritics
        ex = [(strip_diacritics(t), l, i) for t, l, i in ex]
    (out / "config.json").write_text(json.dumps({"base_url": a.base_url, "model": a.model, "shots": a.shots,
                                                 "strip_examples": a.strip_examples,
                                                 "max_tokens": a.max_tokens, "extra": json.loads(a.extra),
                                                 "sets": a.sets, "fewshot_ids": [e[2] for e in ex],
                                                 "system": system_prompt(labs)}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    cl = Client(a.base_url, a.model, a.key_env, a.max_tokens, json.loads(a.extra))
    lock = threading.Lock()
    jobs = []
    for name in a.sets.split(","):
        df = load_set(cfg, name)
        if a.limit:
            df = df.head(a.limit)
        for shots in [int(s) for s in a.shots.split(",")]:
            for pos, rid, text, gold in zip(df["pos"], df["id"], df["text"], df["label"]):
                if (name, shots, pos) not in done:
                    jobs.append((name, shots, pos, rid, text, gold))
    print(f"[llm] {a.model}: {len(jobs)} lượt gọi cần chạy (đã có {len(done)})")

    def work(job):
        name, shots, pos, rid, text, gold = job
        msgs = [{"role": "system", "content": system_prompt(labs)}]
        for t, l, _ in (ex if shots else []):
            msgs += [{"role": "user", "content": t}, {"role": "assistant", "content": l}]
        msgs.append({"role": "user", "content": text})
        t0 = time.time()
        try:
            resp, usage, served = cl.chat(msgs)
        except RuntimeError as e:   # bỏ qua lượt này; lần chạy sau sẽ gọi lại (cache theo pos)
            print(f"  [lỗi] {name} {shots}-shot pos={pos}: {str(e)[:120]}")
            return
        rec = {"set": name, "shots": shots, "pos": pos, "id": rid, "text": text, "gold": gold, "response": resp, "pred": parse(resp, labs),
               "latency": round(time.time() - t0, 3), "usage": usage, "served_model": served}
        with lock:
            with open(cache_f, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with cf.ThreadPoolExecutor(a.workers) as pool:
        for i, _ in enumerate(pool.map(work, jobs), 1):
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}")
    report(cfg, a.tag)


def report(cfg, tag=None):
    labs = labels(cfg)
    base = p(cfg, "results_dir") / "llm"
    rows = []
    for d in sorted(base.iterdir()) if tag is None else [base / tag]:
        f = d / "responses.jsonl"
        if not f.exists():
            continue
        r = pd.DataFrame([json.loads(x) for x in open(f, encoding="utf-8")]).drop_duplicates(["set", "shots", "pos"])
        for (s, k), g in r.groupby(["set", "shots"]):
            gold = [labs.index(x) for x in g["gold"]]
            pred = [labs.index(x) if x in labs else -1 for x in g["pred"]]
            m = compute_metrics(gold, pred, len(labs))
            tok = g["usage"].map(lambda u: (u or {}).get("total_tokens", 0)).sum()
            rows.append({"run": d.name, "set": s, "shots": k, "n": len(g), "acc": round(m["acc"], 2),
                         "mf1": round(m["mf1"], 2), "invalid_%": round(100 * (g["pred"] == "INVALID").mean(), 2),
                         "tokens_per_1k": round(1000 * tok / len(g)), "sec_per_call": round(g["latency"].mean(), 2)})
    t = pd.DataFrame(rows)
    if len(t):
        t.to_csv(base / "summary.csv", index=False)
        print(t.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "report"])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--key-env", default="OPENAI_API_KEY")
    ap.add_argument("--tag", help="tên thư mục kết quả (mặc định = tên model)")
    ap.add_argument("--sets", default="clean,diac50,diac100")
    ap.add_argument("--shots", default="0,7")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N câu đầu mỗi tập (thử prompt)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8, help="model có suy nghĩ in ra (Gemma) cần lớn hơn, vd. 1024")
    ap.add_argument("--strip-examples", action="store_true", help="ví dụ few-shot bỏ hết dấu (review/17, E4)")
    ap.add_argument("--extra", default="{}", help='tham số riêng dạng JSON, vd. {"reasoning_effort": "none"}')
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.step == "report":
        return report(cfg)
    a.tag = a.tag or re.sub(r"[^\w.-]", "_", a.model)
    run(cfg, a)


if __name__ == "__main__":
    main()
