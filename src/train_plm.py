"""Fine-tune PLM (PhoBERT-base, XLM-R-base, ViSoBERT, CafeBERT) cho phân loại cảm xúc 7 nhãn.

Cấu hình theo Mosbach et al. (2021): AdamW có bias correction (torch.optim.AdamW luôn bật),
lr 2e-5, warmup tuyến tính 10% số bước rồi giảm tuyến tính về 0, train nhiều epoch (tr.8);
batch 16, weight decay 0.01, clip gradient (thiết lập mặc định, tr.4). Mọi giá trị nằm trong config.
max_len 128, fp16 (autocast + GradScaler, chỉ khi chạy CUDA). Checkpoint tốt nhất chọn theo Macro-F1 trên valid
(với mốc trần RQ4: Macro-F1 lenient trên vigo_new_val).

Train trên vigo_new (mốc trần RQ4): đa nhãn 7 nhóm, loss BCE, dự đoán = nhóm có điểm cao nhất
(giao thức chuyển đa nhãn -> đơn nhãn của Bostan & Klinger, 2018, tr.9).

Sau khi train, model tốt nhất được chấm luôn trên mọi tập đánh giá (test sạch, test nhiễu, __dn, vigo_new_test),
ghi prediction từng câu, rồi (tùy config) lưu checkpoint fp16 để rq3_robustness.py chấm lại khi cần.

Chạy: python -m src.train_plm --config configs/base.yaml --exp rq1 [--model phobert] [--prep P0] [--seed 42]
"""
import argparse
import gc
import math
import time

import numpy as np
import pandas  # noqa: F401  import trước torch: với torch nightly trên Windows, thứ tự ngược lại làm crash pyarrow
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import build_eval_sets, build_train, eval_gold
from .noise import Noiser
from .preprocess import prepare
from .utils import (compute_metrics, enumerate_runs, get_device, is_done, labels, lenient_targets, load_config,
                    mark_done, run_dir, save_json, save_predictions, save_run_config, set_seed)


# ----------------------------------------------------------------------------- helpers
def load_model(mcfg, cfg, num_labels, multilabel):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    kw = dict(revision=mcfg.get("revision"), cache_dir=cfg["paths"].get("hf_cache"))
    tok = AutoTokenizer.from_pretrained(mcfg["hf"], **kw)
    model = AutoModelForSequenceClassification.from_pretrained(
        mcfg["hf"], num_labels=num_labels,
        problem_type="multi_label_classification" if multilabel else "single_label_classification", **kw)
    return tok, model


def encode(tok, texts, max_len):
    return tok(texts, truncation=True, max_length=max_len)["input_ids"]


class Collator:
    def __init__(self, tok, multilabel, n_labels):
        self.tok, self.multilabel, self.n = tok, multilabel, n_labels

    def __call__(self, batch):
        ids = [b[0] for b in batch]
        enc = self.tok.pad({"input_ids": ids}, return_tensors="pt")
        if batch[0][1] is None:
            return enc
        if self.multilabel:
            y = torch.zeros(len(batch), self.n)
            for i, b in enumerate(batch):
                y[i, b[1]] = 1.0
            enc["labels"] = y
        else:
            enc["labels"] = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return enc


@torch.no_grad()
def predict(model, tok, texts, cfg, device, multilabel):
    model.eval()
    ids = encode(tok, texts, cfg["train"]["max_len"])
    coll = Collator(tok, multilabel, 0)
    dl = DataLoader([(x, None) for x in ids], batch_size=cfg["train"]["eval_batch_size"], shuffle=False,
                    collate_fn=coll)
    out = []
    use_amp = cfg["train"]["fp16"] and device.type == "cuda"
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(**batch).logits.float()
        out.append((torch.sigmoid(logits) if multilabel else torch.softmax(logits, -1)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.config.num_labels))


def selection_score(es, scores, cfg):
    """Điểm chọn checkpoint: Macro-F1 (single) hoặc Macro-F1 lenient (vigo, theo mapping dùng khi train)."""
    pred = scores.argmax(1)
    labs = labels(cfg)
    if es["kind"] == "single":
        return compute_metrics(es["df"]["y"].tolist(), pred, len(labs))["mf1"]
    from .data import add_groups
    df = add_groups(es["df"], cfg, cfg["rq4"]["train_mapping"])
    pl = [labs[i] for i in pred]
    eff = lenient_targets(pl, df["groups"].map(set).tolist(), df["votes"].tolist(), labs)
    return compute_metrics([labs.index(g) for g in eff], pred, len(labs))["mf1"]


def prepared_texts(es, prep, cfg, segment):
    return prepare(es["df"]["text"].tolist(), prep, cfg, segment=segment, dictnorm=es["dictnorm"])


def evaluate_and_save(model, tok, eval_sets, prep, cfg, device, multilabel, segment, d, skip_existing=False):
    """Chấm model trên các tập đánh giá, ghi pred_<set>.csv. Trả về dict metrics (single-label) và thời gian."""
    labs = labels(cfg)
    metrics, times = {}, {}
    for name, es in eval_sets.items():
        f = d / f"pred_{name}.csv"
        if skip_existing and f.exists():
            continue
        texts = prepared_texts(es, prep, cfg, segment)
        t0 = time.time()
        scores = predict(model, tok, texts, cfg, device, multilabel)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times[name] = time.time() - t0
        pred = scores.argmax(1)
        gold_str, gold_int = eval_gold(es, cfg)
        save_predictions(f, es["df"]["id"].tolist(), gold_str, pred, scores, labs)
        if gold_int is not None:
            metrics[name] = compute_metrics(gold_int, pred, len(labs))
    return metrics, times


# ----------------------------------------------------------------------------- one run
def run_one(cfg, spec, force=False):
    d = run_dir(cfg, spec)
    if is_done(d) and not force:
        print(f"[skip] {spec.exp}/{spec.run_id} đã xong")
        return
    exp_cfg = cfg["experiments"][spec.exp]
    mcfg = cfg["models"][spec.model]
    # Thứ tự ghi đè: train chung < train riêng của model < train riêng của thí nghiệm (vd. tái lập cấu hình bài gốc, dò lr)
    tcfg = {**cfg["train"], **mcfg.get("train", {}), **exp_cfg.get("train", {})}
    cfg = {**cfg, "train": tcfg}
    device = get_device(cfg)
    segment = mcfg.get("segment", False)
    print(f"[run] {spec.exp}/{spec.run_id} device={device}")
    save_run_config(cfg, spec, d)
    set_seed(spec.seed)

    labs = labels(cfg)
    train_df, ttype = build_train(cfg, exp_cfg)
    multilabel = ttype == "multi"
    raw = train_df["text"].tolist()
    if spec.train_noise.startswith("tgt_"):   # D2 (review/09): nhân đôi train, bản sao bỏ dấu đúng các từ thuộc W
        from .unfamiliar import load_targets, targeted_copy
        W = load_targets(cfg, spec.train_noise[4:])
        cp = targeted_copy(raw, W)
        print(f"[tgt] W_{spec.train_noise[4:]}: {len(W)} từ; câu bị đổi {sum(a != b for a, b in zip(cp, raw))}/{len(raw)}")
        raw = raw + cp
        train_df = pd.concat([train_df, train_df], ignore_index=True)
    elif spec.train_noise == "dual":   # A-dual (review/12): nhân đôi train, bản sao bỏ HẾT dấu
        from .noise import strip_diacritics
        from .preprocess import base_clean
        raw = raw + [strip_diacritics(base_clean(t)) for t in raw]
        train_df = pd.concat([train_df, train_df], ignore_index=True)
    elif spec.train_noise.startswith("dualkl"):   # A-kl: cặp (gốc, bỏ hết dấu) + KL đối xứng; "dualkl_l2" = λ 2 (review/17)
        pass
    elif spec.train_noise != "none":
        raw, meta = Noiser(cfg).augment(raw, train_df["id"].tolist(), spec.train_noise, spec.seed)
        train_df.assign(text_noisy=raw, noise=[m[0] for m in meta], level=[m[1] for m in meta]) \
            [["id", "text", "text_noisy", "noise", "level"]].to_csv(d / "train_noisy.csv", index=False)
    texts = prepare(raw, spec.prep, cfg, segment=segment)
    ys = train_df["Y"].tolist() if multilabel else train_df["y"].tolist()

    eval_sets = build_eval_sets(cfg, exp_cfg, spec.prep)
    valid = eval_sets["valid"]
    valid_texts = prepared_texts(valid, spec.prep, cfg, segment)

    tok, model = load_model(mcfg, cfg, len(labs), multilabel)
    model.to(device)
    n_params = sum(x.numel() for x in model.parameters())

    ids = encode(tok, texts, tcfg["max_len"])
    g = torch.Generator()
    g.manual_seed(spec.seed)
    paired = spec.train_noise.startswith("dualkl")
    if paired:
        from .noise import strip_diacritics
        from .preprocess import base_clean
        ids_b = encode(tok, prepare([strip_diacritics(base_clean(t)) for t in raw], spec.prep, cfg, segment=segment),
                       tcfg["max_len"])
        base_col = Collator(tok, multilabel, len(labs))
        col = lambda b: (base_col([(x[0], x[2]) for x in b]), base_col([(x[1], x[2]) for x in b]))
        dl = DataLoader(list(zip(ids, ids_b, ys)), batch_size=tcfg["batch_size"], shuffle=True, generator=g, collate_fn=col)
        kl_lambda = float(spec.train_noise.split("_l", 1)[1]) if "_l" in spec.train_noise else float(tcfg.get("kl_lambda", 1.0))
        print(f"[dualkl] lambda = {kl_lambda}")
    else:
        dl = DataLoader(list(zip(ids, ys)), batch_size=tcfg["batch_size"], shuffle=True, generator=g,
                        collate_fn=Collator(tok, multilabel, len(labs)))
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "LayerNorm.bias")
    groups = [
        {"params": [p_ for n, p_ in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": tcfg["weight_decay"]},
        {"params": [p_ for n, p_ in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=float(tcfg["lr"]), betas=tuple(tcfg.get("betas", (0.9, 0.999))),
                            eps=float(tcfg.get("eps", 1e-8)))
    from transformers import get_linear_schedule_with_warmup
    total = len(dl) * tcfg["epochs"]
    sch = get_linear_schedule_with_warmup(opt, int(math.ceil(tcfg["warmup_ratio"] * total)), total)
    use_amp = tcfg["fp16"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best, best_state, history = -1.0, None, []
    t_train = time.time()
    for ep in range(tcfg["epochs"]):
        model.train()
        tot, nb = 0.0, 0
        for batch in dl:
            if paired:
                ba, bb = ({k: v.to(device) for k, v in x.items()} for x in batch)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    oa, ob = model(**ba), model(**bb)
                    la, lb = oa.logits.float().log_softmax(-1), ob.logits.float().log_softmax(-1)
                    kl = 0.5 * (torch.nn.functional.kl_div(la, lb, log_target=True, reduction="batchmean")
                                + torch.nn.functional.kl_div(lb, la, log_target=True, reduction="batchmean"))
                    loss = 0.5 * (oa.loss + ob.loss) + kl_lambda * kl
            else:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    loss = model(**batch).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
            scaler.step(opt)
            scaler.update()
            sch.step()
            tot += loss.item()
            nb += 1
        vs = selection_score(valid, predict(model, tok, valid_texts, cfg, device, multilabel), cfg)
        history.append({"epoch": ep + 1, "train_loss": tot / max(nb, 1), "valid_sel_mf1": vs})
        print(f"  epoch {ep + 1}/{tcfg['epochs']} loss={tot / max(nb, 1):.4f} valid_mf1={vs:.2f}")
        if vs > best:
            best = vs
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    train_time = time.time() - t_train
    peak_vram = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0

    model.load_state_dict(best_state)
    del best_state
    t_eval = time.time()
    metrics, times = evaluate_and_save(model, tok, eval_sets, spec.prep, cfg, device, multilabel, segment, d)
    eval_time = time.time() - t_eval
    n_test = len(eval_sets["test"]["df"]) if "test" in eval_sets else len(eval_sets["vigo_new_test"]["df"])
    t_inf = times.get("test", times.get("vigo_new_test", 0.0))
    out = {
        "metrics": metrics,
        "history": history,
        "best_epoch": int(np.argmax([h["valid_sel_mf1"] for h in history]) + 1),
        "cost": {"train_time_s": train_time, "eval_time_s": eval_time, "test_inference_s": t_inf,
                 "test_inference_ms_per_sent": 1000 * t_inf / max(n_test, 1), "peak_vram_gb": peak_vram,
                 "n_params": int(n_params), "n_train": len(texts), "device": str(device)},
        "model_revision": getattr(model.config, "_commit_hash", None),
    }
    save_json(out, d / "metrics.json")
    if exp_cfg.get("keep_checkpoint", False):
        ck = d / "checkpoint"
        model.to(torch.float16).save_pretrained(ck, safe_serialization=True)
        tok.save_pretrained(ck)
    mark_done(d)
    m = metrics.get("test")
    if m:
        print(f"[done] {spec.run_id} test Acc={m['acc']:.2f} WF1={m['wf1']:.2f} MF1={m['mf1']:.2f} "
              f"train={train_time / 60:.1f}m vram={peak_vram:.1f}GB")
    del model, opt, dl
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model")
    ap.add_argument("--prep")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    for spec in enumerate_runs(cfg, a.exp, kind="plm"):
        if (a.model and spec.model != a.model) or (a.prep and spec.prep != a.prep) or (a.seed and spec.seed != a.seed):
            continue
        run_one(cfg, spec, force=a.force)


if __name__ == "__main__":
    main()
