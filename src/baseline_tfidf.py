"""Baseline TF-IDF + LogisticRegression / LinearSVC.

- Token: regex `\\w+|[^\\w\\s]+` trên văn bản đã tiền xử lý (giữ được emoji, emoticon, từ 1 ký tự).
- Siêu tham số chọn trên tập valid theo Macro-F1, lưới trong config (`tfidf.grid`). Lưới được dò một lần
  cho mỗi (model, prep, train_noise) với seed đầu tiên; các seed sau dùng lại bộ siêu tham số đó.
  LR (lbfgs) cho kết quả gần như tất định nên độ lệch chuẩn giữa các seed có thể bằng 0.
- Train trên vigo_new (mốc trần RQ4): OneVsRest đa nhãn trên 7 nhóm, dự đoán = nhóm có điểm cao nhất.
- Model được lưu bằng joblib để rq3_robustness.py chấm lại khi cần.

Chạy: python -m src.baseline_tfidf --config configs/base.yaml --exp rq1
"""
import argparse
import itertools
import re
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from .data import build_eval_sets, build_train, eval_gold
from .noise import Noiser
from .preprocess import prepare
from .train_plm import selection_score
from .utils import (compute_metrics, enumerate_runs, is_done, labels, load_config, load_json, mark_done, run_dir,
                    save_json, save_predictions, save_run_config, set_seed)

TOKEN_RE = re.compile(r"\w+|[^\w\s]+")


def tokenize(s):
    return TOKEN_RE.findall(s)


def make_clf(name, C, class_weight, seed, max_iter, multilabel):
    if name == "lr":
        base = LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter, random_state=seed)
    elif name == "svm":
        base = LinearSVC(C=C, class_weight=class_weight, max_iter=max_iter, random_state=seed)
    else:
        raise ValueError(name)
    return OneVsRestClassifier(base) if multilabel else base


def scores_of(clf, X):
    if hasattr(clf, "predict_proba") and not isinstance(clf, LinearSVC):
        try:
            return clf.predict_proba(X)
        except AttributeError:
            pass
    s = clf.decision_function(X)
    return s if s.ndim == 2 else np.stack([-s, s], 1)


def n_weights(clf):
    """Số trọng số tuyến tính (coef + intercept)."""
    ests = getattr(clf, "estimators_", [clf])
    return int(sum(e.coef_.size + np.size(e.intercept_) for e in ests))


class TfidfModel:
    def __init__(self, vec, clf):
        self.vec, self.clf = vec, clf

    def scores(self, texts):
        return scores_of(self.clf, self.vec.transform(texts))


def load_model(path):
    """Nạp model.joblib. File được ghi khi module này chạy dưới tên __main__ (python -m src.baseline_tfidf) nên
    pickle tham chiếu __main__.TfidfModel, __main__.tokenize...: đăng ký các class/hàm của module vào __main__."""
    import __main__
    import sys
    mod = sys.modules[__name__]
    for k, v in vars(mod).items():
        if getattr(v, "__module__", None) == __name__:
            setattr(__main__, k, v)
    return joblib.load(path)


def fit(texts, ys, params, clf_name, seed, tcfg, multilabel, n_labels):
    vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                          ngram_range=tuple(params["ngram_range"]), min_df=params["min_df"],
                          sublinear_tf=params["sublinear_tf"])
    X = vec.fit_transform(texts)
    if multilabel:
        ys = MultiLabelBinarizer(classes=list(range(n_labels))).fit_transform(ys)
    clf = make_clf(clf_name, params["C"], params["class_weight"], seed, tcfg["max_iter"], multilabel)
    clf.fit(X, ys)
    return TfidfModel(vec, clf)


def grid_search(texts, ys, valid, valid_texts, clf_name, seed, cfg, multilabel):
    tcfg = cfg["tfidf"]
    g = tcfg["grid"]
    keys = list(g)
    best, best_p, log = -1, None, []
    for combo in itertools.product(*[g[k] for k in keys]):
        params = dict(zip(keys, combo))
        m = fit(texts, ys, params, clf_name, seed, tcfg, multilabel, len(labels(cfg)))
        s = selection_score(valid, m.scores(valid_texts), cfg)
        log.append({**params, "valid_sel_mf1": s})
        if s > best:
            best, best_p = s, params
    return best_p, best, log


def run_group(cfg, specs, force=False):
    """specs: các RunSpec cùng (exp, model, prep, train_noise), khác seed."""
    todo = [s for s in specs if force or not is_done(run_dir(cfg, s))]
    for s in specs:
        if s not in todo:
            print(f"[skip] {s.exp}/{s.run_id} đã xong")
    if not todo:
        return
    first = specs[0]
    exp_cfg = cfg["experiments"][first.exp]
    labs = labels(cfg)
    eval_sets = build_eval_sets(cfg, exp_cfg, first.prep)
    valid = eval_sets["valid"]
    valid_texts = prepare(valid["df"]["text"].tolist(), first.prep, cfg)

    best_file = run_dir(cfg, first) / "best_params.json"
    best_params = load_json(best_file)["params"] if best_file.exists() else None
    for spec in todo:
        d = run_dir(cfg, spec)
        save_run_config(cfg, spec, d)
        set_seed(spec.seed)
        train_df, ttype = build_train(cfg, exp_cfg)
        multilabel = ttype == "multi"
        raw = train_df["text"].tolist()
        if spec.train_noise == "dual":   # nhận xét thầy 2026-10-21: nhân đôi train, bản sao bỏ HẾT dấu (như train_plm)
            import pandas as pd
            from .noise import strip_diacritics
            from .preprocess import base_clean
            raw = raw + [strip_diacritics(base_clean(t)) for t in raw]
            train_df = pd.concat([train_df, train_df], ignore_index=True)
        elif spec.train_noise != "none":
            raw, meta = Noiser(cfg).augment(raw, train_df["id"].tolist(), spec.train_noise, spec.seed)
        texts = prepare(raw, spec.prep, cfg)
        ys = train_df["Y"].tolist() if multilabel else train_df["y"].tolist()
        t0 = time.time()
        if best_params is None:
            best_params, bs, log = grid_search(texts, ys, valid, valid_texts, spec.model, spec.seed, cfg, multilabel)
            save_json({"params": best_params, "valid_sel_mf1": bs, "grid": log}, best_file)
            print(f"[grid] {spec.model} {spec.prep} {spec.train_noise}: {best_params} valid MF1={bs:.2f}")
        model = fit(texts, ys, best_params, spec.model, spec.seed, cfg["tfidf"], multilabel, len(labs))
        train_time = time.time() - t0
        metrics, t_inf = {}, 0.0
        t_eval = time.time()
        for name, es in eval_sets.items():
            et = prepare(es["df"]["text"].tolist(), spec.prep, cfg, dictnorm=es["dictnorm"])
            t1 = time.time()
            sc = model.scores(et)
            if name in ("test", "vigo_new_test") and not t_inf:
                t_inf = time.time() - t1
            pred = sc.argmax(1)
            gs, gi = eval_gold(es, cfg)
            save_predictions(d / f"pred_{name}.csv", es["df"]["id"].tolist(), gs, pred, sc, labs)
            if gi is not None:
                metrics[name] = compute_metrics(gi, pred, len(labs))
        n_test = len(eval_sets.get("test", eval_sets.get("vigo_new_test"))["df"])
        save_json({"metrics": metrics, "best_params": best_params,
                   "cost": {"train_time_s": train_time, "eval_time_s": time.time() - t_eval,
                            "test_inference_s": t_inf, "test_inference_ms_per_sent": 1000 * t_inf / max(n_test, 1),
                            "n_params": n_weights(model.clf), "peak_vram_gb": 0.0,
                            "n_train": len(texts), "device": "cpu"}}, d / "metrics.json")
        joblib.dump(model, d / "model.joblib")
        mark_done(d)
        m = metrics.get("test")
        if m:
            print(f"[done] {spec.exp}/{spec.run_id} test Acc={m['acc']:.2f} WF1={m['wf1']:.2f} MF1={m['mf1']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    groups = {}
    for spec in enumerate_runs(cfg, a.exp, kind="classic"):
        groups.setdefault((spec.model, spec.prep, spec.train_noise), []).append(spec)
    for specs in groups.values():
        run_group(cfg, specs, force=a.force)


if __name__ == "__main__":
    main()
