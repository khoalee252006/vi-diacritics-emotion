"""Tiện ích chung: đọc config, đặt seed, metric, quản lý thư mục run, liệt kê run.

Quy ước thư mục kết quả:
    <results_dir>/<exp>/<run_id>/
        config.yaml        cấu hình đã hợp nhất của run
        metrics.json       Acc / WF1 / MF1 cho từng tập đánh giá + chi phí (thời gian, VRAM, số tham số)
        pred_<set>.csv     prediction từng câu (id, gold, pred, điểm từng lớp)
        DONE               ghi sau cùng; có file này thì run được coi là xong và bị bỏ qua khi chạy lại
"""
import copy
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

LABELS_DEFAULT = ["Enjoyment", "Sadness", "Fear", "Anger", "Disgust", "Surprise", "Other"]


# ----------------------------------------------------------------------------- config
def _deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path):
    """Đọc YAML; khóa `_base_` (đường dẫn tương đối với file hiện tại) để kế thừa và ghi đè.

    Mọi đường dẫn tương đối trong `paths` được giải theo thư mục gốc dự án
    (thư mục cha của thư mục chứa file config gốc, tức `configs/..`).
    """
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "_base_" in cfg:
        base = load_config(path.parent / cfg.pop("_base_"))
        root = base["_root_"]
        cfg = _deep_merge({k: v for k, v in base.items() if k != "_root_"}, cfg)
    else:
        root = str(path.parent.parent)
    cfg["_root_"] = root
    cfg["_config_path_"] = str(path)
    return cfg


def p(cfg, key):
    """Trả về đường dẫn tuyệt đối của cfg['paths'][key] (None nếu không đặt)."""
    v = cfg["paths"].get(key)
    if v is None:
        return None
    v = Path(v)
    return v if v.is_absolute() else Path(cfg["_root_"]) / v


def labels(cfg):
    return cfg.get("labels", LABELS_DEFAULT)


# ----------------------------------------------------------------------------- seed / device
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(cfg):
    import torch
    d = os.environ.get("DEVICE") or cfg.get("device", "auto")  # biến môi trường DEVICE ghi đè config
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


# ----------------------------------------------------------------------------- metrics
def compute_metrics(gold, pred, n_labels):
    """Accuracy, Weighted-F1, Macro-F1 (thang 0–100). Macro-F1 tính trên đủ n_labels lớp."""
    from sklearn.metrics import accuracy_score, f1_score
    labs = list(range(n_labels))
    return {
        "acc": 100 * accuracy_score(gold, pred),
        "wf1": 100 * f1_score(gold, pred, labels=labs, average="weighted", zero_division=0),
        "mf1": 100 * f1_score(gold, pred, labels=labs, average="macro", zero_division=0),
        "n": int(len(gold)),
    }


def canonical_group(gold_set, group_votes, order):
    """Chọn một nhóm đại diện cho tập nhãn vàng nhiều nhóm: nhóm có nhiều nhãn mịn nhất, hòa thì theo `order`."""
    return max(gold_set, key=lambda g: (group_votes.get(g, 0), -order.index(g)))


def lenient_targets(pred, gold_sets, votes, order):
    """Quy ước chấm lenient cho đa nhãn: nếu pred thuộc tập nhóm vàng thì coi gold = pred,
    ngược lại gold = nhóm đại diện (canonical_group). Accuracy khi đó đúng bằng lenient accuracy;
    Weighted/Macro-F1 tính trên cặp (gold hiệu dụng, pred)."""
    return [pr if pr in gs else canonical_group(gs, vt, order) for pr, gs, vt in zip(pred, gold_sets, votes)]


# ----------------------------------------------------------------------------- run io
@dataclass(frozen=True)
class RunSpec:
    exp: str
    kind: str          # "plm" | "classic"
    model: str
    prep: str          # "P0" | "P1"
    train_noise: str   # "none" | "mix" | "diac" | "teen" | "typo" | "emoji"
    seed: int

    @property
    def run_id(self):
        return f"{self.model}_{self.prep}_{self.train_noise}_s{self.seed}"


def run_dir(cfg, spec):
    return p(cfg, "results_dir") / spec.exp / spec.run_id


def is_done(d):
    return (Path(d) / "DONE").exists()


def mark_done(d):
    (Path(d) / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_run_config(cfg, spec, d, extra=None):
    d.mkdir(parents=True, exist_ok=True)
    out = {"run": spec.__dict__, "env": env_info(), **(extra or {}),
           "config": {k: v for k, v in cfg.items() if not k.startswith("_")}}
    with open(d / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)


def env_info():
    info = {"python": sys.version.split()[0]}
    for m in ["numpy", "pandas", "sklearn", "torch", "transformers", "tokenizers"]:
        try:
            info[m] = str(__import__(m).__version__)
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = str(torch.version.cuda)
    except Exception:
        pass
    return info


def save_predictions(path, ids, gold, pred, scores, label_names):
    import pandas as pd
    df = pd.DataFrame({"id": ids, "gold": gold, "pred": [label_names[i] for i in pred]})
    for j, name in enumerate(label_names):
        df[f"s_{name}"] = np.round(scores[:, j], 5)
    df.to_csv(path, index=False)


def mean_std(values, digits=2):
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return ""
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():.{digits}f} ± {sd:.{digits}f}"


def _tex_escape(s):
    s = str(s)
    for a, b in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"),
                 ("±", "$\\pm$")]:
        s = s.replace(a, b)
    return s


def save_table(df, stem, caption=""):
    """Ghi bảng ra <stem>.csv và <stem>.tex (tabular đơn giản, không cần jinja2)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    cols = list(df.columns)
    lines = ["\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{" + "l" * len(cols) + "}", "\\hline",
             " & ".join(_tex_escape(c) for c in cols) + " \\\\", "\\hline"]
    for _, r in df.iterrows():
        lines.append(" & ".join(_tex_escape("" if (isinstance(v, float) and np.isnan(v)) else
                                            (f"{v:.2f}" if isinstance(v, float) else v)) for v in r) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    if caption:
        lines.append("\\caption{" + _tex_escape(caption) + "}")
    lines.append("\\end{table}")
    stem.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[table] {stem}.csv / .tex ({len(df)} dòng)")


def collect_metrics(cfg, exps):
    """Bảng dài: exp, model, prep, train_noise, seed, set, acc, wf1, mf1 (+ cost) từ các run đã DONE."""
    import pandas as pd
    rows = []
    for exp in exps:
        if exp not in cfg["experiments"]:
            continue
        for spec in enumerate_runs(cfg, exp):
            d = run_dir(cfg, spec)
            if not (is_done(d) and (d / "metrics.json").exists()):
                continue
            m = load_json(d / "metrics.json")
            for s, v in m.get("metrics", {}).items():
                rows.append({**spec.__dict__, "kind": spec.kind, "set": s, **v, **m.get("cost", {}),
                             "best_epoch": m.get("best_epoch")})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- run enumeration
def enumerate_runs(cfg, exp_name, kind=None):
    exp = cfg["experiments"][exp_name]
    out = []
    for k, key in [("classic", "classic_models"), ("plm", "plm_models")]:
        if kind and k != kind:
            continue
        for m in exp.get(key, []) or []:
            if k == "plm":
                n = cfg["models"][m].get("max_seeds", len(cfg["seeds"]))
            else:
                n = cfg["tfidf"].get("max_seeds", len(cfg["seeds"]))
            n = min(n, exp.get("max_seeds", n))
            for prep in exp["preps"]:
                for tn in exp["train_noise"]:
                    for s in cfg["seeds"][:n]:
                        out.append(RunSpec(exp_name, k, m, prep, tn, s))
    return out


def experiment_order(cfg):
    return list(cfg["experiments"].keys())


def estimate(cfg):
    """In số run còn lại và ước lượng thời gian.

    Ưu tiên dùng thời gian đo được (train_time_s + eval_time_s trong metrics.json) của các run đã xong,
    theo cặp (exp, model) rồi tới (model). Nếu chưa có run nào, dùng `estimate_minutes` trong config,
    đây là số ĐOÁN, chưa đo.
    """
    measured, measured_model = {}, {}
    for exp in experiment_order(cfg):
        for spec in enumerate_runs(cfg, exp):
            d = run_dir(cfg, spec)
            if is_done(d) and (d / "metrics.json").exists():
                m = load_json(d / "metrics.json").get("cost", {})
                t = (m.get("train_time_s", 0) + m.get("eval_time_s", 0)) / 60
                measured.setdefault((exp, spec.model), []).append(t)
                measured_model.setdefault(spec.model, []).append(t)
    total_left = 0.0
    print(f"{'experiment':<14}{'model':<10}{'done':>6}{'left':>6}{'min/run':>9}{'nguồn':>8}{'phút còn':>10}")
    for exp in experiment_order(cfg):
        groups = {}
        for spec in enumerate_runs(cfg, exp):
            g = groups.setdefault(spec.model, [0, 0])
            g[0 if is_done(run_dir(cfg, spec)) else 1] += 1
        for model, (done, left) in groups.items():
            if (exp, model) in measured:
                per, src = float(np.median(measured[(exp, model)])), "đo"
            elif model in measured_model:
                per, src = float(np.median(measured_model[model])) * cfg["estimate_size_factor"].get(exp, 1.0), "đo~"
            else:
                per, src = cfg["estimate_minutes"].get(model, 5) * cfg["estimate_size_factor"].get(exp, 1.0), "đoán"
            total_left += per * left
            print(f"{exp:<14}{model:<10}{done:>6}{left:>6}{per:>9.1f}{src:>8}{per * left:>10.0f}")
    print(f"Tổng thời gian còn lại ước lượng: {total_left / 60:.1f} giờ "
          f"('đoán' = số trong config chưa đo; 'đo~' = đo ở thí nghiệm khác, nhân hệ số kích thước dữ liệu)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["estimate"])
    ap.add_argument("--config", default="configs/base.yaml")
    a = ap.parse_args()
    estimate(load_config(a.config))
