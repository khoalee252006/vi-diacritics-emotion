"""Loader cho UIT-VSMEC (7 nhãn, đơn nhãn) và ViGoEmotions (28 nhãn, đa nhãn), ánh xạ 28 -> 7 nhãn,
và dựng dữ liệu train/valid/các tập đánh giá cho một run.

File dùng (đã kiểm tra, xem CLAUDE.md):
  - dataset/UIT-VSMEC/UIT-VSMEC/{train,valid,test}_nor_811.xlsx  (cột Emotion, Sentence)
  - dataset/derived/vigo_new_{train,val,test}.csv                 (id, text, labels)
  - dataset/derived/vigo_vsmec_{train,val,test}.csv               (+ vsmec_label, vsmec_split)
"""
import ast
import json
from collections import Counter

import pandas as pd

from .utils import labels, p

VSMEC_SPLITS = {"train": "train", "valid": "valid", "test": "test"}
VIGO_SPLITS = {"train": "train", "valid": "val", "test": "test"}


def _limit(df, cfg):
    n = cfg.get("limit_n")
    return df.head(n).reset_index(drop=True) if n else df.reset_index(drop=True)


# ----------------------------------------------------------------------------- UIT-VSMEC
def load_vsmec(cfg, split):
    """DataFrame [id, text, label(str), y(int)]. Giữ nguyên mọi dòng của split chính thức."""
    f = p(cfg, "vsmec_dir") / cfg["paths"]["vsmec_files"][split]
    df = pd.read_excel(f)
    labs = labels(cfg)
    out = pd.DataFrame({
        "id": df["Unnamed: 0"].astype(str) if "Unnamed: 0" in df else df.index.astype(str),
        "text": df["Sentence"].fillna("").astype(str),
        "label": df["Emotion"].astype(str).str.strip(),
    })
    bad = set(out["label"]) - set(labs)
    if bad:
        raise ValueError(f"Nhãn lạ trong {f}: {bad}")
    out["y"] = out["label"].map(labs.index)
    return _limit(out, cfg)


# ----------------------------------------------------------------------------- ViGoEmotions
def vigo_label_names(cfg):
    with open(p(cfg, "vigo_label_dict"), encoding="utf-8") as f:
        d = json.load(f)
    return [d[str(i)] for i in range(len(d))]


def load_vigo(cfg, part, split):
    """part: 'new' (13,743 câu mới) | 'vsmec' (6,921 câu gốc VSMEC). split: train|valid|test.
    DataFrame [id, text, fine(list[str]), (vsmec_label, vsmec_split)]."""
    f = p(cfg, "vigo_derived_dir") / f"vigo_{part}_{VIGO_SPLITS[split]}.csv"
    df = pd.read_csv(f)
    names = vigo_label_names(cfg)
    df["fine"] = df["labels"].map(lambda s: [names[i] for i in ast.literal_eval(str(s))])
    df["id"] = df["id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    return _limit(df, cfg)


def vigo_mapping(cfg, variant):
    """Bảng ánh xạ nhãn mịn ViGoEmotions -> nhóm VSMEC (project_knowledge.md §3).
    variant 'a': disapproval nằm ngoài phép ánh xạ (kết quả chính).
    variant 'b': disapproval -> Anger (phân tích độ nhạy)."""
    m = {}
    for group, fine_list in cfg["rq4"]["mapping"].items():
        for fl in fine_list:
            m[fl] = group
    if variant == "b":
        m["disapproval"] = "Anger"
    elif variant != "a":
        raise ValueError(variant)
    return m


def map_fine(fine, mapping):
    """Trả về (tập nhóm, số nhãn mịn rơi vào từng nhóm). Nhãn không có trong bảng bị bỏ qua."""
    votes = Counter(mapping[f] for f in fine if f in mapping)
    return set(votes), dict(votes)


def add_groups(df, cfg, variant):
    m = vigo_mapping(cfg, variant)
    mapped = df["fine"].map(lambda f: map_fine(f, m))
    df = df.copy()
    df["groups"] = mapped.map(lambda x: sorted(x[0], key=labels(cfg).index))
    df["votes"] = mapped.map(lambda x: x[1])
    return df


# ----------------------------------------------------------------------------- experiment data
def noise_test_sets(cfg):
    """Danh sách tên tập test nhiễu đã sinh (đọc từ noise_dir/manifest.json)."""
    mf = p(cfg, "noise_dir") / "manifest.json"
    if not mf.exists():
        return []
    with open(mf, encoding="utf-8") as f:
        return [x["name"] for x in json.load(f)["sets"]]


def load_noise_test(cfg, name):
    df = pd.read_csv(p(cfg, "noise_dir") / f"test_{name}.csv", dtype={"id": str})
    df["text"] = df["text"].fillna("").astype(str)
    return df


def build_eval_sets(cfg, exp_cfg, prep):
    """Trả về dict tên -> dict(df, dictnorm, kind). kind: 'single' (gold 1 nhãn) | 'vigo' (gold đa nhãn).

    Model train trên VSMEC: valid, test, (test__dn), các tập nhiễu (+ __dn), vigo_new_test.
    Hậu tố __dn = chuẩn hóa từ điển lúc test (cách bù 1), chỉ áp cho P0 vì P1 đã gồm bước này.
    Model train trên vigo_new (mốc trần RQ4): valid = vigo_new_val, và vigo_new_test.
    """
    sets = {}
    use_dn = prep == "P0" and exp_cfg.get("eval_dictnorm", True)
    if exp_cfg["train_data"] == "vsmec":
        sets["valid"] = dict(df=load_vsmec(cfg, "valid"), dictnorm=False, kind="single")
        test = load_vsmec(cfg, "test")
        sets["test"] = dict(df=test, dictnorm=False, kind="single")
        if exp_cfg.get("eval_noise", False):
            if use_dn:
                sets["test__dn"] = dict(df=test, dictnorm=True, kind="single")
            for name in noise_test_sets(cfg):
                nd = load_noise_test(cfg, name)
                sets[name] = dict(df=nd, dictnorm=False, kind="single")
                if use_dn:
                    sets[f"{name}__dn"] = dict(df=nd, dictnorm=True, kind="single")
        if exp_cfg.get("eval_vigo", False):
            sets["vigo_new_test"] = dict(df=load_vigo(cfg, "new", "test"), dictnorm=False, kind="vigo")
    elif exp_cfg["train_data"] == "vigo_new":
        v = cfg["rq4"]["train_mapping"]
        sets["valid"] = dict(df=_vigo_trainable(load_vigo(cfg, "new", "valid"), cfg, v), dictnorm=False, kind="vigo")
        sets["vigo_new_test"] = dict(df=load_vigo(cfg, "new", "test"), dictnorm=False, kind="vigo")
    else:
        raise ValueError(exp_cfg["train_data"])
    return sets


def _vigo_trainable(df, cfg, variant):
    """Giữ các câu có ít nhất một nhóm sau ánh xạ (câu chỉ có nhãn bị bỏ, vd. disapproval ở variant a, bị loại)."""
    df = add_groups(df, cfg, variant)
    return df[df["groups"].map(len) > 0].reset_index(drop=True)


def build_train(cfg, exp_cfg):
    """Train set gốc (chưa nhiễu, chưa tiền xử lý).
    VSMEC: cột y (int). vigo_new: cột Y (list nhóm, để train đa nhãn BCE theo Bostan & Klinger, 2018, tr.9)."""
    if exp_cfg["train_data"] == "vsmec":
        return load_vsmec(cfg, "train"), "single"
    df = _vigo_trainable(load_vigo(cfg, "new", "train"), cfg, cfg["rq4"]["train_mapping"])
    labs = labels(cfg)
    df["Y"] = df["groups"].map(lambda gs: [labs.index(g) for g in gs])
    return df, "multi"


def eval_gold(es, cfg):
    """Gold để lưu vào file prediction và để chọn checkpoint.
    single: (list nhãn str, list int). vigo: (chuỗi nhãn mịn, None)."""
    df = es["df"]
    if es["kind"] == "single":
        return df["label"].tolist(), df["y"].tolist()
    return df["fine"].map(lambda f: "|".join(f)).tolist(), None
