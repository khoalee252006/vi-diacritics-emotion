# Missing Diacritics in Vietnamese Emotion Recognition

Code, design documents and per-sentence predictions for the paper
*Missing Diacritics in Vietnamese Emotion Recognition: Ambiguous Key Words, Silent Failures, and Dual-View
Consistency Training* (anonymous submission).

## Contents

| Path | What it is |
|---|---|
| `src/` | All code: data loading, noise generation, fine-tuning, analyses, tables and figures |
| `configs/base.yaml` | Every experiment (models, seeds, recipe, noise levels) |
| `scripts/`, `run_all.sh`, `setup_vast.sh` | Environment setup and batch runs on a single GPU machine |
| `code/split_vigoemotions_vsmec.py` | Splits ViGoEmotions into its UIT-VSMEC part and the new part |
| `design_docs/` | Analysis plans with success criteria, written before each experiment was run (in Vietnamese), with change logs |
| `results/` | Aggregated results (CSV) behind every table, and word lists |
| `results/predictions/` | Per-sentence predictions on UIT-VSMEC (validation, test, and test with 25/50/75/100% of diacritics removed): sentence id, gold label, predicted label and class scores |

No dataset text is included. Predictions on ViGoEmotions are not released because its licence prohibits
redistribution.

## Data (download separately)

- **UIT-VSMEC**: place `train_nor_811.xlsx`, `valid_nor_811.xlsx`, `test_nor_811.xlsx` in `dataset/UIT-VSMEC/UIT-VSMEC/`.
- **ViGoEmotions** (non-commercial research only, no redistribution): clone the official repository into
  `dataset/github/ViGoEmotions/`, then run
  `python code/split_vigoemotions_vsmec.py dataset/github/ViGoEmotions/corpus dataset/UIT-VSMEC/UIT-VSMEC dataset/derived`.
- **ViLexNorm** (for validating the diacritic-loss detector): place its `data/*.csv` in `dataset/github/ViLexNorm/data/`.
- **Normalisation dictionary**: `vi-nsw-dict.json` in `dataset/github/vn-nsw-dictionary/`.
- VnCoreNLP (word segmentation for PhoBERT) is downloaded by `setup_vast.sh`.

## Setup

```bash
bash setup_vast.sh          # Java, Python 3.12 venv, torch 2.9.1 (CUDA 12.8), requirements, VnCoreNLP, model weights
source .env.sh
```

All fine-tuning runs used RTX 4090 GPUs; analyses run on CPU.

## Reproducing the paper

| Paper | Commands |
|---|---|
| Benchmark, noise robustness (Appendix A) | `bash run_all.sh`, then `python -m src.paper_tables` |
| Lexicon, class-indicative words, intervention (Sect. 4, Table 1) | `python -m src.ambiguity all`; `python -m src.intervention build|infer|analyze`; `python -m src.intervention_rev clear-build|clear-infer|clear-analyze` |
| Model-specific word lists (appendix) | `python -m src.intervention_rev occl-score|occl-build|occl-infer|occl-analyze` |
| Silent failure (Sect. 5, Table 2) | `python -m src.uncertainty h1`, `h2`, `h3`, `h4`; `python -m src.uncgate prep|run` |
| Targeted exposure (Sect. 6) | `python -m src.train_plm --exp tgt --model <m>`; `python -m src.tgt_eval score|analyze` |
| Dual-view training and baselines (Sect. 6) | `python -m src.train_plm --exp dual --model <m>`, `--exp strip`, `--exp dual_lam`; `python -m src.baseline_tfidf --exp dual_tfidf`; `python -m src.ab_score` |
| Pre-specified tests (non-inferiority, ablation, calibration) | `python -m src.ab_analyze`; `python -m src.rev_analyze`; `python -m src.rev_cpu calibts|restore|dvllm|fewshot` |
| LLMs (Sect. 7) | `python -m src.llm_eval run --base-url <endpoint> --model <name>`; `python -m src.llm_compare`; `python -m src.llm_extra`; open models with `scripts/vast_llm.sh` |
| Natural diacritic loss (Sect. 8) | `python -m src.natural build|infer|analyze|vilexnorm` |
| Tables and figures of Sect. 4–8 | `python -m src.paper_v2` |

`<m>` is one of `xlmr`, `phobert`, `visobert`, `cafebert`. Seeds are fixed in `configs/base.yaml`.
