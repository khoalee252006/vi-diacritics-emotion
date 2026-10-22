"""Tách ViGoEmotions thành phần gốc VSMEC và phần mới, gắn nhãn VSMEC gốc.
Quy tắc: id thuần số < 6927 là bình luận lấy từ UIT-VSMEC (6,921 dòng, khớp Tran et al., 2026, tr.2-3).
Chạy: python split_vigoemotions_vsmec.py <vigo_corpus_dir> <vsmec_dir> <out_dir>
Thư viện: pandas, openpyxl.
"""
import re, sys, unicodedata
import pandas as pd

vg_dir, vs_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]

def norm(s):
    return re.sub(r'[\W_]+', '', unicodedata.normalize('NFC', str(s)).lower())

vs = pd.concat([pd.read_excel(f'{vs_dir}/{s}_nor_811.xlsx').assign(vsmec_split=s)
                for s in ['train', 'valid', 'test']], ignore_index=True)
vs['n'] = vs['Sentence'].map(norm)
vs = vs[vs['n'] != ''].drop_duplicates('n')
vs_map = vs.set_index('n')[['Emotion', 'vsmec_split']]

for sp in ['train', 'val', 'test']:
    d = pd.read_csv(f'{vg_dir}/{sp}.csv')
    ids = d['id'].astype(str)
    is_vs = ids.str.fullmatch(r'\d+') & (pd.to_numeric(ids, errors='coerce') < 6927)
    old = d[is_vs].copy()
    old = old.join(vs_map, on=old['text'].map(norm).rename('n'))
    old.rename(columns={'Emotion': 'vsmec_label'}).drop(columns=['key_0', 'n'], errors='ignore') \
       .to_csv(f'{out}/vigo_vsmec_{sp}.csv', index=False)
    d[~is_vs].to_csv(f'{out}/vigo_new_{sp}.csv', index=False)
    print(sp, 'vsmec-origin', len(old), 'matched label', old['Emotion'].notna().sum(), 'new', int((~is_vs).sum()))
