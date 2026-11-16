#!/usr/bin/env python3
"""Writer-bootstrap intervals and whole-writer permutation tests for the adjusted HC-183 estimates (IJDAR).

Refits the fixed pipelines on cached features: linear SVM (C=0.1), covariates regressed out of every feature
within training folds only, stratified 5-fold writer CV over seeds 2026-2045, writer score = mean margin.
Each case first reproduces its published point estimate exactly (assert). Intervals: class-stratified writer
bootstrap (2000 resamples, seed 7, as run_final_experiments). Tests: 999 whole-writer relabellings (rng 2029),
the whole adjusted pipeline refitted for each. Holm correction over the six one-sided adjusted HC-183 tests;
the LBP case is a separate two-sided diagnostic of its below-chance estimate.
Usage: python scripts/run_adjusted_inference.py [--jobs 6]
Writes results/ijdar_dev_2026-11-12/adjusted_inference/{plan.json,summary.json,writer_predictions.csv.gz,null_*.csv}.
"""
import argparse
import csv
import gzip
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed, parallel_config
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

SEEDS = list(range(2026, 2046))
FRAMING = ['ruled_h', 'ruled_v', 'has_ruled', 'line_left', 'line_right',
           'line_top', 'line_bottom', 'line_dist', 'line_ink_share',
           'block_cx', 'block_cy', 'block_w', 'block_h']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def impute(x):
    return np.where(np.isfinite(x), x, np.nanmedian(x, axis=0))


def load(root):
    base = root / 'results/final42_2026-10-19'
    dev = root / 'results/ijdar_dev_2026-11-12'
    cs = root / 'results/csafe_ijdar_2026-11-12/features'
    paths = [root / 'dataset/metadata/images.csv', base / 'features/features.npz',
             base / 'features/descriptors.csv', dev / 'features/layout.npz',
             dev / 'features/baselines.npz', cs / 'csafe_meta.csv', cs / 'csafe_hand.npz']
    hashes = {str(p.relative_to(root)): digest(p) for p in paths}
    meta = pd.read_csv(paths[0]); z = np.load(paths[1], allow_pickle=True)
    d = pd.read_csv(paths[2]).set_index('image_id').loc[meta.image_id]
    lay = np.load(paths[3], allow_pickle=True); earlier = np.load(paths[4], allow_pickle=True)
    assert list(z['image_id']) == list(lay['image_id']) == list(earlier['image_id']) == list(meta.image_id)
    frame = impute(lay['layout'][:, [list(lay['names']).index(k) for k in FRAMING]])
    ink = np.column_stack([d.ink_density.to_numpy(), z['global_intensity']])
    cm = pd.read_csv(paths[5]); h = np.load(paths[6], allow_pickle=True)
    assert list(h['image_id']) == list(cm.image_id)
    cl = impute(h['layout']); cn = list(h['layout_names'])
    ori = cl[:, [cn.index(k) for k in ['skew_deg', 'line_slope_mean', 'spacing_rel']]]
    scan = np.column_stack([h['ink_density'][:, 0], h['global_intensity']])
    configs = [
        ('bh_ink', meta, lay['hc183_textblock'], ink, .6297619047619046, True),
        ('bh_framing', meta, lay['hc183_textblock'], np.column_stack([ink, frame]), .55, True),
        ('lbp_ink', meta, earlier['lbp'], ink, .3869047619047619, True),
        ('cs_ink', cm, h['hc183_tb'], scan, .6578125, True),
        ('cs_all', cm, h['hc183_tb'], np.column_stack([cl, scan]), .6333333333333334, True),
        ('cs_orientation', cm, h['hc183_tb'], ori, .6640625, False),
        ('bh_whole', meta, z['hc183'], ink, .6452380952380952, False),
    ]
    cases = []
    for name, m, x, cov, expected, test in configs:
        # Preserve CSAFE's later-session pages; BanglaHand only uses shared text.
        if name.startswith('cs_'):
            keep = np.ones(len(m), dtype=bool)
        else:
            keep = m.paragraph_number.to_numpy() <= 6
        mm = m.loc[keep]
        writers, group = np.unique(mm.writer_id.to_numpy(), return_inverse=True)
        labels = mm.groupby('writer_id').label.first().loc[writers].to_numpy(int)
        assert set(labels) == {0, 1}
        cases.append(dict(name=name, X=np.asarray(x[keep], dtype=float),
                          cov=np.asarray(cov[keep], dtype=float), writers=writers,
                          group=group, labels=labels,
                          para=mm.paragraph_number.to_numpy(), expected=expected, test=test))
    return cases, hashes


def predict(case, labels, seed):
    group, para = case['group'], case['para']
    folds = np.empty(len(labels), dtype=int)
    for f, (_, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(case['writers'], labels)):
        folds[test] = f
    image_fold = folds[group]; image_labels = labels[group]
    scores = np.full(len(group), np.nan)
    for f in range(5):
        train = (para <= 6) & (image_fold != f); test = image_fold == f
        reg = LinearRegression().fit(case['cov'][train], case['X'][train])
        xt = case['X'][train] - reg.predict(case['cov'][train])
        xv = case['X'][test] - reg.predict(case['cov'][test])
        model = make_pipeline(StandardScaler(), SVC(kernel='linear', C=.1))
        model.fit(xt, image_labels[train]); scores[test] = model.decision_function(xv)
    out = []
    conditions = [('A', para <= 6)]
    if case['name'].startswith('cs_'):
        conditions.append(('X', para >= 7))
    for condition, keep in conditions:
        # Match the original pandas groupby writer-mean decision rule exactly.
        margins = pd.DataFrame(dict(w=group[keep], s=scores[keep])).groupby('w').s.mean().reindex(range(len(labels))).to_numpy()
        out.append((condition, margins, (margins >= 0).astype(int)))
    return folds, out


def ba(pred, labels):
    return ((pred[:, labels == 1] == 1).mean() + (pred[:, labels == 0] == 0).mean()) / 2


def permutation(case, labels):
    preds = [predict(case, labels, seed)[1] for seed in SEEDS]
    return [ba(np.array([p[j][2] for p in preds]), labels) for j in range(len(preds[0]))]


def interval(pred, labels):
    correct = (pred == labels[None, :]).mean(axis=0)
    left, right = correct[labels == 1], correct[labels == 0]
    rng = np.random.default_rng(7)
    # Same resample sequence as the original class-stratified writer bootstrap.
    vals = [(rng.choice(left, len(left)).mean() + rng.choice(right, len(right)).mean()) / 2
            for _ in range(2000)]
    return np.percentile(vals, [2.5, 97.5]).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis-root', default=Path(__file__).resolve().parents[1], type=Path)
    ap.add_argument('--output-dir', default=None, type=Path)
    ap.add_argument('--permutations', type=int, default=999)
    ap.add_argument('--jobs', type=int, default=6)
    a = ap.parse_args(); root = a.analysis_root.resolve()
    out = (a.output_dir or root / 'results/ijdar_dev_2026-11-12/adjusted_inference').resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases, hashes = load(root)
    plan = dict(status='Post hoc inference for the adjusted estimates (sensitivity analyses), not a preregistration.',
                seeds=SEEDS, permutations=a.permutations, permutation_seed=2029,
                bootstrap_resamples=2000, bootstrap_seed=7,
                family='Holm correction over the six one-sided adjusted HC-183 tests (BanglaHand A: ink+photo, ink+photo+framing; CSAFE A and X: ink+scan, all layout+ink+scan); the LBP case is a separate two-sided diagnostic.',
                inputs_sha256=hashes, script_sha256=digest(Path(__file__)),
                environment=dict(python=platform.python_version(), numpy=np.__version__,
                                 pandas=pd.__version__, sklearn=sklearn.__version__))
    (out / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    records=[]; prediction_rows=[]
    start=time.monotonic()
    with threadpool_limits(limits=1):
        for c in cases:
            begin=time.monotonic(); observed=[predict(c,c['labels'],s) for s in SEEDS]
            observed_ba=ba(np.array([o[1][0][2] for o in observed]), c['labels'])
            assert abs(observed_ba-c['expected']) < 1e-12, (c['name'], observed_ba,c['expected'])
            for j, (condition, _, _) in enumerate(observed[0][1]):
                pred=np.array([o[1][j][2] for o in observed]); lo,hi=interval(pred,c['labels'])
                records.append(dict(case=c['name'], condition=condition, ba=ba(pred,c['labels']),
                                    ci_low=lo,ci_high=hi,recall_left=float((pred[:,c['labels']==1]==1).mean()),
                                    recall_right=float((pred[:,c['labels']==0]==0).mean()), p=None))
                for s, o in zip(SEEDS, observed):
                    for k,w in enumerate(c['writers']):
                        prediction_rows.append(dict(case=c['name'],condition=condition,seed=s,writer_id=w,
                                                    label=int(c['labels'][k]),fold=int(o[0][k]),
                                                    margin=float(o[1][j][1][k]),pred=int(o[1][j][2][k])))
            print(c['name'],'reproduced:',round(observed_ba,6),'observed runtime:',round(time.monotonic()-begin,2),'s',flush=True)
    with gzip.open(out/'writer_predictions.csv.gz','wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(prediction_rows[0]));w.writeheader();w.writerows(prediction_rows)
    (out/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
    if a.permutations:
        for c in [c for c in cases if c['test']]:
            rng=np.random.default_rng(2029); labels=[rng.permutation(c['labels']) for _ in range(a.permutations)]
            null=[]
            with parallel_config(backend='loky',inner_max_num_threads=1):
                for first in range(0,len(labels),50):
                    null.extend(Parallel(n_jobs=a.jobs)(delayed(permutation)(c,l) for l in labels[first:first+50]))
                    print(c['name'],'permutations',len(null),'elapsed',round(time.monotonic()-start,1),'s',flush=True)
            null=np.asarray(null);np.savetxt(out/f'null_{c["name"]}.csv',null,delimiter=',',header='A,X' if null.shape[1]==2 else 'A',comments='')
            for j,condition in enumerate(['A','X'][:null.shape[1]]):
                rec=next(r for r in records if r['case']==c['name'] and r['condition']==condition)
                upper=(1+np.sum(null[:,j]>=rec['ba']-1e-14))/(1+len(null))
                lower=(1+np.sum(null[:,j]<=rec['ba']+1e-14))/(1+len(null))
                rec.update(p=float(min(1,2*min(upper,lower))) if c['name']=='lbp_ink' else float(upper),
                           p_upper=float(upper),p_lower=float(lower),null_mean=float(null[:,j].mean()))
            (out/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
        tested=[r for r in records if r['p'] is not None and r['case']!='lbp_ink'];assert len(tested)==6
        last=0.
        for i,r in enumerate(sorted(tested,key=lambda r:r['p'])):
            last=max(last,min(1.,(len(tested)-i)*r['p']));r['holm_p']=last
        (out/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
    assert hashes=={name:digest(root/name) for name in hashes},'An analysis input changed.'
    print('Finished; all input checksums unchanged.',flush=True)


if __name__=='__main__':
    main()
