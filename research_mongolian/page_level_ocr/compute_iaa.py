#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute inter-annotator agreement for page-level TextColumn annotations.

Expected input: two project-format JSON files with a top-level `pages` list.
The primary file is the current annotation; the secondary file should contain
independent annotations for the same page IDs.
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_layout import evaluate_page, greedy_match, pairwise_order_accuracy


def load_pages(path: Path) -> dict[str, dict]:
    data=json.loads(path.read_text(encoding='utf-8'))
    pages=data.get('pages', data if isinstance(data,list) else [])
    return {p['page_id']:p for p in pages}


def eval_ignore(primary_cols, secondary_cols, iou_threshold=0.5):
    gt=[c for c in primary_cols if c.get('ignore')]
    pred=[c for c in secondary_cols if c.get('ignore')]
    matches=greedy_match([p['bbox'] for p in pred],[g['bbox'] for g in gt],iou_threshold)
    tp=len(matches); fp=len(pred)-tp; fn=len(gt)-tp
    return {
        'ignore_tp':tp,'ignore_fp':fp,'ignore_fn':fn,
        'ignore_precision':tp/(tp+fp) if tp+fp else 0.0,
        'ignore_recall':tp/(tp+fn) if tp+fn else 0.0,
        'ignore_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
        'ignore_mean_iou':sum(m.iou for m in matches)/tp if tp else 0.0,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--primary',type=Path,default=Path('page_level_ocr/page_level_annotations.json'))
    ap.add_argument('--secondary',type=Path,required=True)
    ap.add_argument('--pages-csv',type=Path,default=Path('page_level_ocr/label_studio_import/iaa_5page_selected_pages.csv'))
    ap.add_argument('--output-dir',type=Path,default=Path('page_level_ocr/results/iaa_5page'))
    ap.add_argument('--iou-threshold',type=float,default=0.5)
    args=ap.parse_args()
    primary=load_pages(args.primary); secondary=load_pages(args.secondary)
    page_ids=[]
    with args.pages_csv.open(encoding='utf-8') as f:
        for row in csv.DictReader(f): page_ids.append(row['page_id'])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    totals={'tp':0,'fp':0,'fn':0,'ignore_tp':0,'ignore_fp':0,'ignore_fn':0}
    ious=[]; ro=[]; ignore_ious=[]
    for pid in page_ids:
        p=primary[pid]; s=secondary[pid]
        # secondary valid boxes are predictions; primary valid boxes are GT
        preds=[{**c,'score':1.0} for c in s.get('columns',[]) if not c.get('ignore')]
        gt=[c for c in p.get('columns',[]) if not c.get('ignore')]
        metric=evaluate_page(preds, gt, args.iou_threshold)
        ign=eval_ignore(p.get('columns',[]), s.get('columns',[]), args.iou_threshold)
        row={'page_id':pid,'primary_valid':len(gt),'secondary_valid':len(preds),**{k:metric[k] for k in ['precision','recall','f1','mean_iou','reading_order_accuracy','true_positive','false_positive','false_negative']},**ign}
        rows.append(row)
        totals['tp']+=metric['true_positive']; totals['fp']+=metric['false_positive']; totals['fn']+=metric['false_negative']
        totals['ignore_tp']+=ign['ignore_tp']; totals['ignore_fp']+=ign['ignore_fp']; totals['ignore_fn']+=ign['ignore_fn']
        ious.extend(m['iou'] for m in metric.get('matches',[]))
        if metric.get('reading_order_accuracy') is not None: ro.append(metric['reading_order_accuracy'])
        if ign['ignore_tp']: ignore_ious.append(ign['ignore_mean_iou'])
    tp,fp,fn=totals['tp'],totals['fp'],totals['fn']
    itp,ifp,ifn=totals['ignore_tp'],totals['ignore_fp'],totals['ignore_fn']
    summary={
        'pages':len(page_ids),'iou_threshold':args.iou_threshold,
        'box_precision':tp/(tp+fp) if tp+fp else 0.0,
        'box_recall':tp/(tp+fn) if tp+fn else 0.0,
        'box_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
        'mean_matched_iou':sum(ious)/len(ious) if ious else 0.0,
        'reading_order_pairwise_agreement':sum(ro)/len(ro) if ro else None,
        'ignore_precision':itp/(itp+ifp) if itp+ifp else 0.0,
        'ignore_recall':itp/(itp+ifn) if itp+ifn else 0.0,
        'ignore_f1':2*itp/(2*itp+ifp+ifn) if 2*itp+ifp+ifn else 0.0,
    }
    (args.output_dir/'iaa_summary.json').write_text(json.dumps({'summary':summary,'pages':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    with (args.output_dir/'iaa_per_page.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    md=['# IAA 5-page agreement summary\n',f"IoU threshold: `{args.iou_threshold}`\n",'| Metric | Value |','|---|---:|']
    for k,v in summary.items():
        if k in ('pages','iou_threshold'): continue
        md.append(f'| {k} | {v:.3f} |' if isinstance(v,(int,float)) and v is not None else f'| {k} | {v} |')
    (args.output_dir/'iaa_summary.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
