#!/usr/bin/env python3
from __future__ import annotations
import csv,json,statistics
from pathlib import Path

def stats(vals):
    vals=list(vals)
    return {'min':min(vals),'max':max(vals),'mean':sum(vals)/len(vals),'median':statistics.median(vals)} if vals else {'min':0,'max':0,'mean':0,'median':0}

def is_rot(page):
    s=' '.join(str(page.get(k,'')) for k in ['image_path','notes']).lower()
    return 'rot' in s

def main():
    base=Path('/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr')
    ann=base/'page_level_annotations.json'
    out=base/'results/reading_order_audit_20260512/dataset_stats_after_project16_merge'
    out.mkdir(parents=True,exist_ok=True)
    pages=json.load(open(ann,encoding='utf-8'))['pages']
    rows=[]; widths=[]; heights=[]; cols_per=[]; ign_per=[]; cws=[]; chs=[]
    for p in pages:
        valid=[c for c in p['columns'] if not c.get('ignore')]
        ign=[c for c in p['columns'] if c.get('ignore')]
        widths.append(p['width']); heights.append(p['height']); cols_per.append(len(valid)); ign_per.append(len(ign))
        for c in valid:
            x1,y1,x2,y2=c['bbox']; cws.append(x2-x1); chs.append(y2-y1)
        rows.append({'page_id':p['page_id'],'split':p['split'],'width':p['width'],'height':p['height'],'valid_columns':len(valid),'ignore_regions':len(ign),'rotated_or_corrected':is_rot(p),'mean_col_width':sum((c['bbox'][2]-c['bbox'][0]) for c in valid)/len(valid) if valid else 0,'mean_col_height':sum((c['bbox'][3]-c['bbox'][1]) for c in valid)/len(valid) if valid else 0})
    summary={'pages':len(pages),'valid_text_columns':sum(cols_per),'ignore_regions':sum(ign_per),'rotated_or_corrected_pages':sum(1 for p in pages if is_rot(p)),'pages_with_ignore':sum(1 for n in ign_per if n>0),'page_width_px':stats(widths),'page_height_px':stats(heights),'columns_per_page':stats(cols_per),'ignore_regions_per_page':stats(ign_per),'column_width_px':stats(cws),'column_height_px':stats(chs)}
    (out/'dataset_statistics_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'page_dataset_statistics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
