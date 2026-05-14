#!/usr/bin/env python3
"""Export Label Studio project 16 reviewed old-62 annotations to page_level_annotations schema."""
from __future__ import annotations
import json, sqlite3, shutil
from pathlib import Path
from datetime import date
from typing import Any

BASE=Path('/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr')
SQLITE=BASE/'label_studio_data/label_studio.sqlite3'
CURRENT=BASE/'page_level_annotations.json'
OUT=BASE/'page_level_annotations.project16_reviewed_20260512.json'
BACKUP=BASE/'page_level_annotations.before_project16_review_merge_20260512.bak'
SUMMARY=BASE/'results/reading_order_audit_20260512/project16_merge_summary.json'
PROJECT_ID=16

VALID_ORIENTATIONS={"correct","rotated_90_ccw","rotated_90_cw","rotated_180","ambiguous"}

def pct_to_bbox(value:dict[str,Any], width:int, height:int)->list[int]:
    x1=float(value['x'])/100*width; y1=float(value['y'])/100*height
    x2=x1+float(value['width'])/100*width; y2=y1+float(value['height'])/100*height
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

def parse_annotation_result(result:list[dict[str,Any]], page:dict[str,Any])->list[dict[str,Any]]:
    regions:dict[str,dict[str,Any]]={}
    for item in result:
        rid=item.get('id')
        if not rid: continue
        typ=item.get('type'); fn=item.get('from_name'); val=item.get('value',{}) or {}
        rec=regions.setdefault(rid, {'id':rid, 'label':None, 'bbox':None, 'reading_order':None, 'orientation':None, 'degradation_tags':[]})
        if typ=='rectanglelabels' and fn=='label':
            labs=val.get('rectanglelabels') or []
            rec['label']=labs[0] if labs else None
            rec['bbox']=pct_to_bbox(val, int(page['width']), int(page['height']))
        elif typ=='textarea' and fn=='reading_order':
            text=(val.get('text') or [''])[0]
            try: rec['reading_order']=int(text)
            except Exception: rec['reading_order']=None
        elif typ=='choices' and fn=='orientation':
            choices=val.get('choices') or []
            rec['orientation']=choices[0] if choices else None
        elif typ=='choices' and fn=='degradation_tags':
            rec['degradation_tags']=val.get('choices') or []
    columns=[]
    for rid,rec in regions.items():
        if rec['label'] not in {'TextColumn','Ignore'} or rec['bbox'] is None:
            continue
        ignore=rec['label']=='Ignore'
        orientation=rec['orientation'] or 'correct'
        if orientation not in VALID_ORIENTATIONS:
            orientation='ambiguous'
        columns.append({
            'column_id': rid,
            'bbox': rec['bbox'],
            'reading_order': rec['reading_order'] if not ignore else None,
            'orientation': orientation,
            'transcript': '',
            'degradation_tags': rec.get('degradation_tags') or [],
            'ignore': ignore,
            'notes': 'merged from Label Studio Project 16 reading-order review on 2026-05-12',
        })
    # Keep TextColumns in reading_order order and Ignore regions after them for readability.
    columns.sort(key=lambda c: (1 if c.get('ignore') else 0, c.get('reading_order') if isinstance(c.get('reading_order'), int) else 10**9, c['bbox'][0], c['bbox'][1]))
    return columns

def validate_pages(pages:list[dict[str,Any]])->list[str]:
    issues=[]
    for p in pages:
        valid=[c for c in p['columns'] if not c.get('ignore')]
        ros=[c.get('reading_order') for c in valid]
        if any(not isinstance(x,int) or isinstance(x,bool) for x in ros):
            issues.append(f"{p['page_id']}: non-int reading_order {ros}")
        if sorted(ros)!=list(range(len(valid))):
            issues.append(f"{p['page_id']}: non-consecutive reading_order {sorted(ros)} expected {list(range(len(valid)))}")
        for c in p['columns']:
            x1,y1,x2,y2=c['bbox']
            if not (0 <= x1 < x2 <= p['width'] and 0 <= y1 < y2 <= p['height']):
                issues.append(f"{p['page_id']} {c['column_id']}: bbox out of bounds {c['bbox']} page={p['width']}x{p['height']}")
    return issues

def main():
    conn=sqlite3.connect(f'file:{SQLITE}?mode=ro', uri=True); conn.row_factory=sqlite3.Row
    rows=list(conn.execute('''select t.id task_id,t.data,a.id ann_id,a.result,a.updated_at,a.was_cancelled
from task t left join task_completion a on a.task_id=t.id
where t.project_id=? order by t.id''',(PROJECT_ID,)))
    if len(rows)!=62:
        raise SystemExit(f'Expected 62 tasks, found {len(rows)}')
    current=json.load(open(CURRENT,encoding='utf-8'))
    current_by_id={p['page_id']:p for p in current['pages']}
    reviewed_pages=[]; latest=''
    for row in rows:
        if not row['ann_id'] or row['was_cancelled']:
            raise SystemExit(f"Task {row['task_id']} missing/cancelled annotation")
        data=json.loads(row['data'])
        page_id=data['page_id']
        old=current_by_id.get(page_id, {})
        page={
            'page_id': page_id,
            'image_path': data.get('current_annotation_image_path') or data.get('image') or old.get('image_path',''),
            'original_image_path': data.get('original_image_path') or old.get('original_image_path',''),
            'split': data.get('split') or old.get('split'),
            'width': int(data.get('width') or old.get('width')),
            'height': int(data.get('height') or old.get('height')),
            'columns': [],
            'notes': old.get('notes') or data.get('notes') or '',
        }
        result=json.loads(row['result']) if isinstance(row['result'],str) else row['result']
        page['columns']=parse_annotation_result(result,page)
        reviewed_pages.append(page)
        latest=max(latest, row['updated_at'] or '')
    issues=validate_pages(reviewed_pages)
    if issues:
        raise SystemExit('Validation failed before writing:\n'+'\n'.join(issues[:30]))
    reviewed_by_id={p['page_id']:p for p in reviewed_pages}
    merged=dict(current)
    merged['pages']=[reviewed_by_id.get(p['page_id'],p) for p in current['pages']]
    ds=merged.setdefault('dataset',{})
    note='Project 16 reading-order review merged on 2026-05-12; reviewed 62 old pages in Label Studio.'
    ds['notes']=(ds.get('notes','')+'; ' if ds.get('notes') else '')+note
    OUT.write_text(json.dumps(merged,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if not BACKUP.exists(): shutil.copy2(CURRENT,BACKUP)
    CURRENT.write_text(json.dumps(merged,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    old_valid=sum(1 for p in current['pages'] for c in p['columns'] if not c.get('ignore'))
    old_ignore=sum(1 for p in current['pages'] for c in p['columns'] if c.get('ignore'))
    new_valid=sum(1 for p in merged['pages'] for c in p['columns'] if not c.get('ignore'))
    new_ignore=sum(1 for p in merged['pages'] for c in p['columns'] if c.get('ignore'))
    changed=[]
    for p in reviewed_pages:
        old=current_by_id[p['page_id']]
        ov=sum(1 for c in old['columns'] if not c.get('ignore')); oi=sum(1 for c in old['columns'] if c.get('ignore'))
        nv=sum(1 for c in p['columns'] if not c.get('ignore')); ni=sum(1 for c in p['columns'] if c.get('ignore'))
        if (ov,oi)!=(nv,ni): changed.append({'page_id':p['page_id'],'old_text':ov,'old_ignore':oi,'new_text':nv,'new_ignore':ni})
    summary={'project_id':PROJECT_ID,'tasks':len(rows),'latest_annotation_update':latest,'reviewed_output':str(OUT),'backup':str(BACKUP),'merged_annotation':str(CURRENT),'old_valid_text_columns':old_valid,'old_ignore_regions':old_ignore,'new_valid_text_columns':new_valid,'new_ignore_regions':new_ignore,'pages_with_count_changes':changed}
    SUMMARY.parent.mkdir(parents=True,exist_ok=True)
    SUMMARY.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
