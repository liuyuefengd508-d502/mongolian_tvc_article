#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare a 5-page independent annotation subset for IAA measurement."""
from __future__ import annotations
import json, csv
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
ANNOTATIONS = ROOT / 'page_level_annotations.json'
MEDIA_DIR = ROOT / 'label_studio_media' / 'iaa_png'
IMPORT_DIR = ROOT / 'label_studio_import'
SELECTED = [
    ('80-48-70-2', 'ordinary_multi_column_success_case'),
    ('80-48-62-1(1)', 'rotated_page'),
    ('80-48-69-4', 'few_column_page'),
    ('80-48-72-2', 'many_ignore_regions_noise'),
    ('80-48-73-1', 'long_column_overmerge_failure_case'),
]


def main():
    data = json.loads(ANNOTATIONS.read_text(encoding='utf-8'))
    pages_by_id = {p['page_id']: p for p in data['pages']}
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    tasks=[]
    rows=[]
    for page_id, reason in SELECTED:
        page = pages_by_id[page_id]
        src = Path(page['image_path'])
        out_name = page_id.replace('/', '_').replace('(', '_').replace(')', '_') + '.png'
        out_path = MEDIA_DIR / out_name
        im = Image.open(src)
        im.save(out_path)
        valid = sum(1 for c in page.get('columns',[]) if not c.get('ignore'))
        ignore = sum(1 for c in page.get('columns',[]) if c.get('ignore'))
        rows.append({'page_id':page_id,'reason':reason,'split':page.get('split'),'valid_columns_primary':valid,'ignore_regions_primary':ignore,'image_path':str(out_path),'width':page['width'],'height':page['height']})
        tasks.append({
            'data': {
                'image': f'/data/local-files/?d=iaa_png/{out_name}',
                'page_id': page_id,
                'split': page.get('split'),
                'iaa_reason': reason,
                'png_path': str(out_path),
                'original_image_path': page.get('original_image_path', page.get('image_path')),
                'width': page['width'],
                'height': page['height'],
            }
            # deliberately no predictions: second annotator must annotate independently
        })
    (IMPORT_DIR/'iaa_5page_tasks.json').write_text(json.dumps(tasks,ensure_ascii=False,indent=2),encoding='utf-8')
    with (IMPORT_DIR/'iaa_5page_selected_pages.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(json.dumps({'tasks':len(tasks),'task_file':str(IMPORT_DIR/'iaa_5page_tasks.json'),'media_dir':str(MEDIA_DIR),'selected':rows},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
