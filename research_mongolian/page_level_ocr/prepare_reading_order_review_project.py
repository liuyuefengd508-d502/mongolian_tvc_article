#!/usr/bin/env python3
from __future__ import annotations
import json, shutil, sqlite3, sys
from pathlib import Path
from PIL import Image
import requests

BASE=Path('/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr')
ANN=BASE/'page_level_annotations.json'
MEDIA=BASE/'label_studio_media'/'reading_order_review_62_jpg'
IMPORT=BASE/'label_studio_import'
TASKS=IMPORT/'reading_order_review_62_tasks_with_annotations.json'
SUMMARY=BASE/'results'/'reading_order_audit_20260512'/'labelstudio_review_project_summary.json'
SQLITE=BASE/'label_studio_data'/'label_studio.sqlite3'
API='http://localhost:8080/api'
PROJECT_TITLE='Mongolian OCR Reading-order Review - old 62 pages'
LABEL_CONFIG='''<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="TextColumn" background="#1f77b4"/>
    <Label value="Ignore" background="#d62728"/>
  </RectangleLabels>
  <Header value="Reading order"/>
  <TextArea name="reading_order" toName="image" perRegion="true" editable="true" rows="1"/>
  <Header value="Orientation"/>
  <Choices name="orientation" toName="image" perRegion="true" choice="single">
    <Choice value="correct"/>
    <Choice value="rotated_90_ccw"/>
    <Choice value="rotated_90_cw"/>
    <Choice value="rotated_180"/>
    <Choice value="ambiguous"/>
  </Choices>
  <Header value="Degradation tags"/>
  <Choices name="degradation_tags" toName="image" perRegion="true" choice="multiple">
    <Choice value="severe_fade"/>
    <Choice value="bleed_through"/>
    <Choice value="red_seal"/>
    <Choice value="fold"/>
    <Choice value="stain"/>
    <Choice value="broken_spine"/>
    <Choice value="dense_background"/>
    <Choice value="marginalia"/>
    <Choice value="overlap"/>
    <Choice value="other"/>
  </Choices>
</View>'''

def token():
    conn=sqlite3.connect(f'file:{SQLITE}?mode=ro', uri=True)
    row=conn.execute('select key from authtoken_token limit 1').fetchone()
    if not row: raise RuntimeError('No Label Studio API token found')
    return row[0]

def pct_bbox(bbox,w,h):
    x1,y1,x2,y2=bbox
    return {'x':x1/w*100,'y':y1/h*100,'width':(x2-x1)/w*100,'height':(y2-y1)/h*100,'rotation':0}

def make_region(page, col):
    rid=col['column_id']
    w,h=page['width'],page['height']
    base=pct_bbox(col['bbox'],w,h)
    label='Ignore' if col.get('ignore') else 'TextColumn'
    out=[{'id':rid,'from_name':'label','to_name':'image','type':'rectanglelabels','value':{**base,'rectanglelabels':[label]}}]
    if not col.get('ignore'):
        out.append({'id':rid,'from_name':'reading_order','to_name':'image','type':'textarea','value':{**base,'text':[str(col.get('reading_order',''))]}})
        out.append({'id':rid,'from_name':'orientation','to_name':'image','type':'choices','value':{**base,'choices':[col.get('orientation','correct')]}})
        tags=col.get('degradation_tags') or []
        if tags:
            out.append({'id':rid,'from_name':'degradation_tags','to_name':'image','type':'choices','value':{**base,'choices':tags}})
    return out

def ensure_jpg(page):
    src=Path(page['image_path'])
    if not src.exists(): src=Path(page.get('original_image_path',''))
    if not src.exists(): raise FileNotFoundError(page['page_id'])
    out=MEDIA/f"{page['page_id']}.jpg"
    if not out.exists():
        im=Image.open(src).convert('RGB')
        im.save(out, quality=92)
    return out

def build_tasks():
    data=json.load(open(ANN,encoding='utf-8'))
    MEDIA.mkdir(parents=True, exist_ok=True); IMPORT.mkdir(parents=True, exist_ok=True); SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    split_order={'test':0,'val':1,'train_unlabeled':2}
    pages=sorted(data['pages'], key=lambda p:(split_order.get(p['split'],9), p['page_id']))
    tasks=[]
    for idx,page in enumerate(pages, start=1):
        jpg=ensure_jpg(page)
        result=[]
        for col in page.get('columns',[]): result.extend(make_region(page,col))
        tasks.append({
            'data':{
                'image':f"http://localhost:8090/reading_order_review_62_jpg/{jpg.name}",
                'page_id':page['page_id'], 'split':page['split'], 'review_index':idx,
                'source':'old_62_reading_order_review', 'original_image_path':page.get('original_image_path',''),
                'current_annotation_image_path':page.get('image_path',''), 'width':page['width'], 'height':page['height'],
                'notes':page.get('notes','') or ''
            },
            'annotations':[{'result':result, 'ground_truth':False, 'was_cancelled':False}]
        })
    TASKS.write_text(json.dumps(tasks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return tasks

def api_session():
    s=requests.Session(); s.headers.update({'Authorization':f'Token {token()}'})
    return s

def find_project(s):
    r=s.get(f'{API}/projects', timeout=20); r.raise_for_status(); data=r.json()
    projects=data.get('results',data) if isinstance(data,dict) else data
    for p in projects:
        if p.get('title')==PROJECT_TITLE: return p
    return None

def create_project(s):
    p=find_project(s)
    if p: return p
    r=s.post(f'{API}/projects', json={'title':PROJECT_TITLE,'label_config':LABEL_CONFIG}, timeout=20); r.raise_for_status(); return r.json()

def import_tasks(s, project_id, tasks):
    # If project already has tasks, keep them to avoid deleting user's edits.
    r=s.get(f'{API}/projects/{project_id}', timeout=20); r.raise_for_status(); info=r.json()
    task_count=info.get('task_number') or info.get('tasks_count') or 0
    if task_count:
        return {'skipped_import_existing_tasks':task_count}
    r=s.post(f'{API}/projects/{project_id}/import', json=tasks, timeout=60); r.raise_for_status(); return r.json()

def main():
    tasks=build_tasks(); s=api_session(); p=create_project(s); imp=import_tasks(s,p['id'],tasks)
    url=f'http://localhost:8080/projects/{p["id"]}/data?tab=1'
    summary={'project_id':p['id'],'project_title':PROJECT_TITLE,'project_url':url,'tasks_file':str(TASKS),'media_dir':str(MEDIA),'task_count_prepared':len(tasks),'import_result':imp}
    SUMMARY.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
