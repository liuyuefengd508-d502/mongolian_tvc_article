#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train/evaluate a torchvision Faster R-CNN baseline for page-level TextColumn detection.

This script reads the existing YOLO-format page_level_yolo_dataset and evaluates with
the same project-level metrics used by the rule/YOLO comparison.
"""
from __future__ import annotations

import argparse, csv, json, time, random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

from evaluate_layout import evaluate_page, aggregate_page_metrics
from compare_yolo_rule_gt import load_annotations, assign_reading_order
from export_yolo_dataset import safe_stem


def yolo_to_xyxy(line: str, w: int, h: int) -> list[float]:
    parts = line.strip().split()
    _, cx, cy, bw, bh = map(float, parts[:5])
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return [max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)]


class YoloPageDataset(Dataset):
    def __init__(self, root: Path, split: str, max_side: int = 1280, limit: int = 0):
        self.root = root
        self.split = split
        self.max_side = max_side
        self.images = sorted((root / 'images' / split).glob('*.jpg'))
        if limit:
            self.images = self.images[:limit]

    def __len__(self): return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        scale = min(1.0, self.max_side / max(orig_w, orig_h))
        if scale < 1.0:
            image = image.resize((int(orig_w * scale), int(orig_h * scale)), Image.BILINEAR)
        w, h = image.size
        label_path = self.root / 'labels' / self.split / f'{img_path.stem}.txt'
        boxes=[]
        if label_path.exists():
            for line in label_path.read_text(encoding='utf-8').splitlines():
                if not line.strip(): continue
                box = yolo_to_xyxy(line, orig_w, orig_h)
                box = [v * scale for v in box]
                if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
                    boxes.append(box)
        target={
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.ones((len(boxes),), dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'orig_size': torch.tensor([orig_h, orig_w]),
            'scale': torch.tensor([scale], dtype=torch.float32),
            'file_name': img_path.name,
        }
        return TF.to_tensor(image), target


def collate(batch): return tuple(zip(*batch))


def build_model(num_classes: int = 2, pretrained_coco: bool = False):
    if pretrained_coco:
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
    return model


def train(args):
    torch.manual_seed(args.seed); random.seed(args.seed)
    device=torch.device(args.device)
    ds=YoloPageDataset(args.dataset_root,'train_unlabeled',args.max_side,args.limit_train)
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0,collate_fn=collate)
    model=build_model(pretrained_coco=args.pretrained_coco).to(device)
    opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=args.lr,momentum=0.9,weight_decay=1e-4)
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.0; n=0; t0=time.perf_counter()
        for images,targets in dl:
            images=[im.to(device) for im in images]
            targets=[{k:(v.to(device) if hasattr(v,'to') else v) for k,v in t.items() if k not in ('file_name','orig_size','scale')} for t in targets]
            loss_dict=model(images,targets)
            loss=sum(loss_dict.values())
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach().cpu()); n += 1
        row={'epoch':epoch,'loss':total/max(1,n),'seconds':time.perf_counter()-t0}
        history.append(row); print(json.dumps(row,ensure_ascii=False))
        torch.save({'model':model.state_dict(),'args':vars(args),'history':history}, out/'last.pt')
    (out/'train_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')


def infer_dataset(model, ds, device, score_threshold: float):
    dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate)
    model.eval(); out={}; times=[]
    with torch.no_grad():
        for images,targets in dl:
            img=images[0].to(device); meta=targets[0]
            scale=float(meta['scale'][0])
            t0=time.perf_counter(); pred=model([img])[0]; times.append(time.perf_counter()-t0)
            dets=[]
            for box,score,label in zip(pred['boxes'].cpu().tolist(), pred['scores'].cpu().tolist(), pred['labels'].cpu().tolist()):
                if int(label)!=1 or float(score)<score_threshold: continue
                # scale back to original coordinate space
                box=[float(v)/scale for v in box]
                dets.append({'bbox':box,'score':float(score),'method':'fasterrcnn_mobilenet','orientation':'correct'})
            out[Path(meta['file_name']).stem]=dets
    return out,times


def evaluate(args):
    device=torch.device(args.device)
    ckpt=torch.load(args.checkpoint,map_location='cpu', weights_only=False)
    model=build_model(pretrained_coco=False); model.load_state_dict(ckpt['model']); model.to(device)
    ds=YoloPageDataset(args.dataset_root,args.split,args.max_side,args.limit_eval)
    pred_by_stem,times=infer_dataset(model,ds,device,args.score_threshold)
    gt=load_annotations(args.annotations,args.split)
    stem_to_pid={safe_stem(pid):pid for pid in gt}
    page_metrics=[]; pred_json={}
    for stem,dets in pred_by_stem.items():
        pid=stem_to_pid.get(stem,stem)
        pred_json[pid]=dets
    for pid,page in gt.items():
        dets=assign_reading_order(pred_json.get(pid,[]),page)
        m=evaluate_page(dets,page.get('columns',[]),iou_threshold=args.iou_threshold)
        m.update({'page_id':pid,'method':'fasterrcnn_mobilenet','split':args.split})
        page_metrics.append(m)
    summary=aggregate_page_metrics(page_metrics)
    summary.update({'score_threshold':args.score_threshold,'iou_threshold':args.iou_threshold,'mean_seconds_per_page':sum(times)/len(times) if times else None})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/f'fasterrcnn_{args.split}_predictions.json').write_text(json.dumps(pred_json,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output_dir/f'fasterrcnn_{args.split}_metrics.json').write_text(json.dumps({'summary':summary,'pages':page_metrics},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['train','eval'],required=True)
    p.add_argument('--dataset-root',type=Path,default=Path('page_level_ocr/page_level_yolo_dataset'))
    p.add_argument('--annotations',type=Path,default=Path('page_level_ocr/page_level_annotations.json'))
    p.add_argument('--output-dir',type=Path,default=Path('page_level_ocr/results/fasterrcnn_baseline'))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--split',default='test')
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--lr',type=float,default=0.005)
    p.add_argument('--max-side',type=int,default=1280)
    p.add_argument('--score-threshold',type=float,default=0.35)
    p.add_argument('--iou-threshold',type=float,default=0.5)
    p.add_argument('--limit-train',type=int,default=0)
    p.add_argument('--limit-eval',type=int,default=0)
    p.add_argument('--device',default='cpu')
    p.add_argument('--seed',type=int,default=20260510)
    p.add_argument('--pretrained-coco', action='store_true')
    args=p.parse_args()
    if args.mode=='train': train(args)
    else: evaluate(args)

if __name__=='__main__': main()
