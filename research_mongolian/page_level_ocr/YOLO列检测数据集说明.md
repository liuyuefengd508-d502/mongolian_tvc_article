# YOLO 列检测数据集说明

## 目的

本目录用于把当前页面级人工标注转换为 YOLO 目标检测格式，方便后续训练轻量学习式列检测器，作为规则法 `proposed` 的 stronger baseline 或后续改进方向。

当前导出结果位于：

```text
page_level_ocr/page_level_yolo_dataset/
```

## 数据规模

当前人工标注共 62 页：

| split | pages | TextColumn boxes | ignored boxes |
| --- | ---: | ---: | ---: |
| train_unlabeled | 43 | 660 | 84 |
| val | 8 | 108 | 19 |
| test | 11 | 163 | 50 |
| total | 62 | 931 | 153 |

说明：

- `ignore=True` 的框不会导出为训练标签。
- 旋转重标页面使用标注文件中的 `image_path`，因此 YOLO 坐标与人工标注坐标系一致。
- 导出脚本会把 TIFF 等非浏览器友好格式统一转换为 JPG，方便 YOLO/Ultralytics 读取。
- 当前已有 43 页 `train_unlabeled` 标注，`data.yaml` 中使用 `images/train_unlabeled` 作为训练入口。

## 目录结构

```text
page_level_yolo_dataset/
  data.yaml
  export_stats.json
  images/
    train_unlabeled/
    val/
    test/
  labels/
    train_unlabeled/
    val/
    test/
```

YOLO 标签格式为：

```text
class_id center_x center_y width height
```

其中坐标均按图像宽高归一化到 0 到 1。

## 重新导出命令

在 `research_mongolian` 目录下运行：

```bash
python3 page_level_ocr/export_yolo_dataset.py \
  --annotations page_level_ocr/page_level_annotations.json \
  --output-dir page_level_ocr/page_level_yolo_dataset \
  --splits train_unlabeled,val,test
```

## 训练示例

当前已创建独立 YOLO 环境：

```bash
python3 -m venv page_level_ocr/.venv_yolo
page_level_ocr/.venv_yolo/bin/python -m pip install ultralytics
```

示例训练命令：

```bash
yolo detect train \
  model=yolov8n.pt \
  data=page_level_ocr/page_level_yolo_dataset/data.yaml \
  imgsz=1280 \
  epochs=100 \
  batch=2 \
  project=page_level_ocr/results/yolo_column_detector \
  name=yolov8n_val_train
```

如果使用当前独立环境，请运行：

```bash
page_level_ocr/.venv_yolo/bin/yolo detect train \
  model=yolov8n.pt \
  data=page_level_ocr/page_level_yolo_dataset/data.yaml \
  imgsz=1280 \
  epochs=100 \
  batch=2 \
  device=cpu \
  workers=0 \
  project=page_level_ocr/results/yolo_column_detector \
  name=yolov8n_val_train
```

## Smoke test 结果

已完成 1 epoch smoke train，目的仅为验证数据集格式、路径和训练流程是否可用，不用于论文指标。

环境：

```text
Ultralytics 8.4.47
Torch 2.8.0
Python 3.9.6
设备：CPU；本机检测到 MPS 可用，但 smoke test 为稳妥使用 CPU
```

训练命令：

```bash
page_level_ocr/.venv_yolo/bin/yolo detect train \
  model=yolov8n.pt \
  data=page_level_ocr/page_level_yolo_dataset/data.yaml \
  imgsz=640 \
  epochs=1 \
  batch=1 \
  device=cpu \
  workers=0 \
  project=page_level_ocr/results/yolo_column_detector \
  name=smoke_yolov8n \
  exist_ok=True \
  plots=False \
  save=False \
  val=True
```

验证结论：

```text
train split: 8 images, 108 instances, 0 corrupt
test split: 11 images, 163 instances, 0 corrupt
smoke train 可正常完成
test split 可正常验证
```

Smoke test 输出目录：

```text
runs/detect/page_level_ocr/results/yolo_column_detector/smoke_yolov8n/
runs/detect/page_level_ocr/results/yolo_column_detector/smoke_yolov8n_test/
```

注意：1 epoch 下 mAP 很低是正常现象，不能用于论文对比；它只证明数据和训练管线已经连通。

## 20 epoch 小样本试跑

已完成一个 `YOLOv8n` 20 epoch 小样本试跑，使用 8 页 `val` 作为临时训练集，并在 11 页 `test` 上验证。

训练命令：

```bash
page_level_ocr/.venv_yolo/bin/yolo detect train \
  model=yolov8n.pt \
  data=/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr/page_level_yolo_dataset/data.yaml \
  imgsz=960 \
  epochs=20 \
  batch=1 \
  device=cpu \
  workers=0 \
  project=/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr/results/yolo_column_detector \
  name=yolov8n_20ep_val_train \
  exist_ok=True \
  plots=True \
  save=True \
  val=True
```

结果汇总：

| split | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| val | 0.646 | 0.602 | 0.582 | 0.273 |
| test | 0.643 | 0.429 | 0.465 | 0.224 |

输出文件：

```text
page_level_ocr/results/yolo_column_detector/yolov8n_20ep_summary.md
page_level_ocr/results/yolo_column_detector/yolov8n_20ep_summary.json
page_level_ocr/results/yolo_column_detector/yolov8n_20ep_val_train/
page_level_ocr/results/yolo_column_detector/yolov8n_20ep_test/
```

解读：

- 该结果说明学习式检测器已经能从小样本中学到文本列位置。
- test recall 为 0.429，仍低于当前规则法 `proposed` 的 test recall 0.509，因此暂不替代规则法。
- 该结果可作为“学习式 stronger baseline 的初步可行性验证”，但不能作为论文最终主结果。

小样本情况下建议：

- 使用较小模型，如 `yolov8n.pt` 或 `yolov8s.pt`。
- 使用较大输入尺寸，如 `1280` 或 `1536`，因为文本列较细。
- 先做可行性验证，不把 19 页小样本结果作为最终论文主结果。
- 后续应继续标注更多 train pages，再固定 val/test，避免过拟合和数据泄漏。

## 论文使用建议

当前 YOLO 数据集适合用于：

1. 验证学习式检测器是否能解决规则法的长框过合并问题。
2. 作为下一阶段 stronger baseline 的数据准备。
3. 生成 preliminary result 或 failure analysis 对比。

暂不建议直接声称其为最终实验结果，因为：

- 标注页数较少。
- 当前 `val` 被临时用作训练入口。
- 测试集规模也较小，统计稳定性不足。

正式投稿前建议先扩展训练标注页数，再固定 `val/test` 做最终模型选择和报告。

## 43 页训练标注结果

已完成 `YOLOv8n` 50 epoch 训练，使用当前 43 页 `train_unlabeled` 作为训练集，并在固定 8 页 `val` 和 11 页 `test` 上评估。

训练命令：

```bash
page_level_ocr/.venv_yolo/bin/yolo detect train \
  model=yolov8n.pt \
  data=/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr/page_level_yolo_dataset/data.yaml \
  imgsz=960 \
  epochs=50 \
  batch=1 \
  device=cpu \
  workers=0 \
  project=/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr/results/yolo_column_detector \
  name=yolov8n_50ep_train43 \
  exist_ok=True \
  plots=True \
  save=True \
  val=True
```

独立 test 评估命令：

```bash
page_level_ocr/.venv_yolo/bin/yolo detect val \
  model=/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr/results/yolo_column_detector/yolov8n_50ep_train43/weights/best.pt \
  data=/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr/page_level_yolo_dataset/data.yaml \
  split=test \
  imgsz=960 \
  batch=1 \
  device=cpu \
  workers=0 \
  project=/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr/results/yolo_column_detector \
  name=yolov8n_50ep_train43_test \
  exist_ok=True \
  plots=True \
  save_json=True \
  verbose=True
```

结果汇总：

| split | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| val best | 0.813 | 0.741 | 0.772 | 0.427 |
| test | 0.818 | 0.801 | 0.825 | 0.414 |

输出文件：

```text
page_level_ocr/results/yolo_column_detector/yolov8n_50ep_train43_summary.md
page_level_ocr/results/yolo_column_detector/yolov8n_50ep_train43_summary.json
page_level_ocr/results/yolo_column_detector/yolov8n_50ep_train43/
page_level_ocr/results/yolo_column_detector/yolov8n_50ep_train43_test/
```

解读：

- 43 页训练版本是当前最强 YOLOv8n 结果。
- 与 20 页训练版本相比，test recall 从 0.712 提升到 0.801，mAP50-95 从 0.355 提升到 0.414。
- 该结果已经适合写成论文中的学习式 stronger baseline；后续仍建议补充可视化、失败案例和与规则法统一口径的 IoU@0.5 统计。

## 50 epoch 训练页试跑

已完成一个更合理的数据划分试跑：使用 7 页已经人工标注的 `train_unlabeled` 页面训练 YOLOv8n，固定 8 页 `val` 用于验证，并在 11 页 `test` 上做独立评估。

训练命令：

```bash
page_level_ocr/.venv_yolo/bin/yolo detect train \
  model=yolov8n.pt \
  data=/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr/page_level_yolo_dataset/data.yaml \
  imgsz=960 \
  epochs=50 \
  batch=1 \
  device=cpu \
  workers=0 \
  project=/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr/results/yolo_column_detector \
  name=yolov8n_50ep_train7 \
  exist_ok=True \
  plots=True \
  save=True \
  val=True
```

测试集评估命令：

```bash
page_level_ocr/.venv_yolo/bin/yolo detect val \
  model=/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr/results/yolo_column_detector/yolov8n_50ep_train7/weights/best.pt \
  data=/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr/page_level_yolo_dataset/data.yaml \
  split=test \
  imgsz=960 \
  batch=1 \
  device=cpu \
  workers=0 \
  project=/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr/results/yolo_column_detector \
  name=yolov8n_50ep_train7_test \
  exist_ok=True \
  plots=True \
  save_json=True \
  verbose=True
```

结果汇总：

| split | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| val | 0.596 | 0.574 | 0.533 | 0.202 |
| test | 0.592 | 0.453 | 0.540 | 0.215 |

输出文件：

```text
page_level_ocr/results/yolo_column_detector/yolov8n_50ep_train7_summary.md
page_level_ocr/results/yolo_column_detector/yolov8n_50ep_train7_summary.json
page_level_ocr/results/yolo_column_detector/yolov8n_50ep_train7/
page_level_ocr/results/yolo_column_detector/yolov8n_50ep_train7_test/
```

解读：

- 这次划分比 20 epoch 小样本试跑更规范，因为没有再把 `val` 当训练集。
- 7 页训练已经能让 YOLO 学到文本列大致区域，test mAP50 为 0.540。
- 但 test recall 只有 0.453，仍低于当前规则法 `proposed` 的 test recall 0.509。
- 当前不建议用 YOLO 替代规则法主方法；更适合保留为 stronger baseline，并在更多训练页标注完成后重新训练。

## 训练集扩展记录

2026-05-08 已合并 Label Studio project 6 的 13 页正常方向 `train_batch01` 标注，并跳过 7 页已在 project 7 中旋转重标的横向页面。合并前备份为：

```text
page_level_ocr/page_level_annotations.before_train_batch01_http_merge.json
```

合并后 YOLO 数据集已重新导出：

| split | pages | TextColumn boxes | ignored boxes |
| --- | ---: | ---: | ---: |
| train_unlabeled | 20 | 368 | 56 |
| val | 8 | 108 | 19 |
| test | 11 | 163 | 50 |
| total | 39 | 639 | 125 |

下一步实验应使用这 20 页训练集重新训练 YOLOv8n，并与 7 页训练结果对比。

## 50 epoch 20 页训练集结果

使用 20 页 `train_unlabeled` 训练 YOLOv8n 后，结果如下：

| split | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| val | 0.656 | 0.671 | 0.591 | 0.293 |
| test | 0.806 | 0.712 | 0.810 | 0.355 |

输出文件：

```text
page_level_ocr/results/yolo_column_detector/yolov8n_50ep_train20_summary.md
page_level_ocr/results/yolo_column_detector/yolov8n_50ep_train20_summary.json
page_level_ocr/results/yolo_column_detector/yolov8n_50ep_train20/
page_level_ocr/results/yolo_column_detector/yolov8n_50ep_train20_test/
```

与 7 页训练结果相比，test mAP50 从 0.540 提升到 0.810，test recall 从 0.453 提升到 0.712。这个结果说明扩展训练标注非常有效，YOLOv8n 已经可以作为论文中的 stronger baseline 继续发展。

## 训练集扩展到 34 页

2026-05-09 已合并 batch02 标注。batch02 中 `80-48-66-2(1)` 是中文手写档案页，已从 manifest、Label Studio 任务和后续数据集中排除。其余 14 页传统蒙古文档案已从 Label Studio projects 8、10、11 导出并合并。

合并前备份：

```text
page_level_ocr/page_level_annotations.before_train_batch02_merge.json
```

当前 YOLO 数据集：

| split | pages | TextColumn boxes | ignored boxes |
| --- | ---: | ---: | ---: |
| train_unlabeled | 34 | 527 | 70 |
| val | 8 | 108 | 19 |
| test | 11 | 163 | 50 |
| total | 53 | 798 | 139 |

下一步可以在 34 页训练集上重新训练 YOLOv8n/YOLOv8s，检查是否继续提升 test recall 和 mAP50。

## 训练集扩展到 43 页

2026-05-09 已合并 batch03 标注。batch03 中 6 页中文手写档案已排除，`80-48-69-3` 作为横向蒙古文档案已逆时针旋转 90 度后重标并合并。

合并前备份：

```text
page_level_ocr/page_level_annotations.before_train_batch03_merge.json
```

当前 YOLO 数据集：

| split | pages | TextColumn boxes | ignored boxes |
| --- | ---: | ---: | ---: |
| train_unlabeled | 43 | 660 | 84 |
| val | 8 | 108 | 19 |
| test | 11 | 163 | 50 |
| total | 62 | 931 | 153 |

下一步建议在 43 页训练集上重新训练 YOLOv8n，并与 20 页、34 页训练结果对比。

```text
train: 60-100 pages
val: 10-20 pages
test: 20-30 pages
```

并确保按 page-level 划分，不能让同一页或同一高度相似页面同时进入训练和测试。
