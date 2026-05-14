# 页面级传统蒙古文 OCR

这个子项目用于支撑下一篇“页面级历史档案版面分析”研究。由于当前缺少手写传统蒙古文专家转写，研究主线已从“完整端到端 OCR 识别评估”调整为“整页档案文本列检测、方向处理与阅读顺序恢复”。

当前目标流程如下：

1. 按页面 ID 划分整页档案图像，避免数据泄漏。
2. 标注每一页中的文本列框、阅读顺序、方向和 ignore 区域。
3. 从整页扫描图中自动检测文本列。
4. 评估列检测、阅读顺序恢复和方向校正。
5. 导出 oracle crop 和自动检测 crop，作为 recognition-ready pipeline 演示；在没有列级真实转写前，不报告 CER/WER。

## 数据泄漏规则

验证集和测试集页面绝不能用于以下操作：

- 纸张纹理提取
- SASE 参数选择
- 无监督阈值调参
- 目标域 CTC entropy 监控
- 伪标签选择

这些操作只能使用 `train_unlabeled` 中的未标注目标域训练页面。

## 文件说明

- `annotation_schema.json`：页面级标注格式，包含列框、阅读顺序、方向和退化标签。
- `annotate_pages.md`：人工标注说明文档。
- `bootstrap_annotations.py`：根据自动检测结果生成标注初稿，方便人工修订。
- `visualize_annotations.py`：把标注 JSON 渲染成带彩色列框的页面预览图。
- `validate_annotations.py`：检查人工标注 JSON 是否存在坐标、顺序、标签或 split 错误。
- `make_page_splits.py`：从整页档案扫描图生成稳定的 page-level split manifest。
- `layout_columns.py`：列检测方法，包括 baseline 和 proposed layout-aware extraction。
- `evaluate_layout.py`：计算 IoU、Precision、Recall、F1、mIoU 和 reading-order accuracy。
- `run_layout_experiment.py`：统一实验入口，用于生成检测结果并在有标注时自动评估。
- `run_end_to_end_ocr_experiment.py`：导出 oracle/YOLO 文本列裁切并连接识别模型；当前仅作为 recognition-ready pipeline，不把无转写输出作为 CER/WER。
- `无转写替代研究方案_页面级版面分析.md`：缺少蒙古文转写专家时的投稿主线调整方案。
- `论文改写清单_从端到端OCR调整为页面级版面分析.md`：论文标题、摘要、方法、实验和 limitations 的改写清单。

## 常用流程

生成页面级划分文件：

```bash
python page_level_ocr/make_page_splits.py \
  --data-root "/Users/liuyu/Desktop/mydocuments/codes/autoResearch/data/80-48（61-80）" \
  --output page_level_ocr/page_split_manifest.csv
```

如果还没有人工标注，可以先让程序生成自动检测结果，作为检查和标注的初稿：

```bash
python page_level_ocr/bootstrap_annotations.py \
  --manifest page_level_ocr/page_split_manifest.csv \
  --output page_level_ocr/page_level_annotations.bootstrap.json \
  --method proposed \
  --split val
```

把标注初稿画到页面图上，生成便于肉眼检查的预览图：

```bash
python page_level_ocr/visualize_annotations.py \
  --annotations page_level_ocr/page_level_annotations.bootstrap.json \
  --output-dir page_level_ocr/annotation_previews/bootstrap_val \
  --split val
```

人工修订完成后，把最终标注文件保存为：

```text
page_level_ocr/page_level_annotations.json
```

先运行标注一致性检查：

```bash
python page_level_ocr/validate_annotations.py \
  --annotations page_level_ocr/page_level_annotations.json \
  --manifest page_level_ocr/page_split_manifest.csv
```

然后运行布局检测评估：

```bash
python page_level_ocr/run_layout_experiment.py \
  --manifest page_level_ocr/page_split_manifest.csv \
  --annotations page_level_ocr/page_level_annotations.json \
  --output-dir page_level_ocr/results/layout_baselines \
  --method proposed
```

如果没有提供标注文件，`run_layout_experiment.py` 仍然可以生成检测结果 JSON，供人工检查。

## 当前已生成文件

- `page_split_manifest.csv`：当前 62 页有效传统蒙古文整页图像的 page-level 划分；中文手写档案页已排除。
- `page_level_annotations.json`：当前页面级标注文件，包含 43 页 train、8 页 val、11 页 test；共 931 个有效 TextColumn 和 153 个 ignore 区域。
- `results/layout_lowconf_recovery_summary.md`：规则法 baseline 的 val/test 汇总。
- `results/yolo_column_detector/yolov8n_50ep_train43_summary.md`：43 页训练后的 YOLOv8n 实验摘要。
- `results/yolo_rule_gt_test_comparison_thr035/YOLO与规则法测试集对比分析.md`：YOLO 与规则法在统一 IoU@0.5 口径下的测试集对比。
- `results/end_to_end_ocr_test/端到端OCR实验阶段报告.md`：recognition-ready pipeline 运行记录；因无列级转写，不报告 CER/WER。
- `无转写替代研究方案_页面级版面分析.md`：缺少蒙古文专家转写时的主线调整方案。
- `论文改写清单_从端到端OCR调整为页面级版面分析.md`：投稿论文改写清单。

## 当前主结果

在独立 test split 上，YOLOv8n + rotation-aware reading order 的统一 IoU@0.5 指标为：Precision 0.919、Recall 0.626、F1 0.745、Mean IoU 0.767、Reading-order accuracy 0.920。规则法 baseline 的对应 F1 为 0.582。

## 下一步建议

优先按页面级版面分析论文推进：完善 qualitative figure、failure analysis、阈值敏感性和数据协议描述。端到端 OCR 只作为 recognition-ready pipeline 演示；除非后续获得真实列级转写，否则不报告 CER/WER。
