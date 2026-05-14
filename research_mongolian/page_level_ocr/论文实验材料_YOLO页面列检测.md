# 论文实验材料：YOLO 页面级文本列检测与阅读顺序恢复

更新时间：2026-05-10

## 建议放入论文的位置

本材料适合放入下一篇页面级 OCR 论文的 Method 和 Experiments 部分，作为“学习式文本列检测 baseline / stronger baseline”。当前结果已经比规则法 `proposed` 更强，可以作为页面级 pipeline 的主要检测器候选，同时保留规则法作为可解释传统方法对照。

推荐小节结构：

```text
3. Method
  3.1 Page-level OCR Pipeline
  3.2 Learning-based Text Column Detector
  3.3 Rotation-aware Reading Order Recovery

4. Experiments
  4.1 Dataset and Annotation Protocol
  4.2 Layout Detection Metrics
  4.3 Comparison with Rule-based Column Extraction
  4.4 Threshold Sensitivity and Failure Analysis
```

## 方法小节草稿

### Learning-based text column detector

为补充规则法在复杂退化页面中的泛化不足，我们进一步训练了一个轻量学习式文本列检测器。检测器采用 YOLOv8n 作为 backbone，将页面级传统蒙古文历史档案图像作为输入，将每个竖排文本列作为 `TextColumn` 目标框输出。训练集由 43 页人工标注的整页档案图像组成，验证集和测试集分别包含 8 页和 11 页。所有中文手写档案页已从训练、验证和测试划分中排除；旋转重标页使用旋转后的标注图像作为训练图像，以保证检测框坐标与人工标注坐标一致。

训练时，输入尺寸设为 `960`，batch size 为 `1`，训练 `50` 个 epoch。模型在固定 validation split 上选择置信度阈值，避免使用测试集调参。最终主结果使用 `score_threshold=0.35`，该阈值在验证集的项目自定义 IoU@0.5 F1 指标上表现最佳。

### Rotation-aware reading order recovery

YOLO 输出的是无序检测框，因此需要将检测结果转换为页面阅读顺序。对于普通页面，我们按照检测框列中心的横坐标从左到右排序；对于人工旋转重标页，若标注图像路径包含 `rot90`，则按照列中心从右到左排序。该策略只改变检测框的 `reading_order` 字段，不改变检测框坐标，因此不会影响 Precision、Recall、F1 或 Mean IoU。

该轻量排序策略将 YOLOv8n 在测试集上的 reading-order accuracy 从原始左到右排序的 `0.820` 提升到 `0.920`，略高于规则法 `proposed` 的 `0.907`。

## 数据与训练设置

| split | pages | TextColumn boxes | ignored boxes |
| --- | ---: | ---: | ---: |
| train_unlabeled | 43 | 660 | 84 |
| val | 8 | 108 | 19 |
| test | 11 | 163 | 50 |
| total | 62 | 931 | 153 |

训练配置：

| item | setting |
| --- | --- |
| Detector | YOLOv8n |
| Initial weights | `yolov8n.pt` |
| Input size | 960 |
| Epochs | 50 |
| Batch size | 1 |
| Device | CPU |
| Ultralytics | 8.4.47 |
| Torch | 2.8.0 |
| Threshold selection | validation F1 |
| Final score threshold | 0.35 |

## 主实验表

统一采用项目自定义 IoU@0.5 贪心匹配指标。`Reading-order accuracy` 只在匹配到至少两个文本列的页面上计算 pairwise order accuracy。

| Method | Precision | Recall | F1 | Mean IoU | Reading-order accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rule proposed | 0.680 | 0.509 | 0.582 | 0.831 | 0.907 |
| YOLOv8n + rotation-aware order | 0.919 | 0.626 | 0.745 | 0.767 | 0.920 |

主要结论：

- YOLOv8n 的 F1 比规则法提高 `+0.163`。
- YOLOv8n 的 Precision 明显提高，说明误检显著减少。
- YOLOv8n 的 Recall 也高于规则法，说明漏检减少。
- Mean IoU 低于规则法，说明规则法一旦匹配成功，框边界通常更贴近人工框；YOLO 的优势主要来自更好的候选选择和更少误检。
- 旋转页感知排序后，YOLOv8n 的 reading-order accuracy 略高于规则法。

## 阈值敏感性分析

验证集阈值扫描：

| YOLO score threshold | Precision | Recall | F1 | predictions |
| ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0.506 | 0.833 | 0.629 | 178 |
| 0.15 | 0.596 | 0.833 | 0.695 | 151 |
| 0.20 | 0.667 | 0.815 | 0.733 | 132 |
| 0.25 | 0.746 | 0.787 | 0.766 | 114 |
| 0.30 | 0.812 | 0.759 | 0.785 | 101 |
| 0.35 | 0.876 | 0.722 | 0.792 | 89 |
| 0.40 | 0.873 | 0.639 | 0.738 | 79 |
| 0.45 | 0.896 | 0.556 | 0.686 | 67 |
| 0.50 | 0.941 | 0.444 | 0.604 | 51 |
| 0.60 | 0.963 | 0.241 | 0.385 | 27 |

测试集敏感性补充：

| YOLO score threshold | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: |
| 0.25 | 0.858 | 0.779 | 0.817 |
| 0.30 | 0.894 | 0.675 | 0.769 |
| 0.35 | 0.919 | 0.626 | 0.745 |

论文写法建议：主表报告 `0.35`，因为它由验证集选择；同时在 sensitivity analysis 中说明 `0.25-0.35` 区间内 YOLO 均明显优于规则法。

## 逐页分析

| page_id | Rule F1 | YOLO F1 | Delta | 观察 |
| --- | ---: | ---: | ---: | --- |
| 80-48-72-2 | 0.444 | 0.889 | +0.444 | YOLO 显著减少漏检和误检 |
| 80-48-70-1 | 0.261 | 0.556 | +0.295 | 规则法受背景/伪列影响更明显 |
| 80-48-66-1(1) | 0.400 | 0.632 | +0.232 | YOLO 漏检和误检均减少 |
| 80-48-61-1 | 0.500 | 0.683 | +0.183 | YOLO 召回更高 |
| 80-48-70-2 | 0.780 | 0.955 | +0.174 | YOLO 基本完整检出 |
| 80-48-65-1(1) | 0.743 | 0.857 | +0.114 | YOLO 匹配多数有效列 |
| 80-48-62-1(1) | 0.500 | 0.588 | +0.088 | 旋转页排序分支有效 |
| 80-48-73-1 | 0.194 | 0.276 | +0.082 | 两种方法均困难，仍需后续处理 |
| 80-48-72-5 | 0.903 | 0.968 | +0.065 | YOLO 小幅提升 |
| 80-48-70-4 | 0.947 | 0.900 | -0.047 | YOLO 多出少量误检 |
| 80-48-69-4 | 0.667 | 0.000 | -0.667 | 少列页面在阈值 0.35 下被过滤 |

## 可视化图选择建议

可从以下三联图中选择 3 到 4 张放入论文 qualitative comparison：

```text
page_level_ocr/results/yolo_rule_gt_test_comparison_thr035/images/80-48-72-2_gt_rule_yolo.jpg
page_level_ocr/results/yolo_rule_gt_test_comparison_thr035/images/80-48-70-2_gt_rule_yolo.jpg
page_level_ocr/results/yolo_rule_gt_test_comparison_thr035/images/80-48-65-1(1)_gt_rule_yolo.jpg
page_level_ocr/results/yolo_rule_gt_test_comparison_thr035/images/80-48-73-1_gt_rule_yolo.jpg
page_level_ocr/results/yolo_rule_gt_test_comparison_thr035/images/80-48-69-4_gt_rule_yolo.jpg
```

建议用途：

- `80-48-72-2`：展示 YOLO 明显改善规则法漏检。
- `80-48-70-2`：展示 YOLO 在较复杂页面上的完整检出。
- `80-48-65-1(1)`：展示横向宽图页面上 YOLO 仍能有效检测。
- `80-48-73-1`：展示两种方法都困难的 hard case。
- `80-48-69-4`：展示 YOLO 在高阈值下漏检少列页面的 limitation。

## 可直接复制的 LaTeX 表格

### Main comparison table

```latex
\begin{table}[t]
\centering
\caption{Page-level text column detection results on the test split. All methods are evaluated using IoU@0.5 greedy matching.}
\label{tab:layout_detection_test}
\begin{tabular}{lccccc}
\toprule
Method & Precision & Recall & F1 & Mean IoU & RO Acc. \\
\midrule
Rule proposed & 0.680 & 0.509 & 0.582 & 0.831 & 0.907 \\
YOLOv8n + rotation-aware order & \textbf{0.919} & \textbf{0.626} & \textbf{0.745} & 0.767 & \textbf{0.920} \\
\bottomrule
\end{tabular}
\end{table}
```

### Threshold sensitivity table

```latex
\begin{table}[t]
\centering
\caption{Sensitivity of YOLOv8n to detection confidence thresholds. The final threshold is selected on the validation split.}
\label{tab:yolo_threshold_sensitivity}
\begin{tabular}{lcccc}
\toprule
Split & Threshold & Precision & Recall & F1 \\
\midrule
Validation & 0.25 & 0.746 & 0.787 & 0.766 \\
Validation & 0.30 & 0.812 & 0.759 & 0.785 \\
Validation & 0.35 & \textbf{0.876} & 0.722 & \textbf{0.792} \\
\midrule
Test & 0.25 & 0.858 & 0.779 & 0.817 \\
Test & 0.30 & 0.894 & 0.675 & 0.769 \\
Test & 0.35 & \textbf{0.919} & 0.626 & 0.745 \\
\bottomrule
\end{tabular}
\end{table}
```

### Per-page analysis table

```latex
\begin{table}[t]
\centering
\caption{Representative per-page F1 comparison between the rule-based detector and YOLOv8n.}
\label{tab:per_page_layout_analysis}
\begin{tabular}{lccc}
\toprule
Page ID & Rule F1 & YOLO F1 & Delta \\
\midrule
80-48-72-2 & 0.444 & 0.889 & +0.444 \\
80-48-70-1 & 0.261 & 0.556 & +0.295 \\
80-48-66-1(1) & 0.400 & 0.632 & +0.232 \\
80-48-70-2 & 0.780 & 0.955 & +0.174 \\
80-48-70-4 & 0.947 & 0.900 & -0.047 \\
80-48-69-4 & 0.667 & 0.000 & -0.667 \\
\bottomrule
\end{tabular}
\end{table}
```

## 需要在论文中谨慎表述的点

1. 当前 test split 只有 11 页，结果适合写成阶段性页面级实验，但正式投稿前最好继续扩展独立测试集。
2. YOLOv8n 的 Mean IoU 低于规则法，说明边界紧致性还有提升空间。
3. 高阈值下少列页面可能被漏检，建议在 limitation 中说明。
4. `0.25` 阈值在 test 上 F1 更高，但不是主结果，因为不能用 test 调阈值。
5. 当前阅读顺序恢复依赖 `rot90` 路径标记；更完整的方法应使用自动方向分类器或页面方向元数据。

