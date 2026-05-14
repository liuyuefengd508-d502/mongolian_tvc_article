# 页面级文本列人工标注说明

本文档说明如何为页面级传统蒙古文 OCR 研究创建 `page_level_annotations.json`。推荐做法是先用自动检测结果生成标注初稿，再人工修正列框、阅读顺序和方向标签。

## 需要标注什么

每个验证集或测试集页面需要标注以下内容。

1. **文本列框**

格式为：

```text
[x_min, y_min, x_max, y_max]
```

坐标使用整页图像的像素坐标。每一个可读的蒙古文竖排文本列对应一个框。

2. **阅读顺序**

字段为 `reading_order`，从 `0` 开始编号：

```text
0, 1, 2, ...
```

默认按照传统蒙古文页面中的列顺序标注。当前项目默认从左到右排列文本列。如果某些页面布局特殊，请按真实阅读顺序填写，并在 `notes` 中说明。

3. **方向标签**

字段为 `orientation`。通常填写：

```text
correct
```

只有在页面或列框方向确实异常时，才使用：

```text
rotated_90_ccw
rotated_90_cw
rotated_180
ambiguous
```

4. **转写文本**

字段为 `transcript`。如果没有可靠转写，可以留空：

```json
"transcript": ""
```

只有当这一列已经有可信转写时再填写。

5. **退化标签**

字段为 `degradation_tags`。只标注这一列中肉眼可见的退化情况：

```text
severe_fade
bleed_through
red_seal
fold
stain
broken_spine
dense_background
marginalia
overlap
other
```

## 推荐标注流程

### 方式 A：从自动检测初稿开始

先生成自动标注初稿：

```bash
python page_level_ocr/bootstrap_annotations.py \
  --manifest page_level_ocr/page_split_manifest.csv \
  --output page_level_ocr/page_level_annotations.bootstrap.json \
  --method proposed \
  --split val
```

再生成可视化预览图：

```bash
python page_level_ocr/visualize_annotations.py \
  --annotations page_level_ocr/page_level_annotations.bootstrap.json \
  --output-dir page_level_ocr/annotation_previews/bootstrap_val \
  --split val
```

然后打开 `annotation_previews/bootstrap_val/` 中的预览图，对照修改：

```text
page_level_ocr/page_level_annotations.bootstrap.json
```

需要人工修正的内容包括：

- 删除误检列。
- 补上漏检列。
- 调整列框边界。
- 修正 `reading_order`。
- 修正 `orientation`。
- 添加 `degradation_tags`。

修订完成后，将最终文件命名为：

```text
page_level_ocr/page_level_annotations.json
```

保存后先运行校验脚本：

```bash
python page_level_ocr/validate_annotations.py \
  --annotations page_level_ocr/page_level_annotations.json \
  --manifest page_level_ocr/page_split_manifest.csv
```

如果输出中存在 `ERROR`，需要先修正标注文件再进入实验评估。`WARN` 通常表示需要人工确认，例如某一页还没有标注、某个框过大、或路径与 manifest 不一致。

### 方式 B：从空白示例开始

如果你想从零标注，可以参考：

```text
page_level_ocr/page_level_annotations.example.json
```

按照示例逐页填写即可。

## 列框标注规则

- 列框应包含完整可见文本列，包括顶部、底部的淡笔画。
- 不要把大面积空白边缘框进去，除非它对保留整列文字确实必要。
- 如果两个文本列因为污渍、印章或纸张破损相连，但人工仍能区分，应标成两个独立列。
- 如果某个深色区域明显不是文字列，不要标注为列。

## 阅读顺序规则

- 阅读顺序按页面真实阅读流程标注，不按文件名排序。
- `reading_order` 必须从 `0` 开始，并尽量连续。
- 如果页面存在特殊布局，仍按专家判断给出最合理顺序，并在 `notes` 记录原因。
- 如果顺序确实模糊，可以保留最佳判断，同时把 `orientation` 或 `notes` 标为 `ambiguous` 相关说明。

## ignore 使用规则

只有在以下情况才设置：

```json
"ignore": true
```

- 区域损坏严重，无法稳定定义列框。
- 区域不是文本，但会反复触发误检，需要从评估中排除。
- 该列暂时不参与识别或转写评估。

不要过度使用 `ignore`。只要能明确标出文本列，优先给出真实列框。

## 标注完成检查

每页完成后请检查：

- 所有可读蒙古文文本列都有框。
- 没有明显非文本区域被标成文本列。
- `reading_order` 从 `0` 开始且连续。
- `orientation` 取值合法。
- `degradation_tags` 与图像中可见退化情况一致。
- 页面 `split` 与 `page_split_manifest.csv` 一致。
- `validate_annotations.py` 没有报告 `ERROR`。

## 建议的最小标注批次

先标注验证集全部 8 页，再标注测试集 3 到 5 页。

这样我们很快就可以跑出第一版 layout F1、mIoU 和 reading-order accuracy。等流程稳定后，再扩展到完整测试集。
