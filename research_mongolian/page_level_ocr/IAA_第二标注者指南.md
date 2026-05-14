# IAA 第二标注者独立标注指南

## 目的

本任务用于计算页面级传统蒙古文档案标注的一致性（Inter-Annotator Agreement, IAA）。第二标注者只需要标注文本列框、Ignore 区域和阅读顺序，不需要懂蒙古文，也不需要转写文字内容。

## 标注页面

本次抽取 5 页代表性样本：

| page_id | 类型 | 目的 |
|---|---|---|
| 80-48-70-2 | 普通多列页 | 测试常规页面一致性 |
| 80-48-62-1(1) | 旋转页 | 测试方向和阅读顺序一致性 |
| 80-48-69-4 | 少列页 | 测试少量文本列时的边界判断 |
| 80-48-72-2 | 噪声/Ignore 较多页 | 测试非文本区域处理 |
| 80-48-73-1 | 长列/易过合并页 | 测试复杂列框粒度 |

Label Studio 导入任务文件：

```text
page_level_ocr/label_studio_import/iaa_5page_tasks.json
```

图像目录：

```text
page_level_ocr/label_studio_media/iaa_png/
```

## 标注规则

### 1. TextColumn

标注所有可见的传统蒙古文文本列：

- 每一列文字画一个矩形框；
- 尽量贴合文字主体，不需要包含整页空白；
- 同一列上下断开但明显属于同一竖列时，优先标为一个框；
- 两列之间有明显间距时，应标为两个框；
- 不要求识别文字内容。

### 2. Ignore

以下区域标为 Ignore：

- 明显不是传统蒙古文正文的区域；
- 印章、污渍、边缘阴影导致的伪文字区域；
- 极度模糊、无法判断是否为正文的片段；
- 中文手写内容或非目标文字内容。

### 3. Reading order

对 TextColumn 填写阅读顺序：

- 从 0 开始编号；
- 普通页面按页面上真实阅读顺序编号；
- 旋转后已经变为正向显示的页面，也按视觉上的阅读顺序编号；
- Ignore 区域可以不关心顺序，但如果工具要求填写，可以放在所有 TextColumn 之后。

### 4. Orientation

如果页面已经在工具中显示为正向，选择 `correct`。

只有在明显异常时才选择：

- `rotated_90_ccw`
- `rotated_90_cw`
- `rotated_180`
- `ambiguous`

## 标注完成后的导出

Label Studio 标注完成后导出 JSON，然后需要转换成项目格式。最终第二标注者文件建议命名为：

```text
page_level_ocr/page_level_annotations.iaa_second.json
```

## 计算 IAA

导出并转换完成后运行：

```bash
python page_level_ocr/compute_iaa.py \
  --primary page_level_ocr/page_level_annotations.json \
  --secondary page_level_ocr/page_level_annotations.iaa_second.json \
  --pages-csv page_level_ocr/label_studio_import/iaa_5page_selected_pages.csv \
  --output-dir page_level_ocr/results/iaa_5page
```

输出：

```text
page_level_ocr/results/iaa_5page/iaa_summary.md
page_level_ocr/results/iaa_5page/iaa_summary.json
page_level_ocr/results/iaa_5page/iaa_per_page.csv
```

## 论文中建议报告的 IAA 指标

- box-level F1@0.5；
- mean matched IoU；
- pairwise reading-order agreement；
- ignore-region F1@0.5。
