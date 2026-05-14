# 论文改写清单：从端到端 OCR 调整为页面级版面分析

## 1. 标题建议

优先标题：

**Layout-Aware Text Column Detection and Reading-Order Recovery for Traditional Mongolian Historical Archives**

备选标题：

**Page-Level Layout Analysis for Low-Resource Traditional Mongolian Historical Archives**

## 2. 摘要改写要点

摘要中应突出：

- 研究对象是整页传统蒙古文历史档案；
- 难点是竖排、手写、退化、旋转扫描、印章遮挡和异质档案混入；
- 方法是规则 baseline + YOLOv8n column detector + rotation-aware reading order；
- 主要结果是 test F1 = 0.745、reading-order accuracy = 0.920；
- OCR 识别部分只作为 recognition-ready pipeline，不报告 CER/WER。

避免在摘要中出现未验证的端到端识别准确率。

## 3. Introduction 改写要点

应把问题定义从“识别文本内容”前移到“从整页档案中可靠抽取可识别文本列”：

- 没有可靠列检测，后续识别模型无法稳定使用；
- 传统蒙古文历史档案存在非标准页面方向和复杂背景；
- 对低资源历史文档而言，列框和阅读顺序标注比全文转写更容易获得，因此更适合作为第一阶段研究。

## 4. Method 结构建议

建议方法部分采用以下结构：

1. Page-level annotation protocol；
2. Rule-based column extraction baseline；
3. YOLOv8n text-column detector；
4. Rotation-aware reading-order recovery；
5. Recognition-ready crop export pipeline。

第 5 点只描述接口，不做识别准确率主张。

## 5. Experiment 结构建议

主实验表：

| method | Precision | Recall | F1 | Mean IoU | Reading-order accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rule proposed | 0.680 | 0.509 | 0.582 | 0.831 | 0.907 |
| YOLOv8n + rotation-aware order | 0.919 | 0.626 | 0.745 | 0.767 | 0.920 |

补充实验：

- YOLO 训练规模对比：7、20、43 页训练；
- 阈值敏感性：0.25、0.30、0.35、0.40；
- qualitative comparison：GT / Rule / YOLO 三联图；
- failure cases：少列页、旋转页、长框过合并、噪声伪列。

## 6. Limitations 推荐写法

推荐表述：

> This study focuses on the upstream layout analysis stage. Expert transcription of handwritten traditional Mongolian archive columns is not currently available, so CER/WER is not reported. The released pipeline exports both oracle and automatically detected column crops and can be directly connected to a recognizer once column-level transcripts are added.

## 7. 需要删除或避免的内容

- 不要把当前识别模型输出当作真实 OCR 准确率。
- 不要在 page-level paper 中报告无 GT 支撑的 CER/WER。
- 不要使用“end-to-end OCR achieves ...”这类表述。
- 可以使用“recognition-ready pipeline”“upstream layout analysis”“column crop export”。
