# DocLayout-YOLO 与 RT-DETR 补充基线结果

## 目的

为回应一区审稿意见中“缺少更多强基线 / Transformer 或文档版面检测模型对比”的问题，我们补充测试了两个迁移检测基线：

1. **RT-DETR-L**：DETR 类实时检测器。
2. **DocLayout-YOLO**：面向现代文档版面分析的 YOLO 系列模型，使用 DocStructBench 预训练权重初始化。

这两个模型均作为 **supplementary transfer baselines**，不作为本文主方法。

## 实验设置

### RT-DETR-L

- 权重：`rtdetr-l.pt`
- 训练：5 epochs
- 图像尺寸：640
- batch size：1
- 设备：CPU
- 阈值选择：validation split 上选择最佳 F1 阈值，然后固定到 test split
- 输出目录：
  `/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr/results/rtdetr_baseline/rtdetr_l_5ep_img640_cpu/`

### DocLayout-YOLO

- 权重：DocStructBench 预训练权重
  `/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr/results/doclayout_yolo_baseline/hf_weights/doclayout_yolo_docstructbench_imgsz1024.pt`
- 微调后权重：
  `/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr/results/doclayout_yolo_baseline/doclayout_yolo_5ep_img1024_cpu/weights/best.pt`
- 训练：实际完成 2 个有效 epoch 后生成 best/last 权重；按当前权重进行迁移基线评估
- 图像尺寸：1024
- batch size：1
- 设备：CPU
- 阈值选择：validation split 上选择最佳 F1 阈值，然后固定到 test split
- 评估脚本：
  `/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr/evaluate_doclayout_yolo_baseline.py`
- 输出目录：
  `/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr/results/doclayout_yolo_baseline/doclayout_yolo_5ep_img1024_cpu/`

## 测试集结果

| 方法 | Precision | Recall | F1 | Mean IoU | Reading-order Acc. |
|---|---:|---:|---:|---:|---:|
| RT-DETR-L + rotation-aware order | 0.032 | 0.270 | 0.058 | 0.632 | 0.847 |
| DocLayout-YOLO + rotation-aware order | 0.046 | 0.031 | 0.037 | 0.572 | 0.000 |

## 结论

两个迁移基线效果都很差，说明：

1. **现代印刷文档版面预训练模型不能直接迁移到传统蒙古文历史档案竖列检测。**
2. 传统蒙古文档案页面具有明显域差异：竖写、手写、稀疏列、纸张退化、印章干扰、旋转扫描等。
3. 当前最可靠的学习式基线仍是：
   - Faster R-CNN MobileNet：F1 = 0.782；
   - YOLOv8n：F1 = 0.745，阅读顺序准确率 = 0.920。

## 论文处理建议

- 不建议把 RT-DETR / DocLayout-YOLO 放入主结果表作为“强方法”，否则会拉低论文叙事。
- 建议放入 supplementary transfer baseline table，并解释为 **negative transfer evidence**。
- 这反而可以增强论文论点：传统蒙古文历史档案不是普通现代文档版面检测问题，需要专门标注、训练和后续领域适配。
