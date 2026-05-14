# Page-Level OCR Research Plan for Traditional Mongolian Historical Archives

## 1. Research Target

This follow-up study extends the current isolated text-line recognition work into a page-level archival OCR pipeline for traditional Mongolian historical documents.

Proposed title:

**Unsupervised Page-Level Recognition of Traditional Mongolian Historical Archives via Layout-Aware Column Extraction and Domain-Adaptive Text Recognition**

Core distinction from the current manuscript:

- Current paper: isolated column/text-line recognition using SASE + VS-MSSE + DANN + valid-frame CTC entropy pseudo-labeling.
- Follow-up paper: full-page archival OCR with layout-aware column extraction, orientation correction, reading-order recovery, and end-to-end recognition evaluation.

Target journals:

1. International Journal on Document Analysis and Recognition (IJDAR)
2. Pattern Recognition Letters
3. ACM Journal on Computing and Cultural Heritage (JOCCH)

## 2. Main Contributions

The follow-up paper should make four contributions.

1. **Page-level Mongolian archival OCR formulation**
   - Input: full-page scans of 19th-century traditional Mongolian administrative archives.
   - Output: recognized text columns in correct reading order.
   - Evaluation: layout detection, orientation accuracy, reading-order accuracy, and end-to-end CER/WER.

2. **Layout-aware column extraction**
   - Upgrade the current `segment_columns_v2.py` heuristic into a reproducible experimental module.
   - Combine adaptive Sauvola thresholding, multi-scale vertical projection, column-width priors, broken-column merging, pseudo-column filtering, and orientation correction.

3. **Recognition integration**
   - Reuse the existing SASE + VS-MSSE + DANN + valid-frame CTC entropy recognizer as the fixed recognition backbone.
   - Compare oracle column crops against automatically detected crops to quantify layout-induced recognition loss.

4. **Error propagation analysis**
   - Measure how missed columns, over-segmentation, wrong reading order, wrong orientation, and low-confidence noisy crops affect page-level CER/WER.

## 3. Data and Annotation Protocol

### Page Selection

Select approximately 100-150 full archival pages from the existing 19th-century Qing Dynasty administrative archive scans.

Recommended split:

| Split | Pages | Purpose |
|---|---:|---|
| Unlabeled target train | 70-100 | layout tuning without labels, texture extraction, unsupervised parameter selection |
| Validation | 15-25 | supervised development reporting only |
| Test | 15-25 | final page-level evaluation |

Splits must be page-level. No page may contribute crops, textures, thresholds, or tuning signals to more than one split.

### Annotation Items

For each validation/test page, annotate:

- Column bounding boxes: `(x_min, y_min, x_max, y_max)`
- Reading order index: left-to-right column order for traditional Mongolian pages
- Orientation label: correct / rotated / ambiguous
- Degradation tags: severe fade, bleed-through, red seal, fold, stain, broken spine, dense background

Manual character-level transcription is not required for every page if column transcripts already exist. The minimum requirement is to map each annotated page column to its corresponding transcript or mark it as layout-only.

### Leakage Rule

The following operations must use only unlabeled target training pages:

- Blank paper texture extraction for SASE
- FID or visual-domain parameter selection
- Unsupervised target entropy monitoring
- Any threshold tuning not explicitly reported as validation-based

Validation and test pages must be excluded from texture extraction and unsupervised tuning.

## 4. System Design

### Stage 1: Page Preprocessing

Input a full-page image. Convert to grayscale, normalize contrast, optionally denoise lightly, and keep original resolution for final cropping.

Outputs:

- normalized grayscale image
- binarized foreground mask
- page metadata: size, estimated foreground density, degradation score

### Stage 2: Layout-Aware Column Extraction

Baseline module:

- Sauvola thresholding
- morphological cleanup
- vertical projection
- connected components over projection mask
- width-based column filtering

Proposed module:

- multi-scale Sauvola thresholds
- smoothed vertical projection at multiple kernel widths
- adaptive foreground threshold based on page density
- column-width and gap priors
- broken-column merging
- pseudo-column filtering by ink density and aspect ratio
- column bounding-box padding

Outputs:

- ordered column boxes
- confidence score per detected column
- layout diagnostics for failure analysis

### Stage 3: Orientation Correction

For each detected column crop:

- compute horizontal and vertical projection variance
- detect likely orientation
- rotate only when confidence is high
- preserve an ambiguous flag if the orientation score is close

Metrics:

- orientation accuracy
- ambiguous-rate
- recognition delta with and without orientation correction

### Stage 4: Recognition

Use the existing trained recognizer:

- SASE synthetic source training
- VS-MSSE feature extractor
- DANN domain alignment
- valid-frame CTC entropy pseudo-label filtering

Run two evaluation modes:

- **Oracle mode**: use manually annotated column boxes.
- **Detected mode**: use automatically detected column boxes.

Report the gap between oracle and detected mode as layout-induced recognition loss.

### Stage 5: Reading Order Recovery

Sort detected columns by page coordinate using the traditional Mongolian reading order. The default is left-to-right column ordering. If archival examples show exceptional layouts, record them as failure cases rather than adding complex layout policies in the 8-week version.

Metrics:

- exact reading-order accuracy
- Kendall tau or pairwise ordering accuracy
- page-level CER after concatenating recognized columns in predicted order

## 5. Experimental Matrix

### Layout Detection Table

Compare:

| Method | Precision | Recall | F1 | mIoU | Reading-order Acc. |
|---|---:|---:|---:|---:|---:|
| Sauvola + Projection | | | | | |
| Connected Components + Morphology | | | | | |
| Proposed Layout-Aware Extraction | | | | | |
| Optional Lightweight Detector | | | | | |

Primary target: column detection F1 above 90%.

### Recognition Table

Compare:

| Input Crop Source | Orientation Correction | Confidence Filtering | CER | WER | Page CER |
|---|---|---|---:|---:|---:|
| Oracle boxes | no | no | | | |
| Detected boxes | no | no | | | |
| Detected boxes | yes | no | | | |
| Detected boxes | yes | yes | | | |

Target: detected-mode CER should be within +2% to +5% absolute CER of oracle-mode CER.

### Error Propagation Table

Quantify:

| Error Source | Frequency | CER Impact | Typical Cause |
|---|---:|---:|---|
| Missed column | | | faded or broken column |
| Over-segmentation | | | red seal, fold, stains |
| Wrong reading order | | | irregular page layout |
| Wrong orientation | | | low-contrast crop |
| Pseudo-column insertion | | | background oxidation |

### Ablation Table

Compare:

| Configuration | Layout F1 | Page CER |
|---|---:|---:|
| Full pipeline | | |
| w/o adaptive threshold | | |
| w/o column merge | | |
| w/o pseudo-column filtering | | |
| w/o orientation correction | | |
| w/o confidence filtering | | |

## 6. Eight-Week Schedule

### Week 1: Dataset and Annotation Setup

- Select 100-150 full-page images.
- Define page-level train/validation/test splits.
- Create annotation format for column boxes, reading order, orientation labels, and degradation tags.
- Produce dataset statistics.

Deliverables:

- `page_split_manifest.csv`
- `page_level_annotations.json`
- dataset statistics table

### Week 2: Baseline Layout Experiments

- Standardize `segment_columns_v2.py` into a reproducible baseline.
- Add connected-component baseline.
- Implement IoU matching and layout metrics.

Deliverables:

- baseline layout metrics
- first qualitative detection figure

### Weeks 3-4: Proposed Layout-Aware Extraction

- Add multi-scale thresholding and projection.
- Add column priors, broken-column merging, and pseudo-column filtering.
- Add orientation correction.
- Run ablations.

Deliverables:

- proposed layout metrics
- ablation table
- failure case gallery

### Week 5: End-to-End OCR Integration

- Run oracle crop recognition.
- Run detected crop recognition.
- Compute page-level CER/WER.
- Analyze layout-induced recognition loss.

Deliverables:

- recognition table
- error propagation table

### Week 6: Visualization and Evidence

Use existing visual assets:

- `ocr_uda_architecture_1777988343093.png`: system architecture figure
- `qualitative_results.png`: recognition examples
- `tsne_alignment.png`: feature alignment visualization

Add new visualizations:

- page-level column detection examples
- reading-order examples
- failure cases for red seals, broken columns, pseudo-columns, and wrong order

### Week 7: Manuscript Drafting

Recommended paper structure:

1. Introduction
2. Related Work
3. Page-Level OCR Problem Formulation
4. Layout-Aware Column Extraction
5. Domain-Adaptive Recognition Backbone
6. Experiments
7. Error Propagation Analysis
8. Limitations
9. Conclusion

### Week 8: Submission Hardening

- Run at least 3 random seeds; 5 seeds if time allows.
- Verify all reference metadata.
- Prepare Overleaf-ready LaTeX package.
- Add Data Availability, Ethics, Reproducibility, Funding, and Author Contributions.
- Prepare cover letter explaining why this is not duplicate publication.

## 7. Manuscript Positioning

The follow-up paper must not be framed as simply improving the current recognizer. It should be framed as a new page-level OCR study.

Recommended contribution wording:

- "We extend isolated traditional Mongolian text-line recognition to full-page archival OCR."
- "We quantify the impact of layout errors on recognition accuracy."
- "We provide the first page-level evaluation protocol for degraded traditional Mongolian archival pages."

Avoid overclaiming:

- Do not claim full archival understanding.
- Do not claim solved layout analysis for all historical Mongolian documents.
- Do not imply validation/test pages are used for SASE texture extraction or unsupervised tuning.

## 8. Success Criteria

The study is ready for manuscript submission when:

- column detection F1 is above 90%, or failure modes are clearly explained if lower
- detected-crop CER is within +2% to +5% absolute CER of oracle-crop CER
- proposed layout method beats the Sauvola + projection baseline
- page-level OCR examples show correct column order and readable output
- data leakage controls are documented
- current recognition paper and follow-up page-level paper have clearly separated contributions
