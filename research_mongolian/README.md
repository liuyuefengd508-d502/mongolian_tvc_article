# Unsupervised Domain Adaptation for Isolated Traditional Mongolian Text Line Recognition

This repository contains the dataset structures, preprocessing scripts, model training weights, and the LaTeX manuscript for the Mongolian OCR research project. 

This README serves as a comprehensive guide to the code files, file paths, and dataset specifications utilized to reproduce the experiments and generate the final manuscript.

---

## 1. Manuscript & Publication Files

The final manuscript submitted for peer review is located in the `research_mongolian` directory.

- **LaTeX Source Code**: `/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/manuscript_archives_v24.tex`
  - **用途**: The final Q1-tier peer-reviewed academic manuscript containing the Abstract, Methodology (SASE, VS-MSSE, DANN, valid-frame CTC entropy), Ablation Studies, and 22 verified bibliography references.
- **Compiled PDF**: `/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/manuscript_archives_v24.pdf`
  - **用途**: The final 9-page compiled PDF document ready for journal submission.
- **Figures Directory**: `/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/figures/`
  - **用途**: Stores all figures used in the manuscript, including the authentic data crop (`real_authentic.png`) and the high-resolution generated synthetic text line (`sase_synthetic.png`).

---

## 2. Core Code Files & Scripts

The Python codebase for layout analysis, physical degradation synthesis, and data preprocessing is executed within the sandboxed environment directory.

### 2.1 Synthetic Engine (SASE)
- **Mass Synthesis Script**: `/Users/liuyu/.gemini/antigravity/brain/137f810a-82d5-417e-bab8-1520b50c69ed/scratch/mass_synthesis.py`
  - **用途**: Main generation script. Parses the Mongolian text corpus, extracts vocabularies, loads the `NotoSansMongolian` font, applies elastic deformations, blends authentic paper textures, and generates high-resolution synthetic training data in mass quantities.
- **Advanced Synthesis Module**: `/Users/liuyu/.gemini/antigravity/brain/137f810a-82d5-417e-bab8-1520b50c69ed/scratch/advanced_synthesis.py`
  - **用途**: Contains the core physics-based degradation logic (e.g., Gaussian blur, ink diffusion, texture interpolation).

### 2.2 Layout Analysis & Binarization
- **Column Segmentation**: `/Users/liuyu/.gemini/antigravity/brain/137f810a-82d5-417e-bab8-1520b50c69ed/scratch/segment_columns_v2.py`
  - **用途**: Performs adaptive thresholding and vertical projection profiling to automatically extract isolated text columns from authentic historical manuscript pages.
- **Orientation Detection**: `/Users/liuyu/.gemini/antigravity/brain/137f810a-82d5-417e-bab8-1520b50c69ed/scratch/detect_orientation.py`
  - **用途**: Ensures cropped bounding boxes maintain the correct vertical flow orientation constraint specific to traditional Mongolian.
- **Texture Extraction**: `/Users/liuyu/.gemini/antigravity/brain/137f810a-82d5-417e-bab8-1520b50c69ed/scratch/extract_texture.py`
  - **用途**: Extracts blank paper patches only from unlabeled target training pages, explicitly excluding validation and test pages to prevent data leakage. These patches serve as authentic background noise templates for the SASE engine.

---

## 3. Dataset Specifications

The research utilizes a dual-domain setup: a Source Domain (Synthetic) and a Target Domain (Authentic Archives).

### 3.1 Target Domain: Authentic Historical Archives
- **Path**: `/Users/liuyu/.gemini/antigravity/brain/137f810a-82d5-417e-bab8-1520b50c69ed/scratch/processed_mongolian/`
- **Description**: Contains high-resolution isolated column crops from 19th-century administrative records provided by the Inner Mongolia University Historical Archives (Permission No. 2024-IMU-089). 
- **Structure**: Grouped by manuscript page ID (e.g., `80-48-61-1`, `80-48-62-2(1)`). Contains severe physical degradations like ink bleed-through, faded strokes, and background oxidation. 

### 3.2 Source Domain: SASE Synthetic Dataset
- **Path**: `/Users/liuyu/Desktop/mydocuments/codes/autoResearch/data/synthetic_train/images/`
- **Description**: Contains 2,320 synthetically generated Mongolian text lines used as labeled source data. It leverages physical background simulation and domain-aligned visual properties.
- **Labeling Mapping**: `/Users/liuyu/Desktop/mydocuments/codes/autoResearch/data/synthetic_train/labels.txt`

---

## 4. Model Weights

The pre-trained PyTorch weights across various training stages are stored in the root workspace.

- **Path**: `/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/`
- **Key Files**:
  - `mongolian_baseline_epoch_*.pth` (10 epochs): Vanilla CRNN weights trained purely on synthetic data prior to adaptation.
  - `mongolian_uda_epoch_*.pth` (5 epochs): DANN intermediate domain-adversarial weights aligning Source and Target distributions.
  - `mongolian_ocr_final.pth`: The finalized weights integrating the VS-MSSE structural enhancement and valid-frame CTC entropy pseudo-label refinement. Yields $7.8\%$ CER on the authentic target test set.

---

## 5. Next Research Direction: Page-Level Archival OCR

The next research phase extends the current isolated text-line UDA work into a page-level archival OCR system. The planned follow-up study is documented in:

- **Research Plan**: `/Users/liuyu/Desktop/mydocuments/codes/autoAIScienceforOCR/research_mongolian/page_level_ocr_research_plan.md`
  - **用途**: Eight-week Q1 manuscript plan covering page-level column detection, orientation correction, reading-order recovery, end-to-end OCR evaluation, error propagation analysis, and manuscript preparation.

The follow-up study keeps the current SASE + VS-MSSE + DANN + valid-frame CTC entropy recognizer as the recognition backbone and adds a reproducible layout stage for full-page historical archive images. All page-level splits, texture extraction, unsupervised tuning, and synthetic-domain parameter selection must exclude validation/test pages.
