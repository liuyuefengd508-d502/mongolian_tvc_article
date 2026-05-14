# Mongolian TVC Article Repository

This repository contains the manuscript materials, lightweight research code, figures, and reproducibility metadata for the paper:

**Unsupervised Domain Adaptation for Vertical Cursive Script Recognition with Geometric Stroke Enhancement and Pseudo-Label Refinement**

The repository is directly related to the manuscript currently submitted to *The Visual Computer*. If you use this code, protocol, or released artifacts, please cite the associated manuscript and the Zenodo software archive.

[![DOI](https://zenodo.org/badge/1238487178.svg)](https://doi.org/10.5281/zenodo.20176970)

## Repository structure

- `journal_submission/The_Visual_Computer/`: current resubmission manuscript, response letter draft, bibliography, and journal assets.
- `research_mongolian/`: lightweight experiment scripts, manuscript sources, figures, and planning/reproducibility notes.
- Root-level documentation and metadata: repository overview, citation metadata, and Zenodo metadata.

## What is included

- LaTeX sources for the manuscript and revision package
- Figures used in the paper
- Python scripts for evaluation, visualization, and supporting experiments
- Reproducibility metadata for GitHub and Zenodo archiving

## What is not included

- Full archival dataset
- Large trained weights and experiment output directories
- Unrelated third-party or side-project repositories in this workspace

The complete authentic archival dataset is subject to institutional and cultural-heritage access restrictions. This public repository therefore releases code, protocols, manuscript materials, and lightweight examples only. Additional restricted data can be accessed only with the appropriate permissions.

## Environment

This repository does not yet provide a single frozen environment file. The codebase is primarily Python-based and assumes a standard scientific Python environment for OCR/document-analysis workflows. Before public release, dependencies should be pinned in either `requirements.txt` or `environment.yml` if you want fully automated setup.

## Reproducibility notes

The manuscript focuses on unsupervised domain adaptation for isolated vertical cursive script recognition. To reproduce the paper at a lightweight level:

1. Inspect the manuscript sources in `journal_submission/The_Visual_Computer/`.
2. Review the research notes and scripts in `research_mongolian/`.
3. Use the released figures, evaluation scripts, and split/protocol descriptions as the public benchmark scaffold.
4. Combine the public code with the authorized restricted dataset if full experimental reruns are needed.

## Related manuscript

This repository supports the manuscript submitted to *The Visual Computer*. The GitHub repository URL is:

`https://github.com/liuyuefengd508-d502/mongolian_tvc_article`

The Zenodo archive is now available:

1. Concept DOI: `10.5281/zenodo.20176970`
2. Version DOI for `v1.0.0`: `10.5281/zenodo.20176971`

## Citation

Please cite both:

1. The manuscript submitted to *The Visual Computer*
2. The Zenodo software archive for the public release version you used
