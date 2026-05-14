# Point-by-Point Response to the Editorial Comments

Submission ID: 5e3eebcf-7254-4032-8499-a733072c8fe7  
Revised manuscript title: *Unsupervised Domain Adaptation for Vertical Cursive Script Recognition with Geometric Stroke Enhancement and Pseudo-Label Refinement*  
Journal: *The Visual Computer*

Dear Professor Sheng and Editorial Team,

We sincerely thank you for your careful editorial assessment and for the opportunity to submit a substantially revised manuscript. We have revised the paper extensively to improve its fit with the visual computing community, clarify its reusable research contribution, strengthen its presentation and discoverability, and enhance transparency and reproducibility through public code and archival release materials.

Below we respond point by point to the major editorial concerns and summarize the corresponding revisions in the manuscript.

## Responses to the Three Required Questions

### 1. Does the manuscript clearly articulate a reusable problem formulation, benchmark, dataset, codebase, taxonomy, or methodological insight that other researchers can cite and build upon?

Yes. In the revised manuscript, we explicitly formulate the task as strict unsupervised domain adaptation for isolated vertical cursive script recognition. The revised problem formulation now clearly defines the labeled synthetic source domain, the unlabeled authentic target domain, the target-label exclusion rule during training and model selection, and the final evaluation protocol.

We also strengthened the paper as a reusable benchmark by clarifying:

- the source/target split design
- page-level partitioning to avoid cross-column leakage
- dataset statistics and evaluation metrics
- baseline settings and ablation settings
- the key reusable methodological insights, namely SASE, VS-MSSE, sequence-aware adversarial alignment, and valid-frame-normalized CTC entropy pseudo-label refinement

Corresponding manuscript revisions:

- A concise contribution statement has been added to the Introduction.
- The Methodology section now states the strict UDA setting more explicitly.
- The Experiments section now presents the benchmark design, data-splitting rules, and evaluation setup more clearly.

### 2. Are the title, abstract, keywords, contribution statement, figures, and experimental comparisons sufficiently discoverable by researchers outside the immediate subfield?

Yes. We revised the title, abstract, keywords, and contribution statement to make the paper easier to discover and understand for readers in visual computing, document image analysis, cultural heritage digitization, and low-resource recognition, beyond the immediate Mongolian OCR subfield.

Specifically:

- The revised title foregrounds the broader task of vertical cursive script recognition.
- The revised abstract reduces excessive technical density and more clearly states the cultural-heritage motivation, the core method, the benchmark value, and the main quantitative result.
- The keywords now include broader discovery terms such as “vertical cursive script recognition,” “document image analysis,” and “cultural heritage digitization.”
- The literature review now more clearly connects our work to recent visual-computing topics in structure-preserving restoration and structure-aware recognition.

Corresponding manuscript revisions:

- The title, abstract, and keywords were revised.
- The Introduction now includes a clearer contribution summary.
- The Related Work section now better situates the manuscript within the scope of *The Visual Computer*.
- The figure descriptions and experimental framing now more clearly emphasize the visual-structural motivation of the proposed method.

### 3. Are code, data, pretrained models, demos, or benchmark protocols made available, when ethically and technically feasible?

Yes, to the extent ethically and technically feasible.

We have prepared and publicly released a repository associated with the revised manuscript:

- GitHub repository: `https://github.com/liuyuefengd508-d502/mongolian_tvc_article`
- Zenodo Concept DOI: `https://doi.org/10.5281/zenodo.20176970`
- Zenodo Version DOI for `v1.0.0`: `https://doi.org/10.5281/zenodo.20176971`

The public release includes manuscript materials, public figures, lightweight research code, reproducibility metadata, benchmark-related scripts, protocol descriptions, and annotation examples. Because the complete authentic archival dataset is subject to institutional and cultural-heritage access restrictions, it cannot be fully released publicly. In addition, large internal training artifacts are not part of the lightweight public repository. To support reproducibility as far as possible, we therefore provide a releasable subset, example annotation formats, split/protocol descriptions, and scripts that reproduce the reported evaluation pipeline when authorized users have access to the restricted data.

Corresponding manuscript revisions:

- The abstract now includes the public repository and archival availability statement.
- The Data Availability and Reproducibility Statement now includes the GitHub repository, Zenodo concept DOI, and version DOI.
- The repository and archive pages explicitly state that the code is directly related to the manuscript submitted to *The Visual Computer* and request citation of the manuscript and software archive.

## Detailed Revision Summary

### 1. Title revision

Comment addressed: The original title was less discoverable outside the immediate OCR subfield.

Revision:

**Unsupervised Domain Adaptation for Vertical Cursive Script Recognition with Geometric Stroke Enhancement and Pseudo-Label Refinement**

Rationale: The revised title better reflects the reusable task setting, the geometric visual-computing contribution, and the pseudo-label refinement strategy.

### 2. Abstract revision

Comment addressed: The abstract should more clearly communicate the problem, contribution, results, and broader significance.

Revision: We rewrote the abstract to improve logical flow, reduce unnecessary technical density, highlight the cultural-heritage context, report the main quantitative result more clearly, and explicitly state code and archive availability.

### 3. Keywords revision

Comment addressed: The paper should be more discoverable to readers beyond the immediate niche.

Revision: We revised the keywords to include broader discovery-oriented terms:

vertical cursive script recognition; traditional Mongolian script; historical archives; unsupervised domain adaptation; document image analysis; pseudo-label refinement; cultural heritage digitization.

### 4. Contribution statement added

Comment addressed: The manuscript should clearly articulate reusable contributions that other researchers can cite and build upon.

Revision: We added a concise contribution list in the Introduction. It now explicitly states:

- the reusable UDA benchmark formulation
- the VS-MSSE geometric stroke enhancement module
- the sequence-aware adversarial alignment and valid-frame CTC entropy pseudo-label refinement strategy
- the public release and reproducibility contribution

### 5. Relevance to *The Visual Computer* strengthened

Comment addressed: The manuscript should better align with the journal’s computer graphics and visual computing readership.

Revision: We strengthened the Related Work discussion and selectively incorporated the recommended recent *The Visual Computer* references:

- *Detail-aware image denoising via structure preserved network and residual diffusion model*. *The Visual Computer*, 2025, 41(1): 639-658.
- *SATD: syntax-aware handwritten mathematical expression recognition based on tree-structured transformer decoder*. *The Visual Computer*, 2025, 41(2): 883-900.

Rationale: These references help position the paper within current visual-computing discussions on structure-preserving enhancement and structure-aware recognition, without excessive citation.

### 6. Novelty clarified relative to UDA and text-recognition baselines

Comment addressed: The novelty and theoretical depth should be clarified more clearly.

Revision: We clarified that conventional UDA methods mainly emphasize global feature alignment, whereas our method contributes:

- a script-geometry-aware visual prior through VS-MSSE
- a corrected valid-frame CTC entropy criterion tailored to pseudo-label reliability in degraded sequence recognition

### 7. Comparative experiments and ablations clarified

Comment addressed: Comparative experiments should be stronger and more clearly presented.

Revision: The revised manuscript preserves and clarifies comparisons against CRNN, TrOCR, PARSeq, CycleGAN+CRNN, DANN, CDAN, and a fully supervised upper-bound reference. We also present ablations for SASE, VS-MSSE, DANN, and pseudo-label refinement, together with architectural ablation for vertical asymmetric kernels.

### 8. Figure interpretation and visual validation strengthened

Comment addressed: More intuitive visual evidence should be provided for the proposed module.

Revision: We strengthened the textual explanation of the qualitative figures and the Grad-CAM analysis. The revised text more clearly states that VS-MSSE activates along the central vertical spine and that the remaining error modes are primarily associated with broken spines, visually similar vowels, and background stains/noise.

### 9. Open-source and reproducibility statement expanded

Comment addressed: Code and reproducibility resources should be permanently hosted and documented.

Revision: We expanded the Data Availability and Reproducibility Statement. It now specifies the public repository, the Zenodo concept DOI, the first release DOI, and the types of released materials, including scripts, protocols, figures, annotation examples, and releasable benchmark resources.

Public release information:

- GitHub repository: `https://github.com/liuyuefengd508-d502/mongolian_tvc_article`
- Zenodo Concept DOI: `https://doi.org/10.5281/zenodo.20176970`
- Zenodo Version DOI (`v1.0.0`): `https://doi.org/10.5281/zenodo.20176971`

### 10. Ethical and technical data-release constraints clarified

Comment addressed: Data should be released when ethically and technically feasible.

Revision: We explicitly clarified that the complete authentic archival dataset is restricted by institutional and cultural-heritage access conditions. To balance reproducibility with these constraints, the revised manuscript commits to releasing public benchmark scaffolding, annotation examples, split/protocol descriptions, and code, while reserving full archival access for authorized use.

## Closing Statement

We are grateful for the editorial guidance and believe the revised manuscript is substantially improved in clarity, presentation quality, reproducibility, and relevance to the readership of *The Visual Computer*. We hope that the revised manuscript and this itemized response adequately address the editorial concerns and facilitate an efficient reassessment.

Sincerely,  
Lanying Liang and Yuefeng Liu
