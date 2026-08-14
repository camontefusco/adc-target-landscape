# Codex Prompt: Rapid Go/No-Go Pilot for a Multi-Target ADC Landscape Study Using Precomputed Pathology Embeddings

## Project objective

Build the fastest scientifically defensible pilot for a **NeurIPS 2026 LatinX in AI workshop extended abstract** using existing precomputed pathology foundation-model embeddings and matched transcriptomic data.

The project must answer:

> **Can histology-derived foundation-model representations recover the relative landscape of ADC-relevant target expression within individual breast cancer patients?**

This is not a generic biomarker classifier.

The central novelty is to treat therapeutically relevant targets as a **joint within-patient ranking problem** rather than as isolated binary prediction tasks.

The minimum viable study should use:

* TCGA-BRCA;
* precomputed UNI pathology embeddings;
* matched RNA expression;
* 4–6 ADC-relevant targets;
* simple pooled slide representations;
* simple regularized prediction models;
* within-patient target-ranking metrics;
* no WSI preprocessing;
* no foundation-model fine-tuning;
* no RNA model;
* no multimodal transformer;
* no MIL unless the simple model clearly works first.

The first goal is not to build the full paper.

The first goal is to determine quickly whether the idea has enough signal to justify the workshop abstract.

---

# Your role

Act as:

* computational pathology research engineer;
* biomedical ML scientist;
* oncology biomarker analyst;
* statistical programmer;
* scientific-methods reviewer;
* reproducibility engineer;
* pair programmer.

I am beginner-to-intermediate in Python.

Therefore:

* explain important steps;
* keep code simple and modular;
* avoid unnecessary abstractions;
* use robust libraries;
* write tests for patient matching and leakage;
* prioritize a working scientific result over sophisticated infrastructure.

---

# Critical scope control

Do not begin by:

* downloading whole-slide images;
* generating tiles;
* running UNI;
* running TANGLE training;
* building MIL;
* using LoRA;
* fine-tuning any foundation model;
* adding SHAP;
* adding RNA encoders;
* building Docker;
* designing a new neural architecture.

Use the **already prepared TCGA-BRCA pathology embeddings and expression data** if they are actually available and usable.

If the prepared data fail, report the failure and immediately identify the simplest fallback.

---

# Phase 0 — Immediate feasibility audit

This is the first and most important step.

Before writing modelling code, verify the exact files available from the official TANGLE or associated repository.

Determine:

1. exact download location;
2. file names;
3. file formats;
4. file sizes;
5. whether authentication is required;
6. license;
7. whether patient or slide identifiers are preserved;
8. structure of pathology embedding files;
9. embedding dimensionality;
10. structure of RNA expression files;
11. gene identifier format;
12. whether pathology and RNA are already matched;
13. number of patients;
14. number of slides;
15. number of patients with both modalities.

Do not rely on README assumptions.

Inspect the actual files or repository metadata.

Output a concise feasibility report.

---

# Phase 1 — ADC target audit

Before modelling, inspect a broad candidate list of ADC-relevant targets.

Start with:

```text
ERBB2
TACSTD2
ERBB3
FOLR1
NECTIN4
CD276
MET
EGFR
CEACAM5
F3
LIV1 / SLC39A6
MSLN
```

Use only targets that exist in the expression matrix.

For every available target, calculate:

* number of patients;
* mean;
* standard deviation;
* median;
* IQR;
* minimum;
* maximum;
* percent zeros;
* 5th percentile;
* 95th percentile;
* coefficient of variation;
* skewness.

Also visualize each expression distribution.

Create a table:

```text
target
n
mean
sd
median
iqr
pct_zero
p05
p95
cv
skew
usable_yes_no
reason
```

Define simple target-selection rules.

Prefer targets with:

* sufficient variation;
* few or no missing values;
* few structural zeros;
* clear therapeutic relevance;
* non-identical expression distributions.

Do not select targets merely because they are famous ADC targets.

Recommend 4–6 targets for the pilot.

---

# Phase 2 — Patient and slide matching audit

Determine whether the dataset is organized as:

* one embedding file per slide;
* one embedding file per patient;
* multiple slides per patient.

Create a manifest containing:

```text
patient_id
slide_id
embedding_path
n_patches
embedding_dim
rna_available
selected_targets_complete
```

If multiple slides exist per patient, use a deterministic rule for the pilot.

Preferred order:

1. one slide per patient using a predefined rule;
2. if no defensible rule exists, average slide-level representations at patient level.

Do not allow one patient's slides to appear in different train/test folds.

Add automated tests confirming:

* unique patient IDs;
* no unmatched RNA records;
* no duplicated slide-patient mappings;
* no leakage.

---

# Phase 3 — Create simple slide representations

The pathology input may consist of many patch embeddings per slide.

For the minimum viable model, create slide-level representations using:

## Primary pooling

Mean pooling:

```python
slide_embedding = patch_embeddings.mean(axis=0)
```

## Optional sensitivity pooling

Only if easy:

* mean + standard deviation concatenation;
* median pooling.

Do not develop attention pooling or MIL yet.

Record:

* slide ID;
* patient ID;
* number of patches;
* pooled embedding dimension.

If multiple slides per patient are retained, aggregate slide embeddings to patient level using mean pooling.

---

# Phase 4 — Define the molecular target vector

For each patient, create:

```text
y_i = [
    expression_target_1,
    expression_target_2,
    ...
]
```

Preprocessing must be scientifically transparent.

First inspect whether expression values are:

* raw counts;
* TPM;
* log-transformed values;
* standardized values;
* another representation.

Do not transform until this is known.

Possible preprocessing:

* log1p only when appropriate;
* gene-wise z-score fitted on training data only;
* robust scaling fitted on training data only.

For within-patient ranking, also retain the original or monotonic-transformed expression values.

Do not normalize each patient across targets in a way that destroys biologically meaningful relative differences unless used only for a dedicated ranking sensitivity analysis.

---

# Phase 5 — Train/test design

For the pilot, use simple repeated cross-validation.

Preferred:

* 5-fold patient-level cross-validation;
* repeated 3 times with different seeds if compute is trivial.

If sample size is too small:

* repeated stratified splitting is not appropriate for continuous multi-output targets;
* instead use repeated K-fold CV.

All preprocessing must be fitted within training folds.

No patient may appear in both training and test within a fold.

Store out-of-fold predictions for every patient.

The main analyses should be based on out-of-fold predictions, not training predictions.

---

# Phase 6 — Baseline models

Begin with simple models only.

## Model A — Mean predictor

For each target:

```text
predict training-set mean expression
```

This establishes a trivial baseline.

## Model B — Ridge regression

Use the pooled pathology embedding to predict all selected targets.

Test a very small alpha grid, for example:

```text
0.1
1
10
100
```

Select alpha within training folds only.

## Model C — Elastic Net

Optional only if ridge works.

Use a modest grid.

Do not spend time on extensive hyperparameter tuning.

## Model D — PCA + ridge

Optional sensitivity analysis if embedding dimensionality is much larger than patient count.

Fit PCA on training data only.

Try a small set such as:

```text
16
32
64
128
```

components where feasible.

Do not use the test fold when fitting PCA.

---

# Phase 7 — Conventional gene-level performance

For every selected target, compute out-of-fold:

* Pearson correlation;
* Spearman correlation;
* R²;
* MAE;
* RMSE.

Create a table:

```text
target
pearson_r
spearman_r
r2
mae
rmse
n
```

Use confidence intervals where feasible.

Do not overinterpret modest correlations.

This analysis is secondary.

The main contribution is patient-level target ranking.

---

# Phase 8 — Primary within-patient ranking analysis

For every patient, compare:

```text
true_target_vector
predicted_target_vector
```

Calculate:

## Patient-level Spearman rank correlation

```text
rho_i = Spearman(true_target_order, predicted_target_order)
```

## Patient-level Kendall tau

```text
tau_i
```

## Top-1 target accuracy

Does:

```text
argmax(predicted target expression)
```

equal:

```text
argmax(true target expression)?
```

## Top-2 recall

Is the actual highest-expression target contained in the predicted top two?

## Pairwise ranking accuracy

For every target pair A/B:

```text
true: A > B
predicted: A > B
```

Calculate the fraction of correctly ordered target pairs per patient.

## Optional NDCG

Use only if mathematically appropriate and explain the relevance scores clearly.

Do not include metrics simply because they are common in information retrieval.

---

# Important ranking-scale issue

Different genes may naturally have very different expression ranges.

A naive within-patient ranking across raw gene values may therefore reflect gene-specific measurement scale rather than meaningful relative target biology.

You MUST investigate this before claiming therapeutic ranking.

Run at least two ranking definitions:

## Ranking A — Raw or original transformed expression

Preserves measured cross-gene magnitude.

## Ranking B — Cohort-standardized expression

For each target:

```text
z_ig = (expression_ig - training_target_mean) /
       training_target_sd
```

Then rank each patient's target-specific z-scores.

Interpretation:

> Which targets are unusually high for this patient relative to the population distribution of that target?

This may be the more scientifically defensible ranking.

All standardization parameters must come from training folds only.

Compare both definitions.

If the ranking conclusions differ substantially, report that as a key limitation.

---

# Phase 9 — Therapeutic target prioritization metric

Define an exploratory target-prioritization problem.

For cohort-standardized expression, define the patient's dominant target as:

```text
target with highest target-specific z-score
```

Evaluate:

* top-1 accuracy;
* top-2 recall;
* confusion matrix of true versus predicted dominant targets;
* target-specific recall.

This should be framed as:

> recovery of the relative molecular target landscape

NOT:

> prediction of the optimal ADC

or:

> treatment selection.

RNA expression alone is not sufficient for ADC eligibility.

State this explicitly.

---

# Phase 10 — Simple subtype/confounding control

If subtype or receptor-status labels are easily available in the prepared data or from a simple matched metadata file, test whether the ranking task is merely recovering broad breast-cancer subtype.

Do not create a large TCGA metadata-engineering project if the labels are not readily accessible.

If subtype labels are available:

## Baseline 1

Predict target expression using subtype only.

## Baseline 2

Predict target landscape using pathology embeddings.

## Baseline 3

Subtype + pathology embeddings.

Compare:

* gene-level prediction;
* within-patient ranking;
* top-1 dominant target accuracy.

If pathology improves ranking beyond subtype, report it cautiously.

If subtype explains almost all performance, that is also an important result.

If subtype labels are not readily available by the first modelling day, defer this analysis rather than derail the pilot.

---

# Phase 11 — Cross-target structure

Evaluate whether the model recovers the co-expression structure of the selected targets.

Calculate the true target correlation matrix:

```text
corr_true
```

and the predicted target correlation matrix:

```text
corr_pred
```

Compare using:

* matrix correlation;
* Frobenius norm difference;
* qualitative heatmap comparison.

Also compare target-pair ordering performance.

This asks:

> Does morphology recover the joint target landscape, rather than merely individual target values?

---

# Phase 12 — Critical negative controls

Run at least:

## Shuffled RNA labels

Shuffle patient RNA vectors relative to pathology embeddings.

Performance should collapse.

## Random embeddings

Generate matched-dimensional random features or use a reduced random baseline.

## Mean-only molecular baseline

Predict average expression patterns without pathology input.

If the model does not outperform these controls, do not claim morphology contains informative target-landscape signal.

---

# Phase 13 — Robustness

At minimum:

* repeat cross-validation seeds;
* report variation across folds;
* rerun after removing the weakest-variance target;
* rerun using 4 versus 5 versus 6 targets;
* compare raw-expression ranking versus standardized-expression ranking.

If results depend entirely on one target such as ERBB2, say so explicitly.

That is an important result.

---

# Phase 14 — Main go/no-go criteria

After the simple ridge model is run, classify the project as:

## GO

Proceed to abstract if at least one of these is true:

1. median patient-level ranking correlation is clearly above random;
2. top-1 dominant-target accuracy meaningfully exceeds the class/prevalence baseline;
3. top-2 recall is clearly informative;
4. the model recovers multi-target co-expression structure;
5. morphology adds information beyond an easily available subtype baseline;
6. there is an interesting negative finding showing that individual gene prediction works but within-patient target ranking fails.

## CONDITIONAL GO

Proceed only as a limitations/evaluation paper if:

* individual gene prediction is moderate;
* ranking performance is weak;
* but the contrast demonstrates that conventional biomarker metrics exaggerate therapeutic usefulness.

This could support a title like:

> **Predicting Individual Targets Is Not the Same as Recovering the Therapeutic Target Landscape**

## NO-GO

Stop if:

* individual target prediction is near chance/noise;
* ranking equals random;
* results are unstable;
* matching cannot be verified;
* expression scales make the ranking concept uninterpretable;
* prepared data are incomplete or inconsistent.

If NO-GO, do not add complexity.

Move to a simpler backup project.

---

# Strong possible scientific story A

If ranking works:

> Pathology foundation-model embeddings recovered not only individual ADC-relevant transcript levels but also patient-specific relative target landscapes, supporting the hypothesis that H&E contains multivariate therapeutic-target information beyond isolated biomarker classification.

Use only if supported.

---

# Strong possible scientific story B

If individual prediction works but ranking fails:

> Although pathology embeddings predicted several ADC-relevant transcripts individually, they failed to preserve patient-level target ordering. This distinction suggests that strong single-biomarker performance does not necessarily imply utility for multi-target therapeutic prioritization.

This is potentially a very good workshop result.

---

# Strong possible scientific story C

If subtype explains everything:

> Apparent morphology-based prediction of ADC-relevant targets was largely explained by broad tumour phenotype, emphasizing the need to distinguish target-specific information from correlated lineage structure.

This is less novel but scientifically honest.

---

# Repository structure

Create:

```text
adc_target_landscape/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
├── src/
│   └── adc_landscape/
│       ├── __init__.py
│       ├── data.py
│       ├── targets.py
│       ├── pooling.py
│       ├── models.py
│       ├── ranking.py
│       ├── evaluation.py
│       └── plots.py
├── scripts/
│   ├── audit_data.py
│   ├── audit_targets.py
│   ├── build_manifest.py
│   ├── pool_embeddings.py
│   ├── run_cv.py
│   ├── evaluate_rankings.py
│   └── make_figures.py
├── tests/
│   ├── test_matching.py
│   ├── test_leakage.py
│   ├── test_ranking.py
│   └── test_preprocessing.py
├── results/
│   ├── predictions/
│   ├── metrics/
│   ├── tables/
│   └── figures/
└── notebooks/
    └── 01_results_review.ipynb
```

Keep the implementation small.

Do not add infrastructure unless needed.

---

# Required tests

Implement automated checks verifying:

1. patient IDs match between pathology and RNA;
2. no missing selected-target expression exists after filtering;
3. no patient appears in train and test in the same fold;
4. all gene-wise scaling is fitted using training data only;
5. all PCA is fitted using training data only;
6. out-of-fold predictions exist exactly once per patient per repeat;
7. ranking functions return correct results on synthetic examples;
8. shuffled-label controls are actually shuffled;
9. result tables preserve model, fold, repeat, and target metadata.

---

# Required figures

Produce simple publication-quality figures.

## Figure 1

Study schematic:

```text
TCGA-BRCA H&E
    ↓
precomputed UNI patch embeddings
    ↓
mean pooling
    ↓
multi-output regression
    ↓
ADC-relevant transcript profile
    ↓
within-patient target ranking
```

## Figure 2

Per-target prediction performance.

Use a dot plot or bar plot for:

* Spearman r;
* optionally R².

## Figure 3

Distribution of patient-level ranking correlations.

## Figure 4

True versus predicted dominant target confusion matrix or top-target performance.

## Optional Figure 5

True versus predicted target co-expression matrices.

Do not create decorative plots.

---

# Required tables

## Table 1 — Target audit

```text
target
n
mean
sd
median
iqr
pct_zero
therapeutic_relevance
selected
```

## Table 2 — Individual target prediction

```text
target
pearson
spearman
r2
mae
rmse
```

## Table 3 — Landscape recovery

```text
metric
value
confidence_interval
random_baseline
```

Include:

* median patient Spearman;
* median Kendall tau;
* pairwise ranking accuracy;
* top-1 accuracy;
* top-2 recall.

---

# Statistical principles

Use the patient as the unit of analysis.

Do not treat patches as independent observations.

Use:

* out-of-fold predictions;
* bootstrap confidence intervals at patient level;
* permutation tests for ranking metrics if useful;
* repeated cross-validation for stability.

Do not report dozens of p-values.

The paper is exploratory and preliminary.

Effect sizes and uncertainty are more important.

---

# Interpretation rules

Never say:

> H&E identifies the best ADC.

Never say:

> RNA expression predicts clinical ADC eligibility.

Never say:

> The model can guide treatment selection.

Prefer:

> H&E-derived foundation-model representations were evaluated for their ability to recover relative expression patterns among ADC-relevant molecular targets.

If using standardized target expression, explain:

> The ranking represents relative overexpression of each target compared with its cohort distribution, not absolute protein abundance.

State clearly that:

* RNA is an exploratory molecular phenotype;
* protein abundance may differ;
* membrane localization is not measured;
* spatial heterogeneity is not measured;
* clinical ADC response is not measured.

---

# Workshop abstract objective

The 4-page extended abstract should answer only:

> **Do generic pathology foundation-model embeddings preserve patient-level relative information across multiple therapeutically relevant targets?**

Do not turn the workshop version into:

* a clinical biomarker paper;
* an ADC selection model;
* a multimodal foundation-model paper;
* a treatment-response study.

---

# Possible titles

Do not finalize until results are known.

Potential titles:

* **From Single Biomarkers to Target Landscapes: Can Histology Recover Relative ADC-Relevant Expression?**
* **Can Pathology Foundation Models Recover Patient-Level ADC Target Landscapes from H&E?**
* **Beyond Single-Target Prediction: Evaluating Multi-Target Therapeutic Landscapes in Pathology Foundation Models**
* **Predicting Individual Biomarkers Is Not the Same as Recovering Therapeutic Target Rankings**
* **A Preliminary Evaluation of ADC-Relevant Target Landscape Recovery from Histology Foundation-Model Embeddings**

---

# Immediate execution order

Do exactly this:

## Step 1

Inspect and verify the prepared TANGLE TCGA-BRCA files.

## Step 2

Generate the ADC target audit table.

## Step 3

Confirm patient matching and embedding dimensionality.

## Step 4

Mean-pool embeddings.

## Step 5

Run ridge regression with repeated cross-validation.

## Step 6

Generate out-of-fold target predictions.

## Step 7

Calculate individual-target metrics.

## Step 8

Calculate within-patient ranking metrics.

## Step 9

Run shuffled-label control.

## Step 10

Print a concise GO / CONDITIONAL GO / NO-GO recommendation.

Do not do anything more complicated before completing Step 10.

---

# First response required

Start with the actual feasibility audit.

Do not write modelling code yet.

Return:

1. exact TANGLE data files available;
2. download method;
3. authentication requirement;
4. approximate download size;
5. pathology file structure;
6. RNA file structure;
7. patient identifiers;
8. number of matched patients;
9. embedding dimension;
10. gene naming format;
11. which of the candidate ADC targets exist;
12. distribution statistics for each available target if the data can already be loaded;
13. recommended 4–6 targets;
14. any immediate scientific problem with cross-target ranking;
15. whether raw or cohort-standardized target ranking is more defensible;
16. the exact minimal experiment to run next;
17. GO / CONDITIONAL GO / NO-GO feasibility status.

Do not proceed to modelling until this audit is complete.
