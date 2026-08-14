# Codex Prompt: Build a Small-N Pathology Foundation Model Benchmark for a NeurIPS 2026 LXAI Extended Abstract

## Goal

Help me build a minimal, reproducible research project suitable for a **4-page NeurIPS 2026 LatinX in AI workshop extended abstract**.

The project must be executable quickly, with minimal data-engineering and compute risk.

The core scientific question is:

> **How small is too small? Quantifying instability of pathology foundation-model predictions across small oncology cohort sizes.**

The intended message is:

> In small biomedical cohorts, uncertainty caused by which patients happen to be sampled may be as large as, or larger than, the apparent performance gains obtained by changing machine-learning methods.

The project should initially avoid:

* downloading or preprocessing whole-slide images if possible;
* fine-tuning large foundation models;
* multimodal RNA-seq integration;
* LoRA;
* full MIL development;
* large GPU requirements.

Prefer existing **precomputed pathology foundation-model embeddings** and existing labels.

The minimum study should use:

* one pathology foundation model with precomputed embeddings;
* one oncology prediction task;
* patient-level samples;
* training cohort sizes of 25, 50, 100, 200, and full cohort;
* repeated patient subsampling;
* simple downstream classifiers;
* AUROC;
* AUPRC;
* Brier score;
* variability across patient subsets.

If a suitable ImageNet embedding baseline is readily available, include it. Do not create major infrastructure just to add it.

---

# Your role

Act as:

* senior ML research engineer;
* computational pathology scientist;
* biostatistician;
* reproducibility engineer;
* scientific-methods reviewer;
* pair programmer.

You must help me execute the project, not merely describe it.

My Python ability is beginner-to-intermediate, so:

* explain important code decisions;
* keep the implementation modular but simple;
* avoid unnecessary abstractions;
* use readable Python;
* add tests for critical logic;
* explain errors when they occur;
* prefer robust existing libraries over custom implementations.

---

# First task: feasibility audit

Before writing substantial modelling code, determine whether we can execute the entire project using **existing downloadable embeddings plus labels**.

Search current official or primary sources for:

1. precomputed embeddings from pathology foundation models such as:

   * UNI / UNI2;
   * Virchow;
   * Phikon;
   * Prov-GigaPath;
   * other credible pathology foundation models;

2. associated datasets containing:

   * patient IDs;
   * slide IDs;
   * labels;
   * train/test or cohort metadata;

3. tasks suitable for small-N analysis, such as:

   * tumour subtype;
   * MSI;
   * mutation status;
   * receptor status;
   * molecular subtype;
   * another robust binary oncology endpoint.

Rank candidate datasets by:

* direct availability of embeddings;
* direct availability of labels;
* patient-level identifiers;
* number of patients;
* class balance;
* simplicity of loading;
* licensing;
* absence of special data-access approval;
* absence of huge downloads;
* suitability for 25/50/100/200/full subsets.

Do not start coding the pipeline until you have identified the easiest feasible dataset.

Prefer a dataset where I can download a manageable archive or table and start modelling immediately.

Clearly state:

* chosen dataset;
* chosen endpoint;
* chosen embedding source;
* approximate patient count;
* file size;
* license;
* whether patient IDs are available;
* why this is the fastest safe option.

If no such precomputed dataset is usable, identify the next easiest route.

---

# Scientific design

Use one fixed validation/test design and vary only the size of the training cohort.

Preferred design:

1. Create a patient-level development pool.
2. Create one fixed internal test set.
3. If enough data exist, create one fixed validation set.
4. From the training pool, create nested subsets:

```text
25 ⊂ 50 ⊂ 100 ⊂ 200 ⊂ Full
```

Repeat this with multiple independent patient-subset seeds.

Minimum:

* 5 subset seeds.

Preferred:

* 10 subset seeds.

Use the same test set for every experiment.

For every subset:

* preserve class balance where feasible;
* do not leak patients across train/test;
* do not sample slides independently if several belong to one patient;
* aggregate or select slides using a deterministic patient-level rule.

The unit of analysis must be the patient unless the dataset clearly supports a slide-level endpoint.

---

# Models

Start simple.

## Model 1: Logistic regression

Use frozen embeddings with:

* standardization fitted on training data only;
* regularized logistic regression;
* class weighting only if justified.

This should be the main model.

## Model 2: Shallow MLP

Only after logistic regression works.

Use:

* one hidden layer;
* strong regularization;
* early stopping;
* fixed small architecture.

Do not add deep architectures unless absolutely necessary.

If precomputed ImageNet embeddings are readily available, repeat the logistic-regression analysis with them as a baseline.

---

# Main experiment

For each training cohort size:

* 25;
* 50;
* 100;
* 200;
* full training cohort;

and for each patient-subset seed:

1. train the model;
2. evaluate on the same fixed test set;
3. store predictions;
4. store metrics;
5. store calibration results;
6. store selected hyperparameters;
7. store random seed.

The primary outputs are:

* AUROC;
* AUPRC;
* Brier score;
* calibration slope if sample size permits;
* sensitivity;
* specificity;
* balanced accuracy;
* model coefficient norm or another simple complexity diagnostic;
* runtime.

---

# Primary scientific quantity

Compute the variability caused by patient sampling.

For each N:

```text
Var_subset(AUROC)
Var_subset(AUPRC)
Var_subset(Brier)
```

Also compute:

* standard deviation;
* interquartile range;
* 95% bootstrap or empirical interval;
* best-minus-worst performance across subset seeds.

The main scientific comparison should ask:

> How large is sampling-induced performance variability relative to the apparent improvement obtained by increasing N or changing the downstream classifier?

Define a provisional quantity such as:

```text
Sampling Instability = SD of test AUROC across patient-subset seeds
```

and compare it with:

```text
Method Gain = mean AUROC(MLP) - mean AUROC(Logistic Regression)
```

and:

```text
Sample-Size Gain = mean AUROC(N=100) - mean AUROC(N=50)
```

Do not invent a composite index unless it is clearly labelled exploratory.

---

# Hypotheses

Test these hypotheses:

1. Performance variability will be largest at N=25 and N=50.
2. Sampling variability may exceed the gain from switching from logistic regression to a shallow MLP.
3. Calibration will deteriorate faster than AUROC as N decreases.
4. Some small cohorts will yield deceptively high AUROC purely because of patient composition.
5. Increasing N will reduce both average error and between-subset variance.

Treat these as hypotheses, not expected conclusions.

---

# Reproducibility requirements

Create a clean repository:

```text
small_n_pathology/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
├── src/
│   └── small_n_pathology/
│       ├── __init__.py
│       ├── data.py
│       ├── splits.py
│       ├── models.py
│       ├── metrics.py
│       ├── calibration.py
│       ├── experiment.py
│       └── plots.py
├── scripts/
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── create_splits.py
│   ├── run_experiments.py
│   └── make_figures.py
├── tests/
│   ├── test_splits.py
│   ├── test_no_leakage.py
│   └── test_metrics.py
├── results/
│   ├── metrics/
│   ├── predictions/
│   └── figures/
└── notebooks/
    └── 01_results_review.ipynb
```

Keep notebooks for inspection only.

Core experiment logic must live in Python modules.

---

# Required automated checks

Implement tests that verify:

1. no patient appears in both train and test;
2. nested subsets are truly nested;
3. the same test set is used across all N values;
4. class labels are valid;
5. no duplicate patient IDs exist after aggregation;
6. scaling is fitted on training data only;
7. test labels are never used in model fitting;
8. all experiment results include the subset seed and training size.

Stop execution if leakage is detected.

---

# Configuration

Use a YAML or TOML config with fields such as:

```yaml
dataset:
  name: ...
  embedding_path: ...
  label_column: ...
  patient_id_column: ...
  slide_id_column: ...

splits:
  test_fraction: 0.2
  validation_fraction: 0.1
  subset_sizes: [25, 50, 100, 200]
  subset_seeds: [1, 2, 3, 4, 5]

models:
  logistic_regression:
    C_values: [0.01, 0.1, 1, 10]

  mlp:
    hidden_dim: 64
    dropout: 0.3
    max_epochs: 100

metrics:
  - auroc
  - auprc
  - brier
  - balanced_accuracy
```

Adjust values based on the actual dataset.

---

# Hyperparameter strategy

Keep hyperparameter search deliberately small.

For logistic regression:

* test only a small prespecified set of C values.

For MLP:

* use one or two learning rates;
* one fixed hidden dimension;
* one dropout range.

Do not use massive Optuna searches.

The purpose of the paper is sampling instability, not optimization.

---

# Figures

Generate publication-quality figures using matplotlib only.

Do not use seaborn.

Create separate figures, not subplots unless truly necessary.

Do not manually specify colors unless needed.

Required figures:

## Figure 1: Study design

Show:

```text
Full training pool
      ↓
Repeated patient subsampling
      ↓
25 / 50 / 100 / 200 / full
      ↓
Frozen PFM embeddings
      ↓
Simple classifier
      ↓
Fixed test set
```

## Figure 2: Learning curve

X-axis:

* number of training patients

Y-axis:

* AUROC

Display:

* mean or median;
* variability interval;
* individual subset runs if readable.

## Figure 3: Calibration or Brier score versus N

Show whether calibration deteriorates as sample size decreases.

## Figure 4: Sampling instability

X-axis:

* N

Y-axis:

* standard deviation or IQR of AUROC across patient subsets.

## Optional Figure 5

Compare:

```text
Sampling variability
vs
Method gain
```

to test whether changing patient composition has a larger effect than changing the downstream classifier.

---

# Tables

Create:

## Table 1: Dataset

Columns:

* patients;
* slides;
* endpoint;
* positive class;
* negative class;
* class prevalence;
* embedding dimension.

## Table 2: Performance

Rows:

* N=25;
* 50;
* 100;
* 200;
* full.

Columns:

* mean AUROC;
* SD AUROC;
* mean AUPRC;
* mean Brier;
* best AUROC;
* worst AUROC.

## Table 3: Method comparison

Compare logistic regression and MLP.

---

# Statistical analysis

Keep the statistics simple and defensible.

Use:

* paired comparisons because the same subset/test framework is reused;
* bootstrap confidence intervals where appropriate;
* empirical variability across subset seeds;
* Spearman or regression analysis of N versus instability if useful.

Do not use tile-level sample sizes as independent observations.

Do not report p-values from thousands of embeddings.

The patient is the statistical unit.

---

# Interpretation rules

If performance decreases with smaller N:

Do not say:

> Small cohorts make foundation models fail.

Say:

> Downstream predictive performance and stability deteriorated as fewer labelled patients were available in this cohort.

If some N=25 subsets perform very well:

Investigate whether this is due to:

* class balance;
* disease subtype;
* site;
* scanner;
* age;
* stage;
* other available metadata.

Do not cherry-pick the best subset.

If logistic regression performs as well as MLP:

Treat that as an important result.

If the MLP performs better on average but has greater variance:

Highlight the performance–stability trade-off.

---

# Workshop abstract objective

The eventual extended abstract should make one clear claim.

A possible conclusion structure is:

> Across repeated small patient cohorts, model performance varied substantially as a function of cohort composition. This variability was greatest below approximately [observed N] patients and in some settings exceeded the gain obtained by increasing downstream model complexity. These preliminary findings suggest that repeated patient-level subsampling and stability reporting should accompany performance estimates when pathology foundation models are evaluated in small biomedical cohorts.

Only use this wording if the actual results support it.

---

# Potential title options

Evaluate:

* **How Small Is Too Small? Stability of Pathology Foundation Models Across Oncology Cohort Sizes**
* **Patient Sampling Uncertainty in Small-Cohort Pathology Foundation-Model Studies**
* **When Cohort Composition Matters More Than Model Choice: A Small-N Pathology Foundation-Model Study**
* **Quantifying Predictive Instability of Pathology Foundation Models Under Limited Oncology Sample Sizes**
* **Beyond AUROC: Sampling Stability of Foundation-Model Biomarkers in Small Oncology Cohorts**

Do not finalize the title before seeing the results.

---

# Deliverables

By the end, produce:

1. working GitHub-ready repository;
2. reproducible dataset preparation;
3. patient-level nested subsets;
4. experiment result CSV;
5. prediction-level CSV;
6. figures;
7. tables;
8. concise methods description;
9. concise results summary;
10. 4-page NeurIPS-style extended abstract draft outline;
11. poster outline;
12. limitations;
13. next-step plan for the full transfer-learning paper.

---

# Important scope control

If you encounter a major hurdle, do not expand the project.

Prefer, in order:

1. use a simpler endpoint;
2. use one foundation model;
3. remove the MLP;
4. reduce subset seeds from 10 to 5;
5. use fewer N values;
6. use a smaller dataset.

Do not respond to a hurdle by adding:

* RNA-seq;
* multiple foundation models;
* LoRA;
* full fine-tuning;
* MIL;
* Docker complexity;
* complex hyperparameter optimization.

The priority is a **clean, defensible preliminary result quickly**.

---

# First response required

Start with the feasibility audit only.

Do not write modelling code yet.

Your first response must provide:

1. the easiest currently accessible precomputed pathology embedding dataset;
2. the associated oncology endpoint;
3. number of patients;
4. number of slides;
5. embedding dimensions;
6. class distribution if available;
7. download size;
8. exact access method;
9. license;
10. whether patient IDs and labels are directly available;
11. whether any authentication is required;
12. whether a fixed external test set is available;
13. the next-best dataset if the first one fails;
14. the exact minimum experiment that can realistically be completed first;
15. the first three implementation steps.

Use official repositories, primary model publications, and official dataset documentation.

Clearly distinguish verified facts from provisional estimates.
