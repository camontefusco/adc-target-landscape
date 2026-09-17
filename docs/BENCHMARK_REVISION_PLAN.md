# Benchmark revision plan

## Scope

This revision keeps the original scientific question, TCGA-BRCA cohort, six ADC-relevant targets, precomputed UNI features, and mean-pooling + Ridge as the primary baseline. The contribution is a reproducible benchmark for within-patient target-landscape ranking, not a treatment-selection model.

## Locked task definition

For patient i and target g, let y_ig be the named Xena expression value and let yhat_ig be the out-of-fold prediction from histology. The patient-specific predicted landscape is the ordering of yhat_i1,...,yhat_i6. Primary endpoints are within-patient Spearman correlation and pairwise ordering accuracy across all 15 target pairs. Secondary endpoints are Kendall correlation, top-1 accuracy, and top-2 recall.

## Primary protocol

1. Use one deterministic slide per patient (lexicographically first slide) for the first benchmark; report a mean-pooled multi-slide sensitivity analysis.
2. Split by patient, never by slide.
3. Fit scaling, PCA, and alpha selection inside training folds only.
4. Keep the six targets and aliases fixed: TROP2/TACSTD2, NECTIN4/PVRL4, LIV1/SLC39A6, ERBB2, FOLR1, MET.
5. Store exactly one out-of-fold prediction per patient.
6. Compare against shuffled-label, mean-only, and target-wise-mean ranking controls.
7. Report every pairwise result with confidence intervals, including pairs near chance.
8. Use bootstrap confidence intervals and permutation tests; do not interpret them as external validation.

## Required ablations

- UNI mean pooling (primary)
- UNI max pooling
- UNI mean + max pooling
- PCA versus no PCA
- one pairwise ranking objective as exploratory sensitivity analysis

Attention-based MIL is optional and should not replace the primary model. A second foundation model is out of scope unless raw WSIs and a reproducible feature-extraction path become available.

## Validation tiers

- Tier 1: fixed held-out patients and subtype/site holdouts where metadata permit.
- Tier 2: external cohort validation.
- Tier 3: protein/IHC or spatial-protein ranking validation.
- ADC treatment response is future work, not an endpoint of this benchmark.

## Interpretation rules

Performance above shuffled controls supports measurable histology-to-transcript ranking signal. It does not establish protein-level target availability, ADC efficacy, or clinical utility. If a target pair is at chance, report it as pair-specific failure rather than averaging it away.

## Deliverables

- machine-readable fold assignments and predictions;
- metrics table for all targets and pairs;
- pooling/PCA ablation table;
- permutation and bootstrap summaries;
- exact environment/configuration;
- a limitations section covering the single cohort, RNA surrogate, and single encoder.
