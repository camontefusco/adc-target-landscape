# Reviewer-response implementation checklist

This checklist implements the OpenReview feedback without changing the central project.

## Primary analysis (unchanged)

- Cohort: TCGA-BRCA, patient-level splits.
- Input: precomputed UNI patch embeddings.
- Patient representation: mean-pooled patches.
- Targets: ERBB2, TACSTD2/TROP2, PVRL4/NECTIN4, FOLR1, SLC39A6/LIV1, MET.
- Model: fold-standardized Ridge regression.
- Primary endpoints: within-patient Spearman correlation and 15-pair ordering accuracy.

## Required robustness analyses

1. Pooling: mean, max, and concatenated mean+max.
2. PCA: PCA and no-PCA, with all transformations fitted inside training folds.
3. Controls: shuffled labels, target-wise mean ranking, and feature-only mean baseline.
4. Pairwise reporting: all 15 pairs, with confidence intervals and exact sample counts.
5. Stability: repeated patient-level five-fold cross-validation with fixed published seeds.
6. Missingness: publish the patient inclusion/exclusion manifest and the one-patient Xena mismatch.

## Optional extension

ABMIL and one pairwise ranking objective may be run as sensitivity analyses. They must use the same folds and must not replace the transparent primary baseline.

## Encoder limitation

The current public TANGLE release supplies UNI embeddings. Do not claim cross-foundation-model robustness unless raw WSIs are available and a second encoder is actually run. If raw slides become available, add one frozen encoder comparison as a separate tier.

## Translation boundary

RNA expression is a transcript-level surrogate. The benchmark must not claim protein abundance, membrane accessibility, internalization, ADC response, or treatment selection. IHC/spatial-protein validation and ADC-treated outcomes are future validation tiers.

## Reporting

For every model/configuration, save:

- fold assignments;
- one out-of-fold prediction per patient and target;
- target-level regression metrics;
- patient-level ranking metrics;
- all pairwise metrics;
- bootstrap and permutation summaries;
- software/configuration metadata.

## Go/no-go rules

GO if signal remains above shuffled controls and is stable across the required pooling/PCA analyses.

CONDITIONAL GO if signal is reproducible only in TCGA but the benchmark is technically robust.

NO-GO for translational claims if signal collapses under controls, is driven by leakage, or has no credible path to external/protein validation.
