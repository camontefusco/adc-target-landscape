# Pilot results: ADC target landscape recovery

## Status

**GO** for an LXAI 2026 workshop abstract, subject to a small set of confirmatory analyses before manuscript wording is finalized.

## Dataset and representation

- TCGA-BRCA pathology manifest matched to UCSC Xena primary-tumour expression.
- 983 matched patients.
- UNI pathology representation: 1,024-dimensional patient-level vector after mean pooling patch embeddings within slides and averaging multiple slides per patient.
- Six ADC-relevant targets: `ERBB2`, `TACSTD2`, `PVRL4` (NECTIN4), `FOLR1`, `SLC39A6` (LIV1), and `MET`.
- Cross-target ranking used target-wise z-scores computed from training-fold statistics only.
- Primary model: PCA (128 components) + Ridge regression, evaluated with repeated 5-fold patient-level cross-validation over seeds 17, 41, and 73.

## Main observed ranking results

| Metric | Observed | Shuffled control | Difference |
|---|---:|---:|---:|
| Median patient Spearman rho | 0.3714 | -0.0286 | +0.4000 |
| Median patient Kendall tau | 0.2000 | -0.0667 | +0.2667 |
| Mean pairwise ranking accuracy | 0.6080 | 0.4954 | +0.1126 |
| Top-1 target accuracy | 0.2984 | 0.1726 | +0.1258 |
| Top-2 target recall | 0.5148 | 0.3394 | +0.1753 |

For six targets, the shuffled control is close to the intuitive random baselines of ~16.7% for top-1 and ~33.3% for top-2. The observed model substantially exceeds both shuffled values.

## Individual target prediction

Mean fold-level Spearman correlations:

| Target | Spearman rho mean | SD |
|---|---:|---:|
| `SLC39A6` | 0.566 | 0.060 |
| `MET` | 0.430 | 0.040 |
| `FOLR1` | 0.374 | 0.050 |
| `ERBB2` | 0.278 | 0.053 |
| `PVRL4` | 0.275 | 0.050 |
| `TACSTD2` | 0.221 | 0.062 |

The morphology-derived UNI representation therefore contains predictive information for all six targets, but the signal strength is heterogeneous and strongest for SLC39A6 and MET in this pilot.

## Current interpretation

The result supports the central pilot hypothesis: generic pathology foundation-model embeddings preserve information about the **relative multi-target molecular landscape**, not only isolated target values. The signal is moderate rather than clinically decisive, but it is clearly above a shuffled-patient control across several complementary ranking metrics.

The workshop claim should remain narrow:

> H&E-derived UNI representations contained sufficient information to recover patient-level relative expression patterns across six ADC-relevant transcripts in TCGA-BRCA, with ranking performance exceeding shuffled-patient controls.

This does **not** establish ADC eligibility, optimal ADC selection, protein abundance, membrane localization, spatial heterogeneity, or treatment response.

## Important limitations before abstract submission

1. RNA expression is an exploratory molecular phenotype and is not equivalent to target protein abundance or clinical ADC eligibility.
2. The current analysis uses TCGA-BRCA only; there is no independent external validation yet.
3. The present summary pools repeated CV predictions across three split seeds. Confirmatory uncertainty estimates should respect repeated measurements of the same patient.
4. The target panel is small and partly breast-cancer-specific; SLC39A6 contributes particularly strong signal and should not be allowed to dominate the overall ranking result unnoticed.
5. Mean pooling intentionally ignores spatial heterogeneity and tumour-region selection.
6. Target-wise z-scoring defines relative overexpression versus the cohort distribution, not absolute cross-gene protein abundance.

## Next analyses before writing the final abstract

### Essential

1. Add patient-level confidence intervals or a permutation test for the main ranking metrics.
2. Run sensitivity analyses after removing each target in turn, especially `SLC39A6`, to confirm that target-landscape recovery is not driven by one easy target.
3. Compare 4-, 5-, and 6-target panels.
4. Save a target-level shuffled-control table, not only the aggregate ranking control.
5. Generate the final workshop figures from out-of-fold predictions.

### Strongly desirable if easy

6. Add a simple broad-phenotype/subtype baseline if reliable TCGA subtype metadata are available without major data-engineering work.
7. Compare pathology embeddings against a simple non-image baseline to test incremental information.
8. Report which target is most often the true and predicted top-ranked target to identify class/prevalence imbalance.

### Defer to the full paper

- MIL or attention pooling.
- Foundation-model comparisons.
- Protein/IHC validation.
- External cohorts.
- Spatial target landscapes.
- Direct ADC treatment-response data.

## Recommended workshop framing

Working title:

**From Single Biomarkers to Target Landscapes: Can Histology Recover Relative ADC-Relevant Expression?**

Alternative if sensitivity analysis reveals uneven target contributions:

**Beyond Single-Target Prediction: Evaluating Multi-Target Therapeutic Landscapes in Pathology Foundation-Model Embeddings**
