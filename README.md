# Small-N Pathology Foundation Model Benchmark

Minimal, reproducible research project for studying how patient-cohort size and composition affect downstream predictions from frozen pathology foundation-model embeddings.

## Status

Phase 1: feasibility audit. In accordance with the project brief, modelling code will not be added until an accessible embedding dataset, oncology endpoint, patient identifiers, labels, licensing, and download requirements have been verified from primary sources.

## Research question

> How small is too small? Quantifying instability of pathology foundation-model predictions across small oncology cohort sizes.

The intended experiment uses one fixed patient-level test set, repeated nested training subsets of 25, 50, 100, 200, and the full development cohort, and simple downstream classifiers. The primary outcome is variability across sampled patient cohorts, alongside AUROC, AUPRC, and Brier score.

## Repository map

- `docs/PROJECT_BRIEF.md`: complete research and implementation brief.
- `docs/FEASIBILITY_AUDIT.md`: evidence table and dataset decision record.
- `data/`: ignored local raw/processed data and versionable lightweight manifests.
- `configs/`, `src/`, `scripts/`, `tests/`: reserved for implementation after the feasibility gate.
- `results/`: ignored generated metrics, predictions, and figures.

## Reproducibility policy

- The patient is the unit of analysis.
- No patient may cross train, validation, or test boundaries.
- The fixed test set is reused across all training sizes and subset seeds.
- Data and generated results are not committed unless they are small, redistributable, and explicitly approved.

## Next milestone

Complete and review the feasibility audit, choose the fastest safe dataset, then implement only the minimum end-to-end logistic-regression experiment.
