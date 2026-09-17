# Colab runbook

1. Open `notebooks/02_locked_benchmark_runner.ipynb` in Colab.
2. Run the cells in order and approve the Google Drive mount.
3. Confirm that `patient_labels.csv` reports approximately 983 patients.
4. Confirm that the feature cache contains arrays of shape `(n_patients, 1024)`.
5. Confirm that `oof_predictions.csv` contains 6 targets per patient and all four pooling/PCA combinations.
6. Run the evaluator and retain `patient_ranking_metrics.csv` and `ranking_summary.csv`.

Expected outputs:

- `oof_predictions.csv`: one out-of-fold prediction per patient, target, pooling method, and PCA setting.
- `patient_ranking_metrics.csv`: patient-level Spearman, Kendall, pairwise accuracy, top-1, and top-2 metrics.
- `ranking_summary.csv`: aggregate mean/median/count summaries.

Quality checks:

- no patient appears in more than one fold;
- no missing target values after label preparation;
- no PCA or scaling fit uses held-out patients;
- report all 15 target pairs;
- compare against shuffled-label and target-mean controls before interpreting any gain.

If the Drive mount is slow, start by editing `patient_labels.csv` to a 50–100-patient pilot subset. The cache and modeling stages should then complete quickly; run the full cohort only after the pilot passes.
