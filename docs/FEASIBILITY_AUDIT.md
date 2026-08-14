# Phase 0 feasibility audit

Audit date: 2026-08-14
Decision: **CONDITIONAL GO**

## Bottom line

The pathology side is usable without WSI preprocessing: the official TANGLE Drive exposes per-slide UNI patch-embedding tensors for TCGA-BRCA. Patient and slide identity are preserved in filenames and in a small CSV manifest.

The prepared RNA tensors are not sufficient for the proposed ADC analysis because they are anonymous 4,999-value tensors and the repository does not provide the ordered gene-symbol list. Indexing a target by guessed position would be scientifically unsafe. The minimal defensible fallback is to use the official, public UCSC Xena named expression matrix and match it to slides with TCGA patient IDs.

## Verified TANGLE files

Official source: <https://github.com/mahmoodlab/TANGLE>

Remote root: <https://drive.google.com/drive/folders/1GIJEITf5-7lFKil7Dfi3sSmVFgzh-otv>

```text
brca/
├── csvs/
│   └── tcga_brca.csv
├── rna/
│   └── <slide_id>.pt
└── uni_features/
    ├── tcga_features/
    │   └── <slide_id>.pt
    └── bcnb_features/
```

The manifest is 88,092 bytes and contains 1,049 slide rows, 984 unique `case_id` values, and 1,049 unique `slide_id` values. Sixty-five rows belong to patients with more than one slide; patient-level aggregation is therefore required.

The Drive connector enumerates at most 100 children for these large folders. The first 100 UNI files total 5.28 GB and the first 100 RNA files total 1.88 GB. Extrapolation from those observed files suggests roughly 55 GB of UNI tensors plus 19.7 GB of RNA tensors for 1,049 slides; this is an estimate, not a verified folder total.

No Google authentication is required for public read access. A no-copy shortcut was added at:

```text
My Drive/AI Pathology Biomarkers Project/brca
```

## Tensor inspection

A temporary matched-format sample was inspected and then deleted locally.

- UNI tensor: float32 patch matrix with shape `(n_patches, 1024)`; the inspected slide contained 16,448 patches.
- RNA tensor: float32 vector with 4,999 values.
- Filenames preserve full TCGA slide identity, e.g. `TCGA-A7-A2KD-01Z-00-DX1.<uuid>.pt`.
- The TANGLE loader pairs modalities by exact filename intersection.
- Neither the tensor nor the repository includes gene symbols or a 4,999-position gene map.

## Expression fallback

Dataset: `TCGA.BRCA.sampleMap/HiSeqV2_PANCAN`

Official host: UCSC Xena TCGA hub

Access: public, no authentication
Compressed size: 83,389,754 bytes (about 79.5 MiB)

Xena describes the values as TCGA Level-3 RSEM normalized results, transformed with `log2(x+1)` and then mean-centered per gene across TCGA cancer cohorts. The matrix uses gene symbols as row identifiers and TCGA sample barcodes as columns.

Matching the TANGLE manifest to primary-tumour (`-01`) Xena samples yielded 983 named-expression patients out of 984 manifest patients. The one-patient discrepancy must be identified explicitly during manifest construction.

## Candidate target audit

Statistics below use the 983 matched primary-tumour Xena samples. These values are pan-cancer mean-centered log-expression, so zeros are not biological non-expression and coefficient of variation is not meaningful when a gene mean is near zero. The table is retained as an audit of variation, not as a biological abundance comparison.

| Target | n | Mean | SD | Median | IQR | P05 | P95 | Skew | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ERBB2 | 983 | 1.446 | 1.482 | 1.235 | 1.239 | -0.406 | 5.254 | 1.314 | usable |
| TACSTD2 | 983 | 2.763 | 1.286 | 2.923 | 1.262 | 0.606 | 4.431 | -1.699 | usable |
| ERBB3 | 983 | 1.850 | 0.866 | 1.963 | 1.077 | 0.143 | 3.048 | -0.730 | usable |
| FOLR1 | 983 | -0.967 | 2.959 | -1.020 | 4.568 | -5.721 | 4.157 | 0.227 | usable |
| NECTIN4 | — | — | — | — | — | — | — | — | absent under current symbol |
| PVRL4 (NECTIN4 alias) | 983 | 3.156 | 0.991 | 3.173 | 1.220 | 1.552 | 4.635 | -0.813 | usable alias |
| CD276 | 983 | 0.413 | 0.578 | 0.418 | 0.768 | -0.567 | 1.322 | -0.194 | lower variance |
| MET | 983 | -1.693 | 1.640 | -1.578 | 2.291 | -4.519 | 0.819 | -0.239 | usable |
| EGFR | 983 | -2.039 | 1.963 | -1.983 | 2.611 | -5.379 | 0.854 | -0.062 | usable |
| CEACAM5 | 983 | 0.752 | 3.661 | 0.974 | 5.251 | -5.440 | 6.821 | 0.009 | usable; very broad |
| F3 | 983 | -0.940 | 1.443 | -0.991 | 2.033 | -3.129 | 1.408 | -0.041 | usable |
| SLC39A6 (LIV1) | 983 | 2.581 | 1.804 | 2.577 | 2.832 | -0.158 | 5.618 | 0.064 | usable alias |
| MSLN | 983 | -2.266 | 3.364 | -3.409 | 3.501 | -5.875 | 5.400 | 1.423 | usable; skewed |

Recommended six-target pilot set: **ERBB2, TACSTD2, PVRL4, FOLR1, SLC39A6, and MET**. This set balances therapeutic relevance, expression variation, and non-identical distributions. A four-target sensitivity analysis should use ERBB2, TACSTD2, PVRL4, and FOLR1.

## Ranking-scale decision

Raw cross-gene ranking is not the primary endpoint. The Xena matrix is mean-centered independently per gene, and different genes retain different dynamic ranges. A raw ranking would therefore mix patient biology with gene-specific measurement scale.

The primary ranking will use target-wise standardization fitted within each training fold:

```text
z_ig = (expression_ig - training_mean_g) / training_sd_g
```

This answers which target is unusually high for a patient relative to that target's cohort distribution. Raw-value ranking will be reported only as a sensitivity analysis with an explicit scale warning.

## Exact minimum experiment after the audit gate

1. Select one deterministic diagnostic slide per patient where possible; otherwise mean-pool slides after patch mean-pooling.
2. Mean-pool each remote UNI patch tensor to a 1,024-dimensional slide vector in Colab.
3. Match the resulting patient table to the six named Xena targets.
4. Run 5-fold patient-level cross-validation with ridge regression and alphas `[0.1, 1, 10, 100]`, selected inside each training fold.
5. Store exactly one out-of-fold prediction per patient.
6. Evaluate per-target metrics and standardized within-patient Spearman, Kendall tau, pairwise accuracy, top-1 accuracy, and top-2 recall.
7. Run mean-only and shuffled-RNA controls.
8. Issue GO / CONDITIONAL GO / NO-GO without adding model complexity.

## Conditions before modeling

- Reproduce tensor shapes and full filename intersections in Colab from the Drive shortcut.
- Identify the one manifest patient missing from the Xena primary-tumour match.
- Confirm no slide from one patient can cross folds.
- Preserve the old aliases `PVRL4` and `SLC39A6` in the gene-selection record.
- Do not use the anonymous prepared RNA tensor for named-target modeling unless the authors provide the exact ordered gene list.

## Licensing and access caveat

The TANGLE repository carries CC BY-NC-ND 4.0. TCGA/Xena data are public, but downstream redistribution and attribution requirements must be reviewed before sharing derived artifacts. This audit is technical, not legal advice.
