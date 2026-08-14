# ADC Target Landscape

Rapid go/no-go pilot testing whether precomputed pathology foundation-model representations recover relative ADC-relevant transcript landscapes within TCGA-BRCA patients.

## Current phase

Phase 0 feasibility audit is complete. No modeling code has been added yet.

The prepared TANGLE data are accessible, but the supplied RNA tensors do not include a gene-symbol map. The pilot will therefore use:

- remote TANGLE TCGA-BRCA UNI patch embeddings;
- the official UCSC Xena `HiSeqV2_PANCAN` expression matrix for named targets;
- TCGA patient IDs as the matching key;
- Google Colab for computation, with the TANGLE folder available through a Drive shortcut;
- the local/GitHub repository only for code, configuration, manifests, and lightweight results.

Read [the feasibility audit](docs/FEASIBILITY_AUDIT.md) before implementing the model.

## Scientific question

> Can histology-derived foundation-model representations recover the relative landscape of ADC-relevant target expression within individual breast cancer patients?

The primary analysis is a joint within-patient ranking task. It is not an ADC treatment-selection model and does not establish protein abundance, membrane localization, eligibility, or clinical response.

## Remote data

- TANGLE Drive root: <https://drive.google.com/drive/folders/1GIJEITf5-7lFKil7Dfi3sSmVFgzh-otv>
- Drive shortcut: `My Drive/AI Pathology Biomarkers Project/brca`
- TANGLE repository: <https://github.com/mahmoodlab/TANGLE>
- Xena expression dataset: `TCGA.BRCA.sampleMap/HiSeqV2_PANCAN`

Large data must not be committed or permanently copied into this repository.

## Planned execution order

1. Reproduce the remote file audit in Colab.
2. Build a lightweight slide/patient manifest.
3. Verify target symbols and select 4–6 targets.
4. Mean-pool UNI patch embeddings in Colab.
5. Run the minimum ridge and negative-control experiment only after the audit checkpoint is accepted.

## Repository layout

Core experiment modules will live in `src/adc_landscape/`; notebooks are inspection and remote-execution entrypoints only. Generated predictions, metrics, tables, and figures are ignored by Git.
