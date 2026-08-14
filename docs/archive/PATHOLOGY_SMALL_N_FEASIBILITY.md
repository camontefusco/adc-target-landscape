# Feasibility audit

Status: pending source verification.

This document will record verified facts separately from provisional estimates before modelling begins.

## Required decision fields

| Field | Selected candidate | Next-best candidate |
|---|---|---|
| Embedding dataset | Pending | Pending |
| Oncology endpoint | Pending | Pending |
| Patients / slides | Pending | Pending |
| Embedding dimension | Pending | Pending |
| Class distribution | Pending | Pending |
| Download size | Pending | Pending |
| Exact access method | Pending | Pending |
| License | Pending | Pending |
| Patient IDs and labels | Pending | Pending |
| Authentication | Pending | Pending |
| Fixed external test set | Pending | Pending |

## Feasibility gate

No modelling pipeline is implemented until one candidate has downloadable embeddings and labels, patient-level identity, acceptable licensing/access requirements, and enough patients for the planned nested subsets.

## Minimum first experiment

To be finalized after dataset selection: one embedding source, one binary oncology endpoint, one fixed patient-level test set, five repeated nested subset seeds, and regularized logistic regression.
