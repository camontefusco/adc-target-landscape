# %% [markdown]
# # ADC target landscape — minimum modelling checkpoint
# Run in Colab after the remote audit. This intentionally stops at mean pooling + Ridge.

# %%
!pip -q install gdown pyyaml

# %%
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

# If cloned in Colab:
# !git clone -b analysis-pipeline https://github.com/camontefusco/adc-target-landscape.git
REPO = Path('/content/adc-target-landscape')
sys.path.insert(0, str(REPO / 'src'))
from adc_landscape.analysis import repeated_ridge_oof, shuffled_label_control, summarize_rankings

TARGETS = ['ERBB2', 'TACSTD2', 'PVRL4', 'FOLR1', 'SLC39A6', 'MET']
DRIVE_ROOT = Path('/content/drive/MyDrive/AI Pathology Biomarkers Project/brca')
MANIFEST = DRIVE_ROOT / 'csvs/tcga_brca.csv'
FEATURE_ROOT = DRIVE_ROOT / 'uni_features/tcga_features'
XENA_URL = 'https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FHiSeqV2_PANCAN.gz'
OUT = REPO / 'results'
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. Load manifest and Xena targets
# This assumes the same manifest columns verified in the feasibility audit. Inspect columns before proceeding.

# %%
manifest = pd.read_csv(MANIFEST)
print(manifest.columns.tolist())
display(manifest.head())

xena_path = Path('/content/HiSeqV2_PANCAN.gz')
if not xena_path.exists():
    import urllib.request
    urllib.request.urlretrieve(XENA_URL, xena_path)
expression = pd.read_csv(xena_path, sep='\t', compression='gzip', index_col=0)
assert set(TARGETS).issubset(expression.index)
primary_cols = [c for c in expression.columns if len(c) >= 15 and c[13:15] == '01']
primary = expression.loc[TARGETS, primary_cols]
primary.columns = [c[:12] for c in primary.columns]
primary = primary.T.groupby(level=0).mean().T

# %% [markdown]
# ## 2. Resolve feature paths
# TANGLE feature filenames can vary. The function tries common slide-ID conventions and fails loudly rather than guessing silently.

# %%
def find_feature(slide_id: str):
    candidates = [
        FEATURE_ROOT / f'{slide_id}.pt',
        FEATURE_ROOT / slide_id,
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = list(FEATURE_ROOT.glob(f'{slide_id}*.pt'))
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f'Could not uniquely resolve UNI tensor for {slide_id}; hits={hits[:5]}')

# Identify columns rather than silently assuming them.
case_col = 'case_id' if 'case_id' in manifest.columns else None
slide_candidates = [c for c in ['slide_id', 'slide', 'filename', 'file_name'] if c in manifest.columns]
assert case_col and slide_candidates, f'Need case_id and slide identifier columns; got {manifest.columns.tolist()}'
slide_col = slide_candidates[0]
manifest['patient_id'] = manifest[case_col].astype(str).str[:12]
manifest = manifest[manifest.patient_id.isin(primary.columns)].copy()
print('Matched manifest patients:', manifest.patient_id.nunique())

# %% [markdown]
# ## 3. Mean-pool each slide, then mean-pool slides per patient
# No patch is treated as an independent sample.

# %%
slide_rows = []
for n, row in enumerate(manifest.itertuples(index=False), 1):
    slide_id = str(getattr(row, slide_col))
    patient_id = str(getattr(row, 'patient_id'))
    p = find_feature(slide_id)
    tensor = torch.load(p, map_location='cpu', weights_only=False)
    arr = tensor.detach().cpu().numpy() if hasattr(tensor, 'detach') else np.asarray(tensor)
    assert arr.ndim == 2 and arr.shape[1] == 1024, (slide_id, arr.shape)
    slide_rows.append((patient_id, slide_id, arr.mean(axis=0), arr.shape[0]))
    if n % 100 == 0:
        print('pooled slides:', n)

patient_vectors = {}
for patient_id, slide_id, vec, n_patches in slide_rows:
    patient_vectors.setdefault(patient_id, []).append(vec)
patient_ids = sorted(patient_vectors)
X = np.vstack([np.mean(patient_vectors[p], axis=0) for p in patient_ids])
Y = primary.loc[TARGETS, patient_ids].T.to_numpy(dtype=float)
print('X:', X.shape, 'Y:', Y.shape)
assert X.shape == (len(patient_ids), 1024)
assert np.isfinite(X).all() and np.isfinite(Y).all()

# Save lightweight pooled data for reruns; do not commit it.
np.savez_compressed(OUT / 'pooled_patient_data.npz', X=X, Y=Y, patient_ids=np.array(patient_ids), targets=np.array(TARGETS))

# %% [markdown]
# ## 4. Minimum repeated-CV Ridge experiment

# %%
pred, ranks, per_target = repeated_ridge_oof(X, Y, TARGETS, n_splits=5, seeds=(17, 41, 73), alpha=10.0, pca_components=128)
pred.to_csv(OUT / 'oof_predictions.csv', index=False)
ranks.to_csv(OUT / 'ranking_metrics.csv', index=False)
per_target.to_csv(OUT / 'per_target_fold_metrics.csv', index=False)
print('\nObserved ranking summary')
display(summarize_rankings(ranks).to_frame('value'))
print('\nPer-target fold summary')
display(per_target.groupby('target')[['spearman','mae','rmse']].agg(['mean','std']).round(3))

# %% [markdown]
# ## 5. Shuffled-patient negative control

# %%
_, ranks_shuf, target_shuf = shuffled_label_control(
    X, Y, TARGETS, seed=2026, n_splits=5, seeds=(17, 41, 73), alpha=10.0, pca_components=128
)
ranks_shuf.to_csv(OUT / 'ranking_metrics_shuffled.csv', index=False)
target_shuf.to_csv(OUT / 'per_target_fold_metrics_shuffled.csv', index=False)
print('\nShuffled ranking summary')
display(summarize_rankings(ranks_shuf).to_frame('value'))

# %% [markdown]
# ## 6. Quick GO / CONDITIONAL GO / NO-GO summary
# This is deliberately heuristic; inspect confidence intervals before using manuscript language.

# %%
obs = summarize_rankings(ranks)
shf = summarize_rankings(ranks_shuf)
delta_pair = obs['mean_pairwise_accuracy'] - shf['mean_pairwise_accuracy']
delta_top2 = obs['top2_recall'] - shf['top2_recall']
if delta_pair >= 0.08 or delta_top2 >= 0.10 or obs['median_patient_spearman'] >= 0.25:
    decision = 'GO'
elif delta_pair >= 0.03 or delta_top2 >= 0.05 or obs['median_patient_spearman'] >= 0.10:
    decision = 'CONDITIONAL GO'
else:
    decision = 'NO-GO / inspect whether individual-target prediction is informative while ranking fails'
print({'decision': decision, 'observed': obs.to_dict(), 'shuffled': shf.to_dict(), 'delta_pairwise': delta_pair, 'delta_top2': delta_top2})
