# Interpretability protocol

The primary interpretability analysis uses linear patch contributions from the fitted Ridge model.

For patch embedding z_p, training-fold scaler (mu, sigma), and target coefficient beta:

    contribution_p = ((z_p - mu) / sigma) dot beta

For PCA models, coefficients must first be back-projected to the original UNI space. Contributions are ranked within each slide, and the top/bottom patches are exported.

## Reporting boundary

- If patch coordinates and source tile images are available, render contribution heatmaps and show top/bottom tiles.
- If only an ordered embedding tensor is available, report patch indices and contribution values; do not call them slide heatmaps.
- Contributions are model associations, not causal biological explanations.
- Validate attribution with occlusion or removal of high-contribution patches versus random patches.
- Generate explanations from held-out models/patients and never use ground-truth target values to select displayed patches.

The implementation is in scripts/patch_contributions.py.
