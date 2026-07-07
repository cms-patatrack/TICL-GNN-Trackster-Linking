# Dummy Reconstruction Experiment

This workflow is meant for poster-safe experiments when CMS event data cannot be shown or used.
It creates public HGCAL-like dummy events with known truth labels, trains two models on the same
split, and evaluates both edge classification and reconstructed supertracksters.

## Thesis Anchors

The dummy event generator follows the baseline thesis setup:

- HGCAL endcap coverage: `1.5 <= |eta| <= 3.0`, full `phi`, 47 layers per endcap.
- Densities use the thesis convention `2 * (3 - 1.5) * (2 * 47)`.
- Events use one fixed crowded multiparticle topology with about 200 PU showers.
- Hard and PU axes are deliberately sampled around shared activity centres, so showers overlap in eta-phi.
- Shower fragments include depth-dependent drift, widening, heavy-tailed angular scatter, and broad PU timing.
- The event graph uses the thesis eta-phi window of `0.2`.
- Model outputs are thresholded into connected components, interpreted as supertracksters.

## One-Command Experiment

```bash
python scripts/run_dummy_reco_experiment.py \
  --generate-data \
  --epochs 30
```

For a quick smoke run:

```bash
python scripts/run_dummy_reco_experiment.py \
  --run-name dummy_reco_smoke \
  --generate-data \
  --train-files 2 \
  --val-files 1 \
  --test-files 1 \
  --events-per-file 2 \
  --epochs 1 \
  --limit-train 4 \
  --limit-val 2 \
  --limit-test 2
```

## Outputs

The script uses the same `training_data` and `linking_dataset` subfolder layout as the regular
training scripts, but roots all dummy artifacts in `../data`:
`model_folder = ../data/training_data/dummy_reco_experiment`, and
`data_folder = ../data/linking_dataset/dummy_reco_experiment`.

The experiment writes:

- `../data/linking_dataset/dummy_reco_experiment/histo`: raw train/validation/test parquet files.
- `../data/linking_dataset/dummy_reco_experiment/dataset_dummy_reco*`: processed train/validation/test graph datasets.
- `../data/training_data/dummy_reco_experiment/focal` and `../data/training_data/dummy_reco_experiment/focal_contrastive`: standard `_dict.pt` and traced checkpoints from `save_model`, plus per-model training-loss and validation plots.
- `../data/training_data/dummy_reco_experiment/metrics.json` and `metrics.csv`: poster/table-ready metrics.
- `../data/training_data/dummy_reco_experiment/loss_comparison.png`: focal vs focal+contrastive training curves.
- `../data/training_data/dummy_reco_experiment/reconstruction_metric_bars.png`: held-out reconstruction metrics.

## Poster Stability Plots

After the focal-only and focal+contrastive checkpoints exist, rerun the perturbation-stability
plots used by the LogML poster:

```bash
python scripts/stabilityAnalysis/dummyEdgeStability.py \
  --num-graphs 100 \
  --num-perturbations 50
```

This writes:

- `../data/training_data/dummy_reco_experiment_edge_stability/all_edge_stab.png`
- `../data/training_data/dummy_reco_experiment_edge_stability/signal_edge_stab.png`
- `../data/training_data/dummy_reco_experiment_edge_stability/dummy_edge_stability_summary.json`

Pass `--output-dir LogML_GNN_Poster/images` only when you want to copy the same plots directly
into the poster image folder.

The heatmap value is `focal flip rate - focal+contrastive flip rate` under the same transverse
PCA perturbations. Positive blue regions mean the contrastive model is more stable; negative red
regions mean it flips more often than the focal baseline.

## Poster Metrics

Use the edge metrics to show that the classifiers learned the link task, but lead with reconstruction:

- `b3_f1`: energy-aware B-cubed clustering F1.
- `containment_efficiency_40`: fraction of signal truth showers with at least 40% energy recovered.
- `association_iou_efficiency`: stricter truth-to-reco component association proxy.
- `fake_rate`, `duplicate_rate`, `merge_rate`: reconstruction failure modes.
- `mean_best_iou` and `energy_weighted_iou`: how close predicted supertracksters are to truth.

The `unlinked_baseline` row is the "before linking" reference. The focal and contrastive rows are
the "after reconstruction" comparison.
