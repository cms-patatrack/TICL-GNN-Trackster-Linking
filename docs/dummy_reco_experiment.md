# Dummy Reconstruction Experiment

This workflow is meant for poster-safe experiments when CMS event data cannot be shown or used.
It creates public synthetic HGCAL-like events with known truth labels, trains two models on the same
split, and evaluates both edge classification and reconstructed supertracksters.

## Thesis Anchors

The dummy generator follows the baseline thesis setup:

- HGCAL endcap coverage: `1.5 <= |eta| <= 3.0`, full `phi`, 47 layers per endcap.
- Densities use the thesis convention `2 * (3 - 1.5) * (2 * 47)`.
- Non-PU hard showers use `10-600 GeV` energies; PU-like events use a `pT` gun proxy.
- Signal composition is pion-dominated for the PU-like case.
- The event graph uses the thesis eta-phi window of `0.2`.
- Model outputs are thresholded into connected components, interpreted as supertracksters.

## One-Command Experiment

```bash
python scripts/run_dummy_reco_experiment.py \
  --work-dir outputs/dummy_reco_experiment \
  --generate-data \
  --scenario mixed \
  --epochs 30
```

For a quick smoke run:

```bash
python scripts/run_dummy_reco_experiment.py \
  --work-dir outputs/dummy_reco_smoke \
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

The experiment writes:

- `metrics.json` and `metrics.csv`: poster/table-ready metrics.
- `loss_comparison.png`: focal vs focal+contrastive training curves.
- `reconstruction_metric_bars.png`: held-out reconstruction metrics.
- `focal/*.pt` and `focal_contrastive/*.pt`: checkpoints with chosen validation threshold.

## Poster Metrics

Use the edge metrics to show that the classifiers learned the link task, but lead with reconstruction:

- `b3_f1`: energy-aware B-cubed clustering F1.
- `containment_efficiency_40`: fraction of signal truth showers with at least 40% energy recovered.
- `association_iou_efficiency`: stricter truth-to-reco component association proxy.
- `fake_rate`, `duplicate_rate`, `merge_rate`: reconstruction failure modes.
- `mean_best_iou` and `energy_weighted_iou`: how close predicted supertracksters are to truth.

The `unlinked_baseline` row is the "before linking" reference. The focal and contrastive rows are
the "after reconstruction" comparison.
