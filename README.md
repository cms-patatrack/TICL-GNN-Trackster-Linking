# TICL-GNN-Trackster-Linking

* Graph Neural Network for trackster linking, including scripts for training and dataset creation. 
* Transformer architecture for cluster creation based on token learning. Preliminary, without great training success. Attention architecture included in GNN, would not investigate further.
* Stability Analysis of graph models
* Dummy Data creation based on statistical analysis of trackster dataset

![Python version](https://img.shields.io/badge/python-3.9.21-blue.svg)
![License: MPL-2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data & Inputs](#data--inputs)
- [How to Use](#how-to-use)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Notebooks](#notebooks)
- [Results](#results)
- [Configuration](#configuration)
- [Development Notes](#development-notes)
- [Citing](#citing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Repository Structure

```text
TICL-GNN-Trackster-Linking/
├── tracksterLinker/        # Core Python package: data, architecture, training loop, utilities
├── scripts/                # Command-line scripts for training, evaluation, dataset creation, plotting, etc.
├── notebooks/              # Jupyter notebooks for exploration, debugging, and analysis
├── requirements.txt        # Python dependencies
├── LICENSE                 # MPL-2.0 license
└── README.md               # This file
````

### tracksterLinker

```text
tracksterLinker/
├── GNN/                    # GNN architecture, loss functions, epoch loops
├── datasets/               # Datasets for all available architectures 
├── multiGNN/               # GNN architecture for GNN with multi head attention 
├── transformer/            # Transformer architecture, loss function, epoch loops and full event run
├── utils/                  # Utility functions for plotting, graph building and perturbations
````
---

### scripts

```text
scripts/
├── dummyDataGeneration/        # Data analysis and dummy data generation
    ├── dataStatistics.py       # Analyse defined dataset and create gaussian mixture clusters based on trackster positions
    ├── dummyGenerations.py     # Generate dummy data based on stored statistics
├── globalMetrics/              # Calculate per event metrics of energy merging 
    ├── metricsCreation.py      # Store metrics for signal and pu separately
    ├── metricsPlotting.py      # Plot GraphHeatmap based on stored metrics
├── stabilityAnalysis/          # Per edge stability checks
    ├── edgeStability.py        # Plot heatmap of edge flips under PCA perturbation, compare two models
    ├── plotPerturbations.py    # Plot heatmap of single node PCA perturbation distribution
    ├── stabilityAnalysis.py    # Store metrics of graph and PCS perturbated graphs
    ├── stabilityPlotting.py    # Plot GraphHeatmap of metrics difference between base and perturbated graphs
├── compareGNNDatasets.py       # Compare GNNDataset and NeoGNNDataset build with same hiso.root to check for correct postprocessing
├── createClusterDataset.py     # Create Transformer dataset based on GNNDataset
├── createGNNDataset.py         # Create GNN dataset based on histo.root. Run before training, if no dataset!
├── testSplitPU.py              # Analyze changes in graph structure
├── trainGNN.py                 # Train GNN with focal loss
├── trainGNN_contrastive.py     # Train GNN with multi objective loss function, focal+contrastive
````

---
## Installation

Clone the repository:

```bash
git clone https://github.com/cms-patatrack/TICL-GNN-Trackster-Linking.git
cd TICL-GNN-Trackster-Linking
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Development Notes

Things unsucessfully tried:
* Heavy high energy weighting by squaring calculated weights -> small trackster get so close to 0 weights, that their precision drops to `60%` and almost no trackster get merged, independent of the cut.
* Transformer approach would need more work to set up full pipeline of learning cluster sequence, like done with LLMs.


Things, that could be interesting to try:
* Use [PQuant](https://github.com/nroope/PQuant/blob/main/docs/pruning_methods.md) to prune the model. Find way to optimize pruning value based on validation efficiency, as accuracy alone could not be meaningful enough.
* Add Global Attention for all nodes and use as input for EdgeConv, rather than per node feature attention. Has to be checked how large global attention is meaningful. Old transformer said up to 500 is meaninful, but could have changed with newer architectures.
* Is there a loss function that dynamically adapts to difficult training data and focuses on this. -> Heavy tracksters have a way worse result, splitting of true and false edges is not definite.

---

## References

* EdgeConv Implementation based on [Dynamic Graph CNN for Learning on Point Clouds](https://doi.org/10.48550/arXiv.1801.07829)
* Contrastive learning: [Contrastive Representation Learning: A Framework and Review](https://doi.org/10.1109/ACCESS.2020.3031549)

## License

This project is licensed under the **Mozilla Public License 2.0 (MPL-2.0)**.
See the [`LICENSE`](./LICENSE) file for details.

---
