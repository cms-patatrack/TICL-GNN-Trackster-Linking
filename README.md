# TICL-GNN-Trackster-Linking

* Graph Neural Network for trackster linking, including scripts for training and dataset creation. 
* Transformer architecture for cluster creation based on token learning. Preliminary, without great training success. Attention architecture included in GNN, would not investigate further.
* Stability Analysis of graph models
* Dummy Data creation based on statistical analysis of trackster dataset

![Python version](https://img.shields.io/badge/python-3.9.21-blue.svg)
![License: MPL-2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)

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

## Run

### Installation

If on a patatrack machine or with a python version less than `3.9.21` use either a virtual environment or use the `cmsenv` of a CMS release. If no release is checked out yet, follow the description on how to set up the inference.

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

### Train

On the `patatrack-bg-01.cern.ch` machine the dataset for the GNN is provided. To train the GNN with contrastive learning, change the model specific data of the file `scripts/trainGNN_contrastive.py`.
All script paths are setup to be used with the prepared data from `data/czeh` out of the box. For training change the `model_folder` to a new name and number from the temporal `9999_CHANGE_TO_NEW_MODEL`.

```python
load_weights = True  # If pretrained model should be used for startup
model_name = "model_2025-11-19_epoch_5_dict" # Name of the pretrained

base_folder = "/data/czeh"
load_model_folder = osp.join(base_folder, "model_results/0002_model_large_contr_att") # path to pretrained model
model_folder = osp.join(base_folder, "training_data/9999_CHANGE_TO_NEW_MODEL") # path to store trained model data and validation plots
data_folder_training = osp.join(base_folder, "linking_dataset/dataset_hardronics") # path to train dataset
data_folder_test = osp.join(base_folder, "linking_dataset/dataset_hardronics_test") # path to test dataset
```

Then call:
```
python3 scripts/trainGNN_contrastive.py
```

For the focal only training with the script `trainGNN.py` the procedure is the same. However no pretrained model exists, and the model architecture is without multi head attention, `GNN_TrackLinkingNet`. `PUNet` in directory `multiGNN` uses the multi head attention.

```python
model = GNN_TrackLinkingNet(input_dim=len(dataset_training.model_feature_keys),
                            edge_feature_dim=dataset_training[0].edge_features.shape[1], niters=4,
                            edge_hidden_dim=32, hidden_dim=64, weighted_aggr=True, dropout=0.3,
                            node_scaler=dataset_training.node_scaler, edge_scaler=dataset_training.edge_scaler)
```

### Inference

Run the following commands to set up a CMSSW release to run the GNN inference on.

```
cmsrel CMSSW_16_0_0_pre1
cd CMSSW_16_0_0_pre1/src/
cmsenv
git cms-init
git remote add czeh-cmssw git@github.com:chrisizeh/cmssw.git
git fetch czeh-cmssw
git checkout gnn-inference
git diff --name-only $CMSSW_VERSION | cut -d/ -f-2 | sort -u | xargs -r git cms-addpkg
scram b -j 64
mkdir RecoHGCal/TICL/models
cp /data/czeh/model_results/0002_model_large_contr_att/model_2025-10-29_traced.pt RecoHGCal/TICL/model/0002_model_large_contr_att.pt
```

Run this commands from the `src` folder of the CMSSW release to validate the model on 200 PU, single pion.
```
cp -r /data/czeh/gnn-validation/29896.203_CloseByPGun_CE_E_Front_120um+Run4D110PU_ticl_v5/ .
cd 29896.203_CloseByPGun_CE_E_Front_120um+Run4D110PU_ticl_v5/
cmsRun -n 4 step3_RAW2DIGI_RECO_RECOSIM_PAT_VALIDATION_DQM_PU.py
cmsRun step4_HARVESTING.py
mv DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root 0002_model_large_contr_att_05_08_cut_assoc.root
makeHGCalValidationPlots.py TICL_v5.root 0002_model_large_contr_att_05_08_cut_assoc.root --collection tracksters --png --ticlv 5
```

The resulting plots can be moved to the personal website (\[username\].web.cern.ch) to check them.

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
