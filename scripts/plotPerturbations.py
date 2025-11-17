import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp

from torch_geometric.loader.dataloader import DataLoader

from tracksterLinker.datasets.NeoGNNDataset import NeoGNNDataset
from tracksterLinker.utils.dataStatistics import *
from tracksterLinker.utils.dataUtils import *
from tracksterLinker.utils.perturbations.inErrorBars import *

# User-defined phase space region
MAX_ETA = 2.7
MIN_ETA = 1.7
MAX_PHI = np.pi
MIN_PHI = -np.pi

# Number of random draws
N_ITER = 2000
WITH_Z = True

base_folder = "/home/czeh"
output_folder = osp.join(base_folder, "stability/perturbations")
hist_folder = osp.join(base_folder, "_")
data_folder_test = osp.join(base_folder, "GNN/dataset_hardronics_test")
os.makedirs(output_folder, exist_ok=True)

# Prepare Dataset
batch_size = 1
dataset_test = NeoGNNDataset(data_folder_test, hist_folder, test=True, only_signal=False)
test_dl = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)

sample = next(iter(test_dl))
node_features = sample.x
fdict = NeoGNNDataset.node_feature_dict

# Select nodes in eta/phi window
eta_idx = fdict["barycenter_eta"]
phi_idx = fdict["barycenter_phi"]
mask = (
    (node_features[:, eta_idx] >= MIN_ETA) &
    (node_features[:, eta_idx] <= MAX_ETA) &
    (node_features[:, phi_idx] >= MIN_PHI) &
    (node_features[:, phi_idx] <= MAX_PHI)
)

nodes_in_region = torch.nonzero(mask).squeeze()

if len(nodes_in_region.shape) == 0:
    nodes_in_region = nodes_in_region.unsqueeze(0)

if len(nodes_in_region) == 0:
    raise RuntimeError("No nodes found in requested η–φ region!")

# Pick one node (first match)
i = nodes_in_region[0].item()
print(f"Selected node index {i} with η={node_features[i, eta_idx]:.2f}, φ={node_features[i, phi_idx]:.2f}")

# Get its original barycenter
bx, by, bz = (
    fdict["barycenter_x"],
    fdict["barycenter_y"],
    fdict["barycenter_z"],
)
base = node_features[i, bx:bz+1].detach().cpu().numpy()

# Collect perturbations
deltas = []

for _ in range(N_ITER):
    out = perturbate(node_features, num_samples=1, with_z=WITH_Z)
    pert_xyz = out[0, i, bx:bz+1].detach().cpu().numpy()
    deltas.append(pert_xyz - base)

deltas = np.array(deltas)
dx, dy, dz = deltas[:, 0], deltas[:, 1], deltas[:, 2]

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dx, dy, dz, s=10, alpha=0.5)
ax.set_xlabel("Δx [cm]")
ax.set_ylabel("Δy [cm]")
ax.set_zlabel("Δz [cm]")
ax.set_title(f"Perturbations for one node (η≈{node_features[i, eta_idx]:.2f}, φ≈{node_features[i, phi_idx]:.2f})")

# For 2D projections
plt.figure(figsize=(15, 4))
plt.subplot(1,3,1)
plt.hist2d(dx, dy, bins=40, cmap='plasma')
plt.xlabel("Δx")
plt.ylabel("Δy")
plt.title("XY Projection")

plt.subplot(1,3,2)
plt.hist2d(dx, dz, bins=40, cmap='plasma')
plt.xlabel("Δx")
plt.ylabel("Δz")
plt.title("XZ Projection")

plt.subplot(1,3,3)
plt.hist2d(dy, dz, bins=40, cmap='plasma')
plt.xlabel("Δy")
plt.ylabel("Δz")
plt.title("YZ Projection")

plt.tight_layout()
plt.savefig(osp.join(output_folder, f"pert_{i}.png"))
