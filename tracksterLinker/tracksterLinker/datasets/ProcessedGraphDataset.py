import os.path as osp
from glob import glob

import torch
from torch_geometric.data import Dataset

from tracksterLinker.utils.dataUtils import processing_device
from tracksterLinker.utils.dataUtils import sorted_basenames


class ProcessedGraphDataset(Dataset):
    node_feature_keys = [
        "barycenter_x",
        "barycenter_y",
        "barycenter_z",
        "barycenter_eta",
        "barycenter_phi",
        "eVector0_x",
        "eVector0_y",
        "eVector0_z",
        "num_hits",
        "raw_energy",
        "z_min",
        "z_max",
        "time",
    ]
    node_feature_dict = {k: v for v, k in enumerate(node_feature_keys)}
    model_feature_keys = node_feature_keys

    def __init__(
        self,
        root,
        histo_path=None,
        transform=None,
        test=False,
        pre_transform=None,
        pre_filter=None,
        edge_scaler=None,
        node_scaler=None,
        split=None,
        device=None,
        **_,
    ):
        self.root_dir = root
        self.split = split or ("test" if test else "train")
        self.test = test if split is None else self.split != "train"
        self.device = processing_device(device or "cpu")
        self.node_scaler = self._to_device(node_scaler) if node_scaler is not None else self._load_scaler("node_scaler.pt")
        self.edge_scaler = self._to_device(edge_scaler) if edge_scaler is not None else self._load_scaler("edge_scaler.pt")
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["DONE"]

    @property
    def processed_file_names(self):
        return ["DONE"]

    @property
    def processed_data_paths(self):
        return [osp.join(self.processed_dir, name) for name in sorted_basenames(f"{self.processed_dir}/data_*.pt")]

    def download(self):
        pass

    def process(self):
        if not glob(f"{self.processed_dir}/data_*.pt"):
            raise FileNotFoundError(
                f"No processed graph files found in {self.processed_dir}. "
                "Create them with scripts/create_colliderml_gnn_dataset.py first."
            )

    def len(self):
        return len(self.processed_data_paths)

    def get(self, idx):
        return torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"), weights_only=False, map_location=self.device)

    def _load_scaler(self, name):
        path = osp.join(self.root_dir, name)
        if osp.isfile(path):
            return torch.load(path, weights_only=False, map_location=self.device)
        return None

    def _to_device(self, value):
        return value.to(self.device) if hasattr(value, "to") else value
