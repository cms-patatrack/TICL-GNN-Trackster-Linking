import os
import os.path as osp
import sys
from glob import glob

import numpy as np
import torch

from tqdm import tqdm

from tracksterLinker.datasets.ClusterDataset import ClusterDataset
from tracksterLinker.datasets.lang import Lang

from tracksterLinker.utils.graphUtils import find_connected_components

from concurrent.futures import ProcessPoolExecutor, as_completed


def process_event(path, event_nr, input_length, component_dir, sequence_dir, component_dict_dir, output_group_dir, output_group, store_component_dict, raw_energy,
                  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    data_access = []
    max_nodes = 0

    sample = torch.load(path, weights_only=False)

    if (sample == None):
        return

    if isinstance(sample, str):
        return

    components = find_connected_components(sample.edge_index, sample.x.shape[0])

    for comp_cnt, component in enumerate(components):
        visited = []
        component = np.array(component, dtype=int)

        if (component.shape[0] <= 1):
            continue

        if (component.shape[0] > max_nodes):
            max_nodes = component.shape[0]

        cluster = torch.unique(sample.cluster[component])
        cluster = cluster[cluster >= 0]

        converter = Lang(trackster_list=component)

        data = {}
        data["x"] = sample.x[component].float().to(device)
        data["name"] = f"{event_nr}_{comp_cnt}"
        data["lang"] = converter.getTracksterList()
        data["nTrackster"] = component.shape[0]
        data["inputs"] = {}
        data["seqs"] = []
        data["roots"] = []

        # torch.save(sample.x[component].float().to(device), osp.join(component_dir, f'comp_{event_nr}_{comp_cnt}.pt'))

        while component.shape[0] > 0:
            # Root node with max "raw_energy"
            root = component[torch.argmax(sample.x[component, raw_energy]).item()].item()
            sample_seq, root_group = converter.y2seq(root, component, np.array(sample.cluster))

            data["seqs"].append(sample_seq)
            data["roots"].append(root)

            for i in range(sample_seq.shape[0]-3):
                vals = {}
                seq = torch.from_numpy(converter.subseq(sample_seq, seq_length=input_length+1, index=i-input_length+2)).long().to(device)
                vals["input"] = seq[:-1]
                vals["y"] = seq[1:]

                # torch.save({"input": seq[:-1], "output": seq[1:]}, osp.join(sequence_dir, f'comp_{event_nr}_{comp_cnt}_{i}.pt'))
                if (output_group and output_group_dir is not None):
                    last_word = seq[-2].item()
                    visited.append(converter.index2word[last_word])
                    group = np.setdiff1d(root_group, visited)
                    group = torch.tensor(list(map(converter.word2index.get, group)))

                    if (group.shape[0] == 0):
                        group = torch.unsqueeze(seq[-1], dim=0)

                    ys = torch.cat([torch.unsqueeze(seq[1:], dim=0)] * group.shape[0], dim=0).long()
                    ys[:, -1] = group
                    vals["options"] = ys.long().to(device)

                    torch.save(vals["options"], osp.join(output_group_dir, f'comp_{event_nr}_{comp_cnt}_{i}.pt'))

                data_access.append({"event": event_nr, "component": comp_cnt, "step": i})

                data["inputs"][i] = vals

            component = np.setdiff1d(component, root_group)

        torch.save(data, osp.join(component_dict_dir, f'comp_{event_nr}_{comp_cnt}.pt'))
    return data_access, max_nodes


class ClusterDatasetBuilder:

    def __init__(self, root, data_path, input_length, output_group=False, store_component_dict=False):

        self.path = data_path
        self.output_group = output_group
        self.store_component_dict = store_component_dict

        self.processed_dir = root
        self.component_dir = osp.join(self.processed_dir, "component")
        self.component_dict_dir = osp.join(self.processed_dir, "component_dict")
        self.sequence_dir = osp.join(self.processed_dir, "sequence")
        self.output_group_dir = osp.join(self.processed_dir, "output_group")

        self.input_length = input_length

    def metadata_exists(self):
        return osp.isfile(osp.join(self.processed_dir, "metadata.pt"))

    def generate(self, num_workers=64, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        print('Processing...', file=sys.stderr)

        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.component_dir, exist_ok=True)
        os.makedirs(self.sequence_dir, exist_ok=True)
        os.makedirs(self.component_dict_dir, exist_ok=True)

        if self.output_group:
            os.makedirs(self.output_group_dir, exist_ok=True)

        files = glob(f"{self.path}/data_*.pt")
        data_access = []
        max_nodes = 0

        with tqdm(total=len(files)) as pbar:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        process_event, file, event, self.input_length, self.component_dir, self.sequence_dir, self.component_dict_dir, self.output_group_dir, self.output_group,
                        self.store_component_dict, ClusterDataset.node_feature_dict["raw_energy"],
                        device) for event, file in enumerate(files)]

                for future in as_completed(futures):
                    metadata, node_nums = future.result()

                    if (metadata is not None):
                        data_access.extend(metadata)

                        if (node_nums > max_nodes):
                            max_nodes = node_nums
                    pbar.update()

        self.max_nodes = max_nodes
        self.count = len(files)
        self.data_access = data_access
        metadata = {"max_nodes": max_nodes, "count": self.count, "output_group": self.output_group, "data_access": data_access}
        torch.save(metadata, osp.join(self.processed_dir, f'metadata.pt'))

        print('Done!', file=sys.stderr)
