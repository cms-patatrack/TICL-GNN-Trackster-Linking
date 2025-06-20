import torch
import numpy as np

# For a gapless initialization use num_nodes, otherwise provide trackster list
# If Trackster list is provided, num_nodes will be ignored


class Lang:
    def __init__(self, num_nodes=0, trackster_list=None):
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, ";": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: ";"}
        self.n_words = 4  # Count SOS, EOS and PAD
        self.next_index = 4

        if (trackster_list is not None):
            self.max_trackster = np.max(trackster_list)
            for trackster in trackster_list:
                self.index2word[self.next_index] = trackster
                self.word2index[trackster] = self.next_index
                self.n_words += 1
                self.next_index += 1
        elif (num_nodes > 0):
            for i in range(num_nodes):
                self.index2word[self.next_index] = i
                self.word2index[i] = self.next_index
                self.n_words += 1
                self.next_index += 1

            self.max_trackster = self.n_words - 4

    def getTracksterList(self):
        keys = list(self.word2index.keys())
        keys = keys[4:]
        return np.array(keys, dtype=int)

    def y2seq(self, root, trackster, arr):
        root_group = arr[root]
        group = np.where(arr == root_group)[0]
        group = np.intersect1d(group, trackster)

        res = np.full(group.shape[0]+2, self.word2index["<PAD>"])
        res[0] = self.word2index["<SOS>"]
        res[1] = self.word2index[root]
        res[-1] = self.word2index["<EOS>"]

        res[2:-1] = np.array(list(map(self.word2index.get, group[group != root])))
        return res, group

    def seq2y(self, arrs, nodes=None, start_group=0):
        if nodes is not None:
            numTrackster = nodes
        else:
            numTrackster = self.max_trackster + 1
        y = np.full(numTrackster, -1)

        for group, arr in enumerate(arrs):
            trackster = list(map(self.index2word.get, arr[1:-1]))
            y[trackster] = start_group + group

        return y

    # Remove padding:  add in dataset preparation
    def subseq(self, seq, index=0, seq_length=-1):
        if (index <= 0):
            seq = np.pad(seq, (np.abs(index), 0), constant_values=self.word2index["<PAD>"])
            index = 0

        if seq.shape[0] < seq_length:
            seq = np.pad(seq, (seq_length - seq.shape[0], 0), constant_values=self.word2index["<PAD>"])

        if (seq_length >= 1):
            return seq[index:seq_length+index]

        return np.trim_zeros(seq[index:], trim='b')

    def starting_seq(self, root, seq_length):
        seq = torch.full((seq_length, ), self.word2index["<PAD>"])
        seq[0] = self.word2index["<SOS>"]
        seq[1] = self.word2index[root]
        return seq.long()


if __name__ == "__main__":
    lang = Lang(14)

    tracksters = np.array([3, 4, 1, 2, 5])
    y = np.array([-1, 0, 0, 1, 1, 1, 0, 1])

    arr1, tracksters1 = lang.y2seq(3, tracksters, y)
    print("seq 1", arr1, tracksters1)

    arr2, tracksters2 = lang.y2seq(1, tracksters, y)
    print("seq 2", arr2, tracksters2)

    print(lang.seq2y([arr1, arr2], nodes=8))
