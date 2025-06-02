import torch
import numpy as np
import itertools
import awkward as ak


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
        if (arr.shape[0] > 0):
            numGroups = int(np.max(arr)+1)
        else:
            numGroups == 0
        res = np.full(numGroups+arr.shape[0]+4, self.word2index["<PAD>"])
        res[0] = self.word2index["<SOS>"]
        j = 1
        groups = 0
        root_group = arr[root]

        inGroup = 0
        for i in range(trackster.shape[0]):
            if (arr[trackster[i]] == root_group):
                res[j] = self.word2index[trackster[i]]
                inGroup += 1
                groups += 1
                j += 1

        if (inGroup > 1):
            res[j] = self.word2index[";"]
            j += 1

        for group in range(numGroups):
            if (group == root_group):
                continue
            inGroup = 0
            for i in range(trackster.shape[0]):
                if (arr[trackster[i]] == group):
                    res[j] = self.word2index[trackster[i]]
                    inGroup += 1
                    groups += 1
                    j += 1

            if (inGroup > 1):
                res[j] = self.word2index[";"]
                j += 1
        res[j] = self.word2index["<EOS>"]
        return res[:j+2]

    def seq2y(self, arr, nodes=None, start_group=0):
        if nodes:
            numTrackster = nodes
        else:
            numTrackster = self.max_trackster + 1
        y = np.full(numTrackster, -1)

        group = start_group
        for i in arr:
            if (i == self.word2index[";"]):
                group += 1
            elif (i > self.word2index[";"]):
                y[self.index2word[i]] = group
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

    def permute_groups(self, arr):
        blocks = ak.Array(np.split(arr[1:-1], np.nonzero(arr == self.word2index[";"])[0])[:-1])
        numGroups = len(blocks)
        opts = np.array(list(itertools.permutations(list(range(numGroups)))))

        permutations = ak.flatten(blocks[opts], -1).to_numpy()
        permutations = np.pad(permutations, ((0, 0), (1, 0)), constant_values=self.word2index["<SOS>"])
        permutations = np.pad(permutations, ((0, 0), (0, 1)), constant_values=self.word2index["<EOS>"])
        return permutations

    def starting_seq(self, root, seq_length):
        seq = torch.full((seq_length, ), self.word2index["<PAD>"])
        seq[0] = self.word2index["<SOS>"]
        seq[1] = self.word2index[root]
        return seq.long()


if __name__ == "__main__":
    lang = Lang(14)

    # print(lang.y2seq(np.array([0, 0, 0, 1, 1, 1, -1, -1, 1, 0])))
    # print(lang.y2seq(np.array([0, 0, 0, 1, 1, 1, -1, -1, 1, 0]), index=3))
    # print(lang.y2seq(np.array([0, 0, 0, 1, 1, 1, -1, -1, 1, 0]), index=3, seq_length=5))
    # print(lang.y2seq(np.array([0, 1]), seq_length=5))
    # print(lang.y2seq(np.array([0, 1]), index=1, seq_length=5))
    # print(lang.seq2y(np.array([1, 4, 5, 8, 3, 6, 2])))

    arr = lang.y2seq(np.array([0, 0, 0, 1, 1, 1, -1, -1, 0]))
    print(lang.permute_groups(arr))
