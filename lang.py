import numpy as np

# Tokens for model to know if sequence is start or and of text
SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, num_nodes):
        self.word2index = {";": 3, "<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: ";"}
        self.n_words = 4  # Count SOS, EOS and PAD

        for i in range(num_nodes):
            self.index2word[self.n_words] = str(i)
            self.word2index[str(i)] = self.n_words
            self.n_words += 1

    # transform a string into an encoded numpy array.
    # If max_len provided, string will be truncated
    # If min_len provided, padding is added to array
    def seq2arr(self, seq, index=-1, max_len=-1, min_len=0):
        seq = seq.split(".")
        if max_len <= 0:
            max_len = len(seq)

        arr = np.zeros(max_len, dtype=np.int64)
        j = 0
        if (index > 0):
            seq = seq[index]

        for val in enumerate(seq[:max_len]):
            arr[j] = self.word2index[val]

        if (min_len > len(arr[:j])):
            arr = np.pad(arr[:j], (min_len - len(arr[:j]), 0))

        if (index == -1 and arr[0] == 0):
            print(min_len - len(arr[:j]))
            arr[min_len - len(arr[:j]) - 1] = 1

        if (len(seq) == 0 or (arr[-1] == self.word2index[seq[-1]] and arr[-2] == self.word2index[seq[-2]])):
            if (arr[0] == 0):
                arr = np.pad(arr[1:], (0, 1), constant_values=2)
        return arr

    def subseq2arr(self, seq, arr_length, index, length):
        arr = np.zeros(arr_length, dtype=np.int64)
        stop = 0

        if (index+length >= len(seq)):
            arr[-1] = 2
            stop = 1
            length = len(seq) - index

        if (length + stop > arr_length):
            length = arr_length - stop

        if (index < 0):
            return arr
        elif (index == 0):
            arr[arr_length-length-stop-1] = 1

        if length > 0:
            arr[arr_length-length-stop:arr_length-stop] = self.seq2arr(seq[index:index+length])

        return arr

    def arr2seq(self, arr, ignoreTokens=False):
        if (ignoreTokens):
            return "".join([self.index2word[idx.item()] for idx in arr if idx > 2])
        return "".join([self.index2word[idx.item()] for idx in arr])

    def y2seq(self, arr, index=0, seq_length=-1):
        if (arr.shape[0] > 0):
            numGroups = np.max(arr)+1
        else:
            numGroups == 0
        res = np.zeros(numGroups+arr.shape[0]+1)

        j = 0
        for group in range(numGroups):
            for (i, trackster) in enumerate(arr):
                if (trackster == group):
                    res[j] = self.word2index[str(i)]
                    j += 1

            res[j] = self.word2index[";"]
            j += 1
        res[j] = self.word2index["<EOS>"]

        if (index == 0):
            res = np.pad(res, (1, 0), constant_values=self.word2index["<SOS>"])

        if res.shape[0] < seq_length:
            res = np.pad(res, (seq_length - res.shape[0], 0), constant_values=self.word2index["<PAD>"])

        if (seq_length >= 1):
            return res[index:seq_length+index]

        return res[index:]

    def seq2y(self, arr):
        numTrackster = np.max(arr)-3
        y = np.full(numTrackster, -1)

        if (arr[0] != 1 and arr[-1] != 1):
            print("not full sequence!")
        else:
            group = 0
            for i in arr[1:-1]:
                if (i == 3):
                    group += 1
                    continue
                y[i-4] = group
        return y


if __name__ == "__main__":
    lang = Lang(14)

    print(lang.y2seq(np.array([0, 0, 0, 1, 1, 1, -1, -1, 1, 0])))
    print(lang.y2seq(np.array([0, 0, 0, 1, 1, 1, -1, -1, 1, 0]), index=3))
    print(lang.y2seq(np.array([0, 0, 0, 1, 1, 1, -1, -1, 1, 0]), index=3, seq_length=5))
    print(lang.y2seq(np.array([0, 1]), seq_length=5))
    print(lang.y2seq(np.array([0, 1]), index=1, seq_length=5))
    print(lang.seq2y(np.array([1, 4, 5, 8, 3, 6, 2])))
