import numpy as np
import torch

vocab = ["-", "A", "a", "B", "b", "C", "c", "D", "d", "E", "e", "F", "f", "G", "g",
      "H", "h", "I", "i", "J", "j", "K", "k", "L", "l", "M", "m", "N", "n", "O", "o",
      "P", "p", "Q", "q", "R", "r", "S", "s", "T", "t", "U", "u", "V" , "v", "W", "w",
      "X", "x", "Y", "y", "Z", "z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

char2index = {char: vocab.index(char) for char in vocab}
index2char = {vocab.index(char): char for char in vocab}
blank_index = 0

print(len(vocab))


def decode_sequence(labels):
    seqs = []
    for label in labels:
        seqs.append(''.join(index2char[char] for char in label))
    return seqs


def decode_path(probs):
    blank_index = 0
    batch_seq = []
    for prob in probs:
        best_path = np.argmax(prob, axis=0).tolist()
        seq = []
        for i, pred_char in enumerate(best_path):
            if pred_char == blank_index:
                continue
            elif i != 0 and pred_char == best_path[i - 1]:
                continue
            else:
                seq.append(pred_char)
        batch_seq.append(seq)
    return batch_seq
