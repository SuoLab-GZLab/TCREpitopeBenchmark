import torch


def sequencePadding(sequence, max_len, aa2idx, padding="-"):
    """
    Default padding style: right padding.
    Note that the sequencePadding use One-hot encode style.
    :param sequence: original amino acid sequence
    :param max_len: max length of the final sequence
    :param padding: padding character
    :return:
    """
    sequence = sequence[:max_len] if len(sequence) >= max_len else sequence + "-" * (max_len - len(sequence))
    padding = torch.zeros(max_len, 21)
    for i in range(min(max_len, len(sequence))):
        padding[i][aa2idx[sequence[i]]] = 1
    return padding


def onehot2Sequence(onehot, idx2aa):
    """
    Convert the batched onehot or proba tensor to batched amino acid sequences.
    :param onehot: tensor
    :param idx2aa: dict
    :return:
    """
    batch, max_len, onehot_dim = onehot.shape
    ret_list = []
    for i in range(batch):
        sequence = ""
        for j in range(max_len):
            sequence += idx2aa[int(torch.argmax(onehot[i][j]))]
        ret_list.append(sequence)
    return ret_list
