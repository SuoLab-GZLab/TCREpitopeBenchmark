import torch
import torch.nn as nn
import torch.optim as optim
from random import shuffle
import time
import numpy as np
import torch.autograd as autograd
from ERGO_models import DoubleLSTMClassifier
from sklearn.metrics import roc_auc_score, roc_curve


def get_lists_from_pairs(pairs):
    tcrs = []
    peps = []
    signs = []
    for pair in pairs:
        tcr, pep, label, weight = pair
        tcrs.append(tcr)
        peps.append(pep)
        if label == 'p':
            signs.append(1.0)
        elif label == 'n':
            signs.append(0.0)
    return tcrs, peps, signs


def convert_data(tcrs, peps, amino_to_ix):
    for i in range(len(tcrs)):
        if any(letter.islower() for letter in tcrs[i]):
            print(tcrs[i])
        tcrs[i] = [amino_to_ix[amino] for amino in tcrs[i]]
    for i in range(len(peps)):
        peps[i] = [amino_to_ix[amino] for amino in peps[i]]


def get_batches(tcrs, peps, signs, batch_size):
    """
    Get batches from the data
    """
    # Initialization
    batches = []
    index = 0
    # Go over all data
    while index < len(tcrs):
        # Get batch sequences and math tags
        batch_tcrs = tcrs[index:index + batch_size]
        batch_peps = peps[index:index + batch_size]
        batch_signs = signs[index:index + batch_size]
        # Update index
        index += batch_size
        # Pad the batch sequences
        padded_tcrs, tcr_lens = pad_batch(batch_tcrs)
        padded_peps, pep_lens = pad_batch(batch_peps)
        # Add batch to list
        batches.append((padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs))
    # Return list of all batches
    return batches


def get_full_batches(tcrs, peps, signs, batch_size, amino_to_ix):
    """
    Get batches from the data, including last with padding
    """
    # Initialization
    batches = []
    index = 0
    # Go over all data
    while index < len(tcrs) // batch_size * batch_size:
        # Get batch sequences and math tags
        batch_tcrs = tcrs[index:index + batch_size]
        batch_peps = peps[index:index + batch_size]
        batch_signs = signs[index:index + batch_size]
        # Update index
        index += batch_size
        # Pad the batch sequences
        padded_tcrs, tcr_lens = pad_batch(batch_tcrs)
        padded_peps, pep_lens = pad_batch(batch_peps)
        # Add batch to list
        batches.append((padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs))
    # pad data in last batch
    missing = batch_size - len(tcrs) + index
    if missing < batch_size:
        padding_tcrs = ['A'] * missing
        padding_peps = ['A' * (batch_size - missing)] * missing
        convert_data(padding_tcrs, padding_peps, amino_to_ix)
        batch_tcrs = tcrs[index:] + padding_tcrs
        batch_peps = peps[index:] + padding_peps
        padded_tcrs, tcr_lens = pad_batch(batch_tcrs)
        padded_peps, pep_lens = pad_batch(batch_peps)
        batch_signs = [0.0] * batch_size
        # Add batch to list
        batches.append((padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs))
        # Update index
        index += batch_size
    # Return list of all batches
    return batches
    pass


def pad_batch(seqs):
    """
    Pad a batch of sequences (part of the way to use RNN batching in PyTorch)
    """
    # Tensor of sequences lengths
    lengths = torch.LongTensor([len(seq) for seq in seqs])
    # The padding index is 0
    # Batch dimensions is number of sequences * maximum sequence length
    longest_seq = max(lengths)
    batch_size = len(seqs)
    # Pad the sequences. Start with zeros and then fill the true sequence
    padded_seqs = autograd.Variable(torch.zeros((batch_size, longest_seq))).long()
    for i, seq_len in enumerate(lengths):
        seq = seqs[i]
        padded_seqs[i, 0:seq_len] = torch.LongTensor(seq[:seq_len])
    # Return padded batch and the true lengths
    return padded_seqs, lengths


def train_epoch(batches, model, loss_function, optimizer, device):
    model.train()
    shuffle(batches)
    total_loss = 0
    for batch in batches:
        padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        batch_signs = torch.tensor(batch_signs).to(device)
        model.zero_grad()
        probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
        # print(probs, batch_signs)
        # Compute loss
        weights = batch_signs * 0.84 + (1-batch_signs) * 0.14
        loss_function.weight = weights
        ##---------
        batch_signs = batch_signs.unsqueeze(1)
        loss = loss_function(probs, batch_signs)
        # with open(sys.argv[1], 'a+') as loss_file:
        #    loss_file.write(str(loss.item()) + '\n')
        # Update model weights
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # print('current loss:', loss.item())
        # print(probs, batch_signs)
    # Return average loss
    return total_loss / len(batches)


def train_model(batches, test_batches, device, args, params):
    """
    Train and evaluate the model
    """
    losses = []
    # We use Cross-Entropy loss
    loss_function = nn.BCELoss()
    # Set model with relevant parameters
    if args['siamese'] is True:
        model = SiameseLSTMClassifier(params['emb_dim'], params['lstm_dim'], device)
    else:
        model = DoubleLSTMClassifier(params['emb_dim'], params['lstm_dim'], params['dropout'], device)
    # Move to GPU
    model.to(device)
    # We use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    # Train several epochs
    best_auc = 0
    best_roc = None
    for epoch in range(params['epochs']):
        print('epoch:', epoch + 1)
        epoch_time = time.time()
        # Train model and get loss
        loss = train_epoch(batches, model, loss_function, optimizer, device)
        losses.append(loss)
        # Compute auc
        train_auc = evaluate(model, batches, device)[0]
        print('train auc:', train_auc)
        with open(args['train_auc_file'], 'a+') as file:
            file.write(str(train_auc) + '\n')
        if params['option'] == 2:
            test_w, test_c = test_batches
            test_auc_w = evaluate(model, test_w, device)
            print('test auc w:', test_auc_w)
            with open(args['test_auc_file_w'], 'a+') as file:
                file.write(str(test_auc_w) + '\n')
            test_auc_c = evaluate(model, test_c, device)
            print('test auc c:', test_auc_c)
            with open(args['test_auc_file_c'], 'a+') as file:
                file.write(str(test_auc_c) + '\n')
        else:
            test_auc, roc = evaluate(model, test_batches, device)

            # nni.report_intermediate_result(test_auc)

            if test_auc > best_auc:
                best_auc = test_auc
                best_roc = roc
            print('test auc:', test_auc)
            with open(args['test_auc_file'], 'a+') as file:
                file.write(str(test_auc) + '\n')
        print('one epoch time:', time.time() - epoch_time)
    return model, best_auc, best_roc


def evaluate(model, batches, device):
    model.eval()
    true = []
    scores = []
    shuffle(batches)
    for batch in batches:
        padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
        # print(np.array(batch_signs).astype(int))
        # print(probs.cpu().data.numpy())
        true.extend(np.array(batch_signs).astype(int))
        scores.extend(probs.cpu().data.numpy())
    # Return auc score
    auc = roc_auc_score(true, scores)
    fpr, tpr, thresholds = roc_curve(true, scores)
    return auc, (fpr, tpr, thresholds)


def evaluate_full(model, batches, device):
    model.eval()
    true = []
    scores = []
    index = 0
    for batch in batches:
        padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
        true.extend(np.array(batch_signs).astype(int))
        scores.extend(probs.cpu().data.numpy())
        batch_size = len(tcr_lens)
        assert batch_size == 50
        index += batch_size
    border = pep_lens[-1]
    if any(k != border for k in pep_lens[border:]):
        print(pep_lens)
    else:
        index -= batch_size - border
        true = true[:index]
        scores = scores[:index]
    if int(sum(true)) == len(true) or int(sum(true)) == 0:
        # print(true)
        raise ValueError
    # Return auc score
    auc = roc_auc_score(true, scores)
    fpr, tpr, thresholds = roc_curve(true, scores)
    return auc, (fpr, tpr, thresholds)


def predict(model, batches, device):
    model.eval()
    preds = []
    index = 0
    for batch in batches:
        padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
        preds.extend([t[0] for t in probs.cpu().data.tolist()])
        batch_size = len(tcr_lens)
        assert batch_size == 50
        index += batch_size
    border = pep_lens[-1]
    if any(k != border for k in pep_lens[border:]):
        print(pep_lens)
    else:
        index -= batch_size - border
        preds = preds[:index]
    return preds
