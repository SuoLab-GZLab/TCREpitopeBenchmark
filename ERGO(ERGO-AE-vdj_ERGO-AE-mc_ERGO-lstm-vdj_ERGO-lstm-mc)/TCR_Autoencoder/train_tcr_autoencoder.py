import torch
import torch.optim as optim
import sys
from random import shuffle
from tcr_autoencoder import PaddingAutoencoder
from sklearn.model_selection import train_test_split
import os
import csv


def load_all_data(path):
    all_data = []
    for directory, subdirectories, files in os.walk(path):
        for file in files:
            with open(os.path.join(directory, file), mode='r') as infile:
                reader = csv.reader(infile)
                data = [row[1] for row in reader]
                all_data += [str(i) + 'X' for i in data[1:] if str(i).find('*') == -1 and str(i).find('X') == -1]
    # a one file full path
    if len(all_data) == 0:
        with open(path, mode='r') as infile:
            reader = csv.reader(infile)
            data = [row[1] for row in reader]
            all_data = data[1:]
    return all_data


def find_max_len(tcrs):
    return max([len(tcr) for tcr in tcrs])


def pad_one_hot(tcr, amino_to_ix, max_length):
    padding = torch.zeros(max_length, 20 + 1)
    for i in range(len(tcr)):
        amino = tcr[i]
        padding[i][amino_to_ix[amino]] = 1
    return padding


def get_batches(tcrs, amino_to_ix, batch_size, max_length):
    # Initialization
    batches = []
    index = 0
    # Go over all data
    while index < len(tcrs) // batch_size * batch_size:
        # Get batch sequences and math tags
        batch_tcrs = tcrs[index:index + batch_size]
        # Update index
        index += batch_size
        # Pad the batch sequences
        padded_tcrs = torch.zeros((batch_size, max_length, 20 + 1))
        for i in range(batch_size):
            padded_tcrs[i] = pad_one_hot(batch_tcrs[i], amino_to_ix, max_length)
        # Add batch to list
        batches.append(padded_tcrs)
    # Return list of all batches
    return batches


def train_epoch(batches, batch_size, model, loss_function, optimizer, device):
    model.train()
    shuffle(batches)
    total_loss = 0
    for batch in batches:
        padded_tcrs = batch
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        model.zero_grad()
        pred = model(batch_size, padded_tcrs)
        # Compute loss
        loss = loss_function(pred, padded_tcrs)
        # Update model weights
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # Return average loss
    return total_loss / len(batches)


def train_model(batches, batch_size, max_len, encoding_dim, epochs, device):
    model = PaddingAutoencoder(max_len, 20 + 1, encoding_dim)
    model.to(device)
    loss_function = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    for epoch in range(epochs):
        print('epoch:', epoch + 1)
        train_epoch(batches, batch_size, model, loss_function, optimizer, device)
    return model


def read_pred(pred, ix_to_amino):
    batch_tcrs = []
    for tcr in pred:
        c_index = torch.argmax(tcr, dim=1)
        t = ''
        for index in c_index:
            if ix_to_amino[index.item()] == 'X':
                break
            t += ix_to_amino[index.item()]
        batch_tcrs.append(t)
    return batch_tcrs


def count_mistakes(true_tcr, pred_tcr):
    mis = 0
    for i in range(min(len(true_tcr), len(pred_tcr))):
        if not true_tcr[i] == pred_tcr[i]:
            mis += 1
    return mis


def evaluate(batches, batch_size, model, ix_to_amino, device):
    model.eval()
    shuffle(batches)
    acc = 0
    acc_1mis = 0
    acc_2mis = 0
    acc_3mis = 0
    count = 0
    for batch in batches:
        padded_tcrs = batch
        true_tcrs = read_pred(padded_tcrs, ix_to_amino)
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        # model.zero_grad()
        pred = model(batch_size, padded_tcrs)
        pred_tcrs = read_pred(pred, ix_to_amino)
        # print('true:', true_tcrs)
        # print('pred:', pred_tcrs)
        for i in range(batch_size):
            count += 1
            mis = count_mistakes(true_tcrs[i], pred_tcrs[i])
            if mis == 0:
                acc += 1
            if mis <= 1:
                acc_1mis += 1
            # if mis <= 2:
                acc_2mis += 1
            if mis <= 3:
                acc_3mis += 1
    acc /= count
    acc_1mis /= count
    acc_2mis /= count
    acc_3mis /= count

    # with open(sys.argv[3], 'a+') as file:
    #     file.write('acc 0 mistakes: ' + str(acc) + '\n')
    #     file.write('acc up to 1 mistakes: ' + str(acc_1mis) + '\n')
    #     file.write('acc up to 2 mistakes: ' + str(acc_2mis) + '\n')
    #     file.write('acc up to 3 mistakes: ' + str(acc_3mis) + '\n')


def main(argv):
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    amino_to_ix = {amino: index for index, amino in enumerate(amino_acids + ['X'])}
    ix_to_amino = {index: amino for index, amino in enumerate(amino_acids + ['X'])}
    batch_size = 50
    tcrs = load_all_data(argv[1])
    train, test, _, _ = train_test_split(tcrs, tcrs, test_size=0.2)
    max_len = find_max_len(tcrs)
    train_batches = get_batches(train, amino_to_ix, batch_size, max_len)
    test_batches = get_batches(test, amino_to_ix, batch_size, max_len)
    device = argv[2] if torch.cuda.is_available() else 'cpu'
    encoding_dim = int(argv[4])
    model = train_model(train_batches, batch_size, max_len, encoding_dim=encoding_dim,
                        epochs=300, device=device)
    evaluate(test_batches, batch_size, model, ix_to_amino, device)
    torch.save({
        'amino_to_ix': amino_to_ix,
        'ix_to_amino': ix_to_amino,
        'batch_size': batch_size,
        'max_len': max_len,
        'enc_dim': encoding_dim,
        'model_state_dict': model.state_dict(),
    }, argv[3])


if __name__ == '__main__':
    main(sys.argv)
    # argv[1] = 'BM_data_CDR3s'
    # argv[2] = 'cuda:0'
    # argv[3] = 'tcr_autoencoder_model.pt'
    # argv[4] = 30

