import os
import re
import sys
import time
import math
import torch
import argparse
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from data_loader import AMINO_MAP, AMINO_MAP_REV, AMINO_MAP_REV_
from collections import defaultdict

BASICITY = {'A': 206.4, 'B': 210.7, 'C': 206.2, 'D': 208.6, 'E': 215.6, 'F': 212.1, 'G': 202.7,
            'H': 223.7, 'I': 210.8, 'K': 221.8, 'L': 209.6, 'M': 213.3, 'N': 212.8, 'P': 214.4,
            'Q': 214.2, 'R': 237.0, 'S': 207.6, 'T': 211.7, 'V': 208.7, 'W': 216.1, 'X': 210.2,
            'Y': 213.1, 'Z': 214.9, '*': 213.1, '@': 0}

HYDROPHOBICITY = {'A': 0.16, 'B': -3.14, 'C': 2.50, 'D': -2.49, 'E': -1.50, 'F': 5.00, 'G': -3.31,
                  'H': -4.63, 'I': 4.41, 'K': -5.00, 'L': 4.76, 'M': 3.23, 'N': -3.79, 'P': -4.92,
                  'Q': -2.76, 'R': -2.77, 'S': -2.85, 'T': -1.08, 'V': 3.02, 'W': 4.88, 'X': 4.59,
                  'Y': 2.00, 'Z': -2.13, '*': -0.25, '@': 0}

HELICITY = {'A': 1.24, 'B': 0.92, 'C': 0.79, 'D': 0.89, 'E': 0.85, 'F': 1.26, 'G': 1.15, 'H': 0.97,
            'I': 1.29, 'K': 0.88, 'L': 1.28, 'M': 1.22, 'N': 0.94, 'P': 0.57, 'Q': 0.96, 'R': 0.95,
            'S': 1.00, 'T': 1.09, 'V': 1.27, 'W': 1.07, 'X': 1.29, 'Y': 1.11, 'Z': 0.91, '*': 1.04, '@': 0}

MUTATION_STABILITY = {'A': 13.0, 'B': 8.5, 'C': 52.0, 'D': 11.0, 'E': 12.0, 'F': 32.0, 'G': 27.0, 'H': 15.0,
                      'I': 10.0, 'K': 24.0, 'L': 34.0, 'M':  6.0, 'N':  6.0, 'P': 20.0, 'Q': 10.0, 'R': 17.0,
                      'S': 10.0, 'T': 11.0, 'V': 17.0, 'W': 55.0, 'X': 20.65, 'Y': 31.0, 'Z': 11.0, '*': 20.65, '@': 0}


def cuda(tensor, is_cuda):

    if is_cuda:
        return tensor.cuda()
    else:
        return tensor


def idxtobool(idx, size, is_cuda):

    V = cuda(torch.zeros(size, dtype=torch.float), is_cuda)
    if len(size) > 2:

        for i in range(size[0]):
            for j in range(size[1]):
                subidx = idx[i, j, :]
                V[i, j, subidx] = float(1)
    elif len(size) == 2:

        for i in range(size[0]):
            subidx = idx[i, :]
            V[i, subidx] = float(1)

    else:
        raise argparse.ArgumentTypeError('len(size) should be larger than 1')

    return V


def create_tensorboard(tensorboard_name):

    tbf = None
    summary_dir = Path('tensorboards').joinpath(tensorboard_name)
    if not summary_dir.exists():
        summary_dir.mkdir(parents=True)
    tbf = SummaryWriter(log_dir=str(summary_dir))

    return tbf


def write_blackbox_output_batchiter(loader, model, wf, device='cpu', ifscore=True):

    model.eval()

    rev_peploader = loader['pep_amino_idx']
    rev_tcrloader = loader['tcr_amino_idx']
    loader = loader['loader']
    for batch in loader:

        X_pep, X_tcr, y = batch.X_pep.to(
            device), batch.X_tcr.to(device), batch.y.to(device)
        score = model(X_pep, X_tcr).data.cpu().tolist()
        score = [s[0] for s in score]
        pred = [round(s) for s in score]

        for i in range(len(pred)):

            pep_seq = ''.join([rev_peploader[x] for x in X_pep[i]])
            pep_seq = re.sub(r'<pad>', '', pep_seq)
            pep_seq = re.sub(r'@', '', pep_seq)
            tcr_seq = ''.join([rev_tcrloader[x] for x in X_tcr[i]])
            tcr_seq = re.sub(r'<pad>', '', tcr_seq)
            tcr_seq = re.sub(r'@', '', tcr_seq)
            if ifscore:
                wf.writerow([pep_seq, tcr_seq, int(y[i]),
                             int(pred[i]), float(score[i])])
            else:
                wf.writerow([pep_seq, tcr_seq, int(pred[i])])


def get_performance_batchiter(loader, model, device='cpu'):
    '''
    print classification performance for binary task

    Args:
     loader  - data loader
     model   - classification model
     loss_ft - loss function
    '''
    model.eval()

    loss = 0
    score, label = [], []
    for batch in loader:

        X_pep, X_tcr, y = batch.X_pep.to(
            device), batch.X_tcr.to(device), batch.y.to(device)
        yhat = model(X_pep, X_tcr)
        y = y.unsqueeze(-1).expand_as(yhat)
        loss += F.binary_cross_entropy(yhat, y, reduction='sum').item()
        score.extend(yhat.data.cpu().tolist())
        label.extend(y.data.cpu().tolist())

    perf = get_performance(score, label)
    perf['loss'] = round(loss, 4)

    return perf


def get_performance(score, label):
    '''
    get classification performance for binary task

    Args:
     score - 1D np.array or list
     label - 1D np.array or list
    '''

    accuracy = None
    precision1, precision0 = None, None
    recall1, recall0 = None, None
    f1macro, f1micro = None, None
    auc = None

    # if type(score) is list():
    #    score = np.array(score)
    # if type(label) is list():
    #    label = np.array(label)

    label_pred = [round(s[0]) for s in score]
    accuracy = accuracy_score(label, label_pred)
    precision1 = precision_score(
        label, label_pred, pos_label=1, zero_division=0)
    precision0 = precision_score(
        label, label_pred, pos_label=0, zero_division=0)
    recall1 = recall_score(label, label_pred, pos_label=1, zero_division=0)
    recall0 = recall_score(label, label_pred, pos_label=0, zero_division=0)
    f1macro = f1_score(label, label_pred, average='macro')
    f1micro = f1_score(label, label_pred, average='micro')
    auc = roc_auc_score(np.array(label), np.array(score)) if len(
        np.unique(np.array(label))) != 1 else -1

    ndigits = 4
    performance = {'accuracy': round(accuracy, ndigits),
                   'precision1': round(precision1, ndigits), 'precision0': round(precision0, ndigits),
                   'recall1': round(recall1, ndigits), 'recall0': round(recall0, ndigits),
                   'f1macro': round(f1macro, ndigits), 'f1micro': round(f1micro, ndigits),
                   'auc': round(auc, ndigits)}
    tn, fp, fn, tp = confusion_matrix(label, label_pred, labels=[0, 1]).ravel()
    print(tn, fp, fn, tp)
    return performance


def print_performance(perf, printif=True, writeif=False, boardif=False, **kargs):
    '''
    print classification performance for binary task

    Args:
     per   - dictionary with measure name as keys and performance as values 
             or perf = get_performance(score, label)
     kargs - epoch, loss, global_step
             wf = open(outfile_name, 'w')
             tbf = create_tensorboard(tensorboard_name)
    '''

    measures = sorted(perf.keys())

    if printif:
        maxchrlen = max([len(x) for x in measures])
        for mea in measures:
            print(mea + ' ' * (maxchrlen - len(mea)) +
                  ' {:.4f}'.format(perf[mea]))
        print('')

    if boardif:
        assert 'tbf' in kargs.keys(), 'missing argument: tbf'
        assert 'global_step' in kargs.keys(), 'missing argument: global_step'
        assert 'mode' in kargs.keys(), 'missing argument: mode'
        for mea in measures:
            kargs['tbf'].add_scalars(main_tag='performance/{}'.format(mea),
                                     tag_scalar_dict={
                                         kargs['mode']: perf[mea]},
                                     global_step=kargs['global_step'])

    if writeif:
        assert 'wf' in kargs.keys(), 'missing argument: wf'
        #newrow = [perf[x] for x in measures]
        # kargs['wf'].writerow(newrow)
        kargs['wf'].writerow(perf)
        return kargs['wf']


def str2bool(v):
    """
    Convert string to boolean object

    """

    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True', 'Y', 'Yes', 'YES', 'YEs', 'ye'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False', 'N', 'NO', 'No'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def timeSince(since):
    """
    Credit: https://github.com/1Konny/VIB-pytorch
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60

    return '%dm %ds' % (m, s)


def check_model_name(model_name, file_path='models'):
    """
    Check whether model name is overlapped or not 
    """
    if model_name in os.listdir(file_path):

        valid = {"yes": True, "y": True, "ye": True, 'true': True, 't': True,
                 '1': True, "no": False, "n": False, 'false': False, 'f': False, '0': False}
        sys.stdout.write(
            "The file {} already exists. Do you want to overwrite it? [yes/no]".format(model_name))
        choice = input().lower()

        if choice in valid:
            if not valid[choice]:
                sys.stdout.write(
                    "Please assign another name. (ex. 'original_2.ckpt')")
                model_name = input().lower()
                check_model_name(model_name=model_name, file_path=file_path)

        else:
            sys.stdout.write("Please respond with 'yes' or 'no'\n")
            check_model_name(model_name=model_name, file_path=file_path)

    return model_name


def get_physchem_properties_batchiter(loader, lenpep, lentcr, device='cpu'):
    '''
    print physiochemical properties 

    Args:
     loader  - data loader
     args   - arguments
    '''

    global features
    features = None

    for batch in loader:

        # Input
        X_pep, X_tcr, y_true = batch.X_pep.to(
            device), batch.X_tcr.to(device), batch.y.to(device)

        get_physchem_properties(X_pep[np.where(y_true == 1.0)].tolist(),
                                X_tcr[np.where(y_true == 1.0)].tolist(),
                                lenpep, lentcr,
                                exclude=set(['X', '*']))

    return features


def get_physchem_properties(pep_batch, tcr_batch,
                            max_len_pep, max_len_tcr, exclude):

    global features

    pep_batch = num2seq(pep_batch, AMINO_MAP_REV_, max_len=max_len_pep,
                        align=False, exclude=exclude)
    tcr_batch = num2seq(tcr_batch, AMINO_MAP_REV, max_len=max_len_tcr,
                        align=False, exclude=exclude)

    if not features:
        features = defaultdict(lambda: defaultdict(list))

    for pep, tcr in zip(pep_batch, tcr_batch):
        features[pep]['tcr'].append(tcr)
        features[pep]['basicity'].append([BASICITY[aa] for aa in tcr])
        features[pep]['hydrophobicity'].append(
            [HYDROPHOBICITY[aa] for aa in tcr])
        features[pep]['helicity'].append([HELICITY[aa] for aa in tcr])
        features[pep]['mutation_stability'].append(
            [MUTATION_STABILITY[aa] for aa in tcr])

        # features[pep]['length'].append(len(tcr))
        # features[pep]['fast_mass'].append(mass.fast_mass(tcr))
        # features[pep]['pI'].append(electrochem.pI(tcr))
        #ac_comp = parser.amino_acid_composition(tcr)
        # for aa in AMINO_MAP_REV[:-2]:
        #    features[pep][aa].append(ac_comp[aa])


def print_physchem_properties(perf, wf, measures=None):

    if measures is None:
        measures = sorted(perf.keys())

    wf.writerow(measures)
    for i in range(len(perf[measures[0]])):
        wf.writerow([perf[mea][i] for mea in measures])


def seq2num(seq_list, mapping, max_len=None, align=True):

    num_list = []

    if align:

        for seq in seq_list:

            if max_len is None:
                num = [mapping[seq[i]] for i in range(len(num))]
            elif max_len > len(seq):
                num = [mapping[seq[i]] for i in range(
                    len(seq))] + [mapping['<pad>'] for _ in range(max_len - len(seq))]
            else:
                num = [mapping[seq[i]] for i in range(max_len)]

            num_list.append(num)

    else:

        for seq in seq_list:

            if max_len is None or max_len > len(seq):
                num = [mapping[seq[i]]
                       for i in range(len(num)) if seq[i] != '<pad>']
            else:
                num = [mapping[seq[i]]
                       for i in range(len(seq)) if seq[i] != '<pad>']

            num_list.append(num)

    return num_list


def num2seq(num_list, mapping, max_len=None, align=True, exclude=set(['@'])):

    seq_list = []

    if align:

        for num in num_list:

            if max_len is None:
                seq = [mapping[num[i]] for i in range(len(num))]
            elif max_len > len(num):
                seq = [mapping[num[i]] for i in range(len(num))]
            else:
                seq = [mapping[num[i]] for i in range(max_len)]

            seq_list.append(''.join(seq))

    else:

        for num in num_list:

            if max_len is None or max_len > len(num):
                seq = [mapping[num[i]]
                       for i in range(len(num)) if mapping[num[i]] not in exclude]
            else:
                seq = [mapping[num[i]]
                       for i in range(max_len) if mapping[num[i]] not in exclude]

            seq_list.append(''.join(seq))

    return seq_list
