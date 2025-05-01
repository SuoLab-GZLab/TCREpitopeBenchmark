import random
import numpy as np
import csv
import os
import sklearn.model_selection as skl

# todo count how many TCRs and peps there are in each set (for TN FN TP FP tables)


def read_data(csv_file, file_key, _protein=False, _hla=False):
    with open(csv_file, 'r', encoding='unicode_escape') as file:
        file.readline()
        if file_key == 'mcpas':
            reader = csv.reader(file)
        elif file_key == 'vdjdb':
            reader = csv.reader(file, delimiter='\t')
        elif file_key == 'tumor':
            reader = csv.reader(file, delimiter='\t')
        elif file_key == 'nettcr':
            reader = csv.reader(file, delimiter='\t')
        tcrs = set()
        peps = set()
        all_pairs = []
        for line in reader:
            if file_key == 'mcpas':
                if _protein:
                    protein = line[9]
                    if protein == 'NA':
                        continue
                if _hla:
                    hla = line[13]
                    if hla == 'NA':
                        continue
                    if line[2] != 'Human':
                        continue
                tcr, pep = line[1], line[11]
            elif file_key == 'vdjdb':
                if _protein:
                    protein = line[10]
                    if protein == 'NA':
                        continue
                if _hla:
                    hla = line[6]
                    if hla == 'NA':
                        continue
                    if line[5] != 'HomoSapiens':
                        continue
                tcr, pep = line[2], line[9]
                if line[1] != 'TRB':
                    continue
            elif file_key == 'tumor':
                tcr, pep = line
            elif file_key == 'nettcr':
                tcr, pep = line[1], line[0]
                # Klinger et al. removed C and F
                tcr = 'C' + tcr + 'F'
            # Proper tcr and peptides
            if any(att == 'NA' or att == "" for att in [tcr, pep]):
                continue
            if any(key in tcr + pep for key in ['#', '*', 'b', 'f', 'y', '~',
                                                'O', '/', '1', 'X', '_', 'B', '7']):
                continue
            tcrs.add(tcr)
            pep_data = [pep]
            if _protein:
                pep_data.append(protein)
            if _hla:
                pep_data.append(hla)
            peps.add(tuple(pep_data))
            all_pairs.append((tcr, tuple(pep_data)))
    train_pairs, test_pairs = train_test_split(all_pairs)
    return all_pairs, train_pairs, test_pairs


def train_test_split(all_pairs):
    '''
    Splitting the TCR-PEP pairs
    '''
    train_pairs = []
    test_pairs = []
    for pair in all_pairs:
        # 80% train, 20% test
        p = np.random.binomial(1, 0.8)
        if p == 1:
            train_pairs.append(pair)
        else:
            test_pairs.append(pair)
    return train_pairs, test_pairs


def positive_examples(pairs):
    examples = []
    for pair in pairs:
        tcr, pep_data = pair
        examples.append((tcr, pep_data, 'p'))
    return examples


def negative_examples(pairs, all_pairs, size, _protein=False):
    '''
    Randomly creating intentional negative examples from the same pairs dataset.
    '''
    examples = []
    i = 0
    # Get tcr and peps lists
    tcrs = [tcr for (tcr, pep_data) in pairs]
    peps = [pep_data for (tcr, pep_data) in pairs]
    while i < size:
        pep_data = random.choice(peps)
        for j in range(5):
            tcr = random.choice(tcrs)
            if _protein:
                tcr_pos_pairs = [pair for pair in all_pairs if pair[0] == tcr]
                tcr_proteins = [pep[1] for (tcr, pep) in tcr_pos_pairs]
                protein = pep_data[1]
                attach = protein in tcr_proteins
            else:
                attach = (tcr, pep_data) in all_pairs
            if attach is False:
                if (tcr, pep_data, 'n') not in examples:
                    examples.append((tcr, pep_data, 'n'))
                    i += 1
    return examples


def read_naive_negs(tcrgp_dir, benny_chain_dir):
    neg_tcrs = []
    for file in os.listdir(tcrgp_dir):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            with open(tcrgp_dir + '/' + filename, 'r') as csv_file:
                csv_file.readline()
                csv_ = csv.reader(csv_file)
                for row in csv_:
                    if row[1] == 'control':
                        tcr = row[-1]
                        neg_tcrs.append(tcr)
    for file in os.listdir(benny_chain_dir):
        filename = os.fsdecode(file)
        is_naive = 'naive' in filename
        if filename.endswith(".cdr3") and 'beta' in filename and is_naive:
            with open(benny_chain_dir + '/' + filename, 'r') as file:
                for row in file:
                    row = row.strip().split(',')
                    tcr = row[0]
                    neg_tcrs.append(tcr)
    train, test, _, _ = skl.train_test_split(neg_tcrs, neg_tcrs, test_size=0.2)
    return train, test


def read_memory_negs(dir):
    neg_tcrs = []
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        is_memory = 'CM' in filename or 'EM' in filename
        if filename.endswith(".cdr3") and 'beta' in filename and is_memory:
            with open(dir + '/' + filename, 'r') as file:
                for row in file:
                    row = row.strip().split(',')
                    tcr = row[0]
                    neg_tcrs.append(tcr)
    train, test, _, _ = skl.train_test_split(neg_tcrs, neg_tcrs, test_size=0.2)
    return train, test


def negative_external_examples(pairs, all_pairs, size, negs, _protein=False):
    examples = []
    i = 0
    # Get tcr and peps lists
    peps = [pep_data for (tcr, pep_data) in pairs]
    while i < size:
        pep_data = random.choice(peps)
        for j in range(5):
            tcr = random.choice(negs)
            if _protein:
                tcr_pos_pairs = [pair for pair in all_pairs if pair[0] == tcr]
                tcr_proteins = [pep[1] for (tcr, pep) in tcr_pos_pairs]
                protein = pep_data[1]
                attach = protein in tcr_proteins
            else:
                attach = (tcr, pep_data) in all_pairs
            if attach is False:
                if (tcr, pep_data, 'n') not in examples:
                    examples.append((tcr, pep_data, 'n'))
                    i += 1
    return examples


def get_examples(pairs_file, key, sampling, _protein=False, _hla=False):
    all_pairs, train_pairs, test_pairs = read_data(pairs_file, key, _protein=_protein, _hla=_hla)
    train_pos = positive_examples(train_pairs)
    test_pos = positive_examples(test_pairs)
    if sampling == 'naive':
        neg_train, neg_test = read_naive_negs('tcrgp_training_data', 'benny_chain')
        train_neg = negative_external_examples(train_pairs, all_pairs, len(train_pos), neg_train, _protein=_protein)
        test_neg = negative_external_examples(test_pairs, all_pairs, len(test_pos), neg_test, _protein=_protein)  # fixed to neg_test, was neg_train before
    elif sampling == 'memory':
        neg_train, neg_test = read_memory_negs('benny_chain')
        train_neg = negative_external_examples(train_pairs, all_pairs, len(train_pos), neg_train, _protein=_protein)
        test_neg = negative_external_examples(test_pairs, all_pairs, len(test_pos), neg_test, _protein=_protein)  # fixed to neg_test, was neg_train before
    elif sampling == 'specific':
        train_neg = negative_examples(train_pairs, all_pairs, 5 * len(train_pos), _protein=_protein)
        test_neg = negative_examples(test_pairs, all_pairs, 5 * len(test_pos), _protein=_protein)
    return train_pos, train_neg, test_pos, test_neg


def load_data(pairs_file, key, sampling, _protein=False, _hla=False):
    if key in ['mcpas', 'vdjdb', 'tumor', 'nettcr']:
        train_pos, train_neg, test_pos, test_neg = get_examples(pairs_file, key, sampling, _protein=_protein, _hla=_hla)
    elif key == 'united':
        mcpas_all_pairs, _, _ = read_data(pairs_file['mcpas'], 'mcpas', _protein=_protein, _hla=_hla)
        vdjdb_all_pairs, _, _ = read_data(pairs_file['vdjdb'], 'vdjdb', _protein=_protein, _hla=_hla)
        print(mcpas_all_pairs + vdjdb_all_pairs)
        all_pairs = list(set(mcpas_all_pairs + vdjdb_all_pairs))
        # split union to train/test
        train_pairs, test_pairs = train_test_split(all_pairs)
        train_pos = positive_examples(train_pairs)
        test_pos = positive_examples(test_pairs)
        if sampling == 'specific':
            train_neg = negative_examples(train_pairs, all_pairs, 5 * len(train_pos), _protein=_protein)
            test_neg = negative_examples(test_pairs, all_pairs, 5 * len(test_pos), _protein=_protein)
    train = train_pos + train_neg
    random.shuffle(train)
    test = test_pos + test_neg
    random.shuffle(test)
    return train, test
    # This - won't work, because mcpas/vdjdb are NOT distinct
    # mcpas_train_pos, mcpas_train_neg, mcpas_test_pos, mcpas_test_neg = \
    #     get_examples(pairs_file['mcpas'], 'mcpas', sampling, _protein=_protein, _hla=_hla)
    # vdjdb_train_pos, vdjdb_train_neg, vdjdb_test_pos, vdjdb_test_neg = \
    #     get_examples(pairs_file['vdjdb'], 'vdjdb', sampling, _protein=_protein, _hla=_hla)
    # train = mcpas_train_pos + mcpas_train_neg + vdjdb_train_pos + vdjdb_train_neg
    # random.shuffle(train)
    # test = mcpas_test_pos + mcpas_test_neg + vdjdb_test_pos + vdjdb_test_neg
    # random.shuffle(test)


def check(file, key, sampling):
    train, test = load_data(file, key, sampling)
    print(train)
    print(test)
    print(len(train))
    print(len(test))


# check('tumor/extended_cancer_pairs', 'tumor', 'specific')
# datafile = {'mcpas': r'data/McPAS-TCR.csv', 'vdjdb': r'data/VDJDB_complete.tsv'}
# check(datafile, 'united', 'specific')
# check('NetTCR/iedb_mira_pos_uniq', 'nettcr', 'specific')
