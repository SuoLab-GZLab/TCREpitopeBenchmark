# THIS IS THE MAIN PYTHON FILE TO RUN
import torch
import pickle
import argparse
import ae_utils as ae
import lstm_utils as lstm
import ergo_data_loader
import numpy as np
from ERGO_models import AutoencoderLSTMClassifier, DoubleLSTMClassifier
import csv


def ae_get_lists_from_pairs(pairs, max_len):
    tcrs = []
    peps = []
    signs = []
    for pair in pairs:
        tcr, pep, label = pair
        if len(tcr) >= max_len:
            continue
        tcrs.append(tcr)
        peps.append(pep[0])
        if label == 'p':
            signs.append(1.0)
        elif label == 'n':
            signs.append(0.0)
    return tcrs, peps, signs


def lstm_get_lists_from_pairs(pairs):
    tcrs = []
    peps = []
    signs = []
    for pair in pairs:
        tcr, pep, label = pair
        tcrs.append(tcr)
        peps.append(pep[0])
        if label == 'p':
            signs.append(1.0)
        elif label == 'n':
            signs.append(0.0)
    return tcrs, peps, signs


def main(args):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    if args.model_type == 'lstm':
        amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    if args.model_type == 'ae':
        pep_atox = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
        tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}

    # hyper-params
    arg = {}
    arg['train_auc_file'] = args.train_auc_file if args.train_auc_file else 'ignore'
    arg['test_auc_file'] = args.test_auc_file if args.test_auc_file else 'ignore'
    if args.test_auc_file == 'auto':
        dir = 'save_results'
        p_key = 'protein' if args.protein else ''
        arg['test_auc_file'] = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key])
    arg['ae_file'] = args.ae_file
    if args.ae_file == 'auto':
        args.ae_file = 'TCR_Autoencoder/tcr_ae_dim_30.pt'
        arg['ae_file'] = 'TCR_Autoencoder/tcr_ae_dim_30.pt'
        pass
    arg['siamese'] = False
    params = {}
    params['lr'] = 1e-4
    params['wd'] = 0
    params['epochs'] = 100
    if args.dataset == 'tumor':
        params['epochs'] = 25
    params['batch_size'] = 50
    params['lstm_dim'] = 500
    params['emb_dim'] = 10
    params['dropout'] = 0.1
    params['option'] = 0
    params['enc_dim'] = 100
    params['train_ae'] = True

    # Load autoencoder params
    if args.model_type == 'ae':
        args.ae_file = 'TCR_Autoencoder/tcr_ae_dim_' + str(params['enc_dim']) + '.pt'
        arg['ae_file'] = args.ae_file
        checkpoint = torch.load(args.ae_file, map_location=args.device)
        params['max_len'] = checkpoint['max_len']
        params['batch_size'] = checkpoint['batch_size']

    # Load data
    if args.dataset == 'mcpas':
        datafile = r'data/McPAS-TCR.csv'
    elif args.dataset == 'vdjdb':
        datafile = r'data/VDJDB_complete.tsv'
    elif args.dataset == 'united':
        datafile = {'mcpas': r'data/McPAS-TCR.csv', 'vdjdb': r'data/VDJDB_complete.tsv'}
    elif args.dataset == 'tumor':
        datafile = r'tumor/extended_cancer_pairs'
    elif args.dataset == 'nettcr':
        datafile = r'NetTCR/iedb_mira_pos_uniq'
    train, test = ergo_data_loader.load_data(datafile, args.dataset, args.sampling,
                                             _protein=args.protein, _hla=args.hla)
    # Save train
    if args.train_data_file == 'auto':
        dir = 'save_results'
        p_key = 'protein' if args.protein else ''
        args.train_data_file = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key, 'train'])
    if args.train_data_file:
        with open(args.train_data_file + '.pickle', 'wb') as handle:
            pickle.dump(train, handle)

    # Save test
    if args.test_data_file == 'auto':
        dir = 'final_results'
        p_key = 'protein' if args.protein else ''
        args.test_data_file = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key, 'test'])
    if args.test_data_file:
        with open(args.test_data_file + '.pickle', 'wb') as handle:
            pickle.dump(test, handle)

    if args.model_type == 'ae':
        # train
        train_tcrs, train_peps, train_signs = ae_get_lists_from_pairs(train, params['max_len'])
        train_batches = ae.get_batches(train_tcrs, train_peps, train_signs, tcr_atox, pep_atox, params['batch_size'], params['max_len'])
        # test
        test_tcrs, test_peps, test_signs = ae_get_lists_from_pairs(test, params['max_len'])
        test_batches = ae.get_batches(test_tcrs, test_peps, test_signs, tcr_atox, pep_atox, params['batch_size'], params['max_len'])
        # Train the model
        model, best_auc, best_roc = ae.train_model(train_batches, test_batches, args.device, arg, params)
        pass
    if args.model_type == 'lstm':
        # train
        train_tcrs, train_peps, train_signs = lstm_get_lists_from_pairs(train)
        lstm.convert_data(train_tcrs, train_peps, amino_to_ix)
        train_batches = lstm.get_batches(train_tcrs, train_peps, train_signs, params['batch_size'])
        # test
        test_tcrs, test_peps, test_signs = lstm_get_lists_from_pairs(test)
        lstm.convert_data(test_tcrs, test_peps, amino_to_ix)
        test_batches = lstm.get_batches(test_tcrs, test_peps, test_signs, params['batch_size'])
        # Train the model
        model, best_auc, best_roc = lstm.train_model(train_batches, test_batches, args.device, arg, params)
        pass

    # Save trained model
    if args.model_file == 'auto':
        dir = 'final_results'
        p_key = 'protein' if args.protein else ''
        args.model_file = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key, 'model.pt'])
    if args.model_file:
        torch.save({
                    'model_state_dict': model.state_dict(),
                    'params': params
                    }, args.model_file)
    if args.roc_file:
        # Save best ROC curve and AUC
        np.savez(args.roc_file, fpr=best_roc[0], tpr=best_roc[1], auc=np.array(best_auc))
    pass


def pep_test(args):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    if args.model_type == 'lstm':
        amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    if args.model_type == 'ae':
        pep_atox = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
        tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}

    if args.ae_file == 'auto':
        args.ae_file = 'TCR_Autoencoder/tcr_ae_dim_30.pt'
    if args.test_data_file == 'auto':
        dir = 'final_results'
        p_key = 'protein' if args.protein else ''
        args.test_data_file = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key, 'test.pickle'])
    if args.model_file == 'auto':
        dir = 'final_results'
        p_key = 'protein' if args.protein else ''
        args.model_file = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key, 'model.pt'])

    # Read test data
    with open(args.test_data_file, 'rb') as handle:
        test = pickle.load(handle)

    device = args.device
    if args.model_type == 'ae':
        test_tcrs, test_peps, test_signs = ae_get_lists_from_pairs(test, 28)
        model = AutoencoderLSTMClassifier(10, device, 28, 21, 30, 50, args.ae_file, False)
        checkpoint = torch.load(args.model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    if args.model_type == 'lstm':
        test_tcrs, test_peps, test_signs = lstm_get_lists_from_pairs(test)
        model = DoubleLSTMClassifier(10, 30, 0.1, device)
        checkpoint = torch.load(args.model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        pass

    # Get frequent peps list
    if args.dataset == 'mcpas':
        datafile = 'McPAS-TCR.csv'
    p = []
    with open(datafile, 'r', encoding='unicode_escape') as file:
        file.readline()
        reader = csv.reader(file)
        for line in reader:
            pep = line[11]
            if pep == 'NA':
                continue
            p.append(pep)
    d = {t: p.count(t) for t in set(p)}
    sorted_d = sorted(d.items(), key=lambda k: k[1], reverse=True)
    peps = [t[0] for t in sorted_d]
    """
    McPAS most frequent peps
    LPRRSGAAGA  Influenza
    GILGFVFTL   Influenza
    GLCTLVAML   Epstein Barr virus (EBV)	
    NLVPMVATV   Cytomegalovirus (CMV)	
    SSYRRPVGI   Influenza
    """
    rocs = []
    for pep in peps[:50]:
        pep_shows = [i for i in range(len(test_peps)) if pep == test_peps[i]]
        test_tcrs_pep = [test_tcrs[i] for i in pep_shows]
        test_peps_pep = [test_peps[i] for i in pep_shows]
        test_signs_pep = [test_signs[i] for i in pep_shows]
        if args.model_type == 'ae':
            test_batches_pep = ae.get_full_batches(test_tcrs_pep, test_peps_pep, test_signs_pep, tcr_atox, pep_atox, 50, 28)
        if args.model_type == 'lstm':
            lstm.convert_data(test_tcrs_pep, test_peps_pep, amino_to_ix)
            test_batches_pep = lstm.get_full_batches(test_tcrs_pep, test_peps_pep, test_signs_pep, 50, amino_to_ix)
        if len(pep_shows):
            try:
                if args.model_type == 'ae':
                    test_auc, roc = ae.evaluate_full(model, test_batches_pep, device)
                if args.model_type == 'lstm':
                    test_auc, roc = lstm.evaluate_full(model, test_batches_pep, device)
                rocs.append((pep, roc))
                print(str(test_auc))
                # print(pep + ', ' + str(test_auc))
            except ValueError:
                print('NA')
                # print(pep + ', ' 'NA')
                pass
    return rocs


def protein_test(args):
    assert args.protein
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    if args.model_type == 'lstm':
        amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    if args.model_type == 'ae':
        pep_atox = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
        tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}

    if args.ae_file == 'auto':
        args.ae_file = 'TCR_Autoencoder/tcr_ae_dim_30.pt'
    if args.test_data_file == 'auto':
        dir = 'final_results'
        p_key = 'protein' if args.protein else ''
        args.test_data_file = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key, 'test.pickle'])
    if args.model_file == 'auto':
        dir = 'final_results'
        p_key = 'protein' if args.protein else ''
        args.model_file = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key, 'model.pt'])

    # Read test data
    with open(args.test_data_file, 'rb') as handle:
        test = pickle.load(handle)

    device = args.device
    if args.model_type == 'ae':
        test_tcrs, test_peps, test_signs = ae_get_lists_from_pairs(test, 28)
        model = AutoencoderLSTMClassifier(10, device, 28, 21, 30, 50, args.ae_file, False)
        checkpoint = torch.load(args.model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    if args.model_type == 'lstm':
        test_tcrs, test_peps, test_signs = lstm_get_lists_from_pairs(test)
        model = DoubleLSTMClassifier(10, 30, 0.1, device)
        checkpoint = torch.load(args.model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        pass

    # Get frequent peps list
    if args.dataset == 'mcpas':
        datafile = 'McPAS-TCR.csv'
    p = []
    protein_peps = {}
    with open(datafile, 'r', encoding='unicode_escape') as file:
        file.readline()
        reader = csv.reader(file)
        for line in reader:
            pep, protein = line[11], line[9]
            if protein == 'NA' or pep == 'NA':
                continue
            p.append(protein)
            try:
                protein_peps[protein].append(pep)
            except KeyError:
                protein_peps[protein] = [pep]

    d = {t: p.count(t) for t in set(p)}
    sorted_d = sorted(d.items(), key=lambda k: k[1], reverse=True)
    proteins = [t[0] for t in sorted_d]
    """
    McPAS most frequent proteins
    NP177   Influenza
    Matrix protein (M1) Influenza
    pp65    Cytomegalovirus (CMV)
    BMLF-1  Epstein Barr virus (EBV)
    PB1 Influenza
    """
    rocs = []
    for protein in proteins[:50]:
        protein_shows = [i for i in range(len(test_peps)) if test_peps[i] in protein_peps[protein]]
        test_tcrs_protein = [test_tcrs[i] for i in protein_shows]
        test_peps_protein = [test_peps[i] for i in protein_shows]
        test_signs_protein = [test_signs[i] for i in protein_shows]
        if args.model_type == 'ae':
            test_batches_protein = ae.get_full_batches(test_tcrs_protein, test_peps_protein, test_signs_protein, tcr_atox, pep_atox, 50,
                                                   28)
        if args.model_type == 'lstm':
            lstm.convert_data(test_tcrs_protein, test_peps_protein, amino_to_ix)
            test_batches_protein = lstm.get_full_batches(test_tcrs_protein, test_peps_protein, test_signs_protein, 50, amino_to_ix)
        if len(protein_shows):
            try:
                if args.model_type == 'ae':
                    test_auc, roc = ae.evaluate_full(model, test_batches_protein, device)
                if args.model_type == 'lstm':
                    test_auc, roc = lstm.evaluate_full(model, test_batches_protein, device)
                rocs.append((pep, roc))
                # print(protein)
                print(str(test_auc))
                # print(protein + ', ' + str(test_auc))
            except ValueError:
                # print(protein)
                print('NA')
                # print(protein + ', ' 'NA')
                pass
    return rocs


def predict(args):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    if args.model_type == 'lstm':
        amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    if args.model_type == 'ae':
        pep_atox = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
        tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}

    # if args.ae_file == 'auto':
    args.ae_file = 'TCR_Autoencoder/tcr_ae_dim_100.pt'
    if args.model_file == 'auto':
        dir = 'models'
        p_key = 'protein' if args.protein else ''
        args.model_file = dir + '/' + '_'.join([args.model_type, args.dataset, args.sampling, p_key, 'model.pt'])
    if args.test_data_file == 'auto':
        args.test_data_file = 'pairs_example.csv'

    # Read test data
    tcrs = []
    peps = []
    signs = []
    max_len = 28
    with open(args.test_data_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            tcr, pep = line
            if args.model_type == 'ae' and len(tcr) >= max_len:
                continue
            tcrs.append(tcr)
            peps.append(pep)
            signs.append(0.0)
    tcrs_copy = tcrs.copy()
    peps_copy = peps.copy()

    # Load model
    device = args.device
    if args.model_type == 'ae':
        model = AutoencoderLSTMClassifier(10, device, 28, 21, 100, 50, args.ae_file, False)
        checkpoint = torch.load(args.model_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
    if args.model_type == 'lstm':
        model = DoubleLSTMClassifier(10, 500, 0.1, device)
        checkpoint = torch.load(args.model_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        pass

    # Predict
    batch_size = 50
    if args.model_type == 'ae':
        test_batches = ae.get_full_batches(tcrs, peps, signs, tcr_atox, pep_atox, batch_size, max_len)
        preds = ae.predict(model, test_batches, device)
    if args.model_type == 'lstm':
        lstm.convert_data(tcrs, peps, amino_to_ix)
        test_batches = lstm.get_full_batches(tcrs, peps, signs, batch_size, amino_to_ix)
        preds = lstm.predict(model, test_batches, device)

    # Print predictions
    for tcr, pep, pred in zip(tcrs_copy, peps_copy, preds):
        print('\t'.join([tcr, pep, str(pred)]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("function")
    parser.add_argument("model_type")
    parser.add_argument("dataset")
    parser.add_argument("sampling")
    parser.add_argument("device")
    parser.add_argument("--protein", action="store_true")
    parser.add_argument("--hla", action="store_true")
    parser.add_argument("--ae_file")
    parser.add_argument("--train_auc_file")
    parser.add_argument("--test_auc_file")
    parser.add_argument("--model_file")
    parser.add_argument("--roc_file")
    parser.add_argument("--train_data_file")
    parser.add_argument("--test_data_file")
    args = parser.parse_args()

    if args.function == 'train':
        main(args)
    elif args.function == 'test' and not args.protein:
        pep_test(args)
    elif args.function == 'test' and args.protein:
        protein_test(args)
    elif args.function == 'predict':
        predict(args)

# example
#  python ERGO.py train lstm mcpas specific cuda:0 --model_file=model.pt --train_data_file=train_data --test_data_file=test_data
