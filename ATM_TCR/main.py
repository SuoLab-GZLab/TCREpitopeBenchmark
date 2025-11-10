import argparse
import os
import sys
import csv
import time
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from data_loader import define_dataloader, load_embedding, load_data_split
from utils import str2bool, timeSince, get_performance_batchiter, print_performance, write_blackbox_output_batchiter

import data_io_tf
device = torch.device('cuda')
# Constants
PRINT_EVERY_EPOCH = 1

def train(model, device, train_loader, optimizer, epoch):

    model.train()

    for batch in train_loader:

        x_pep, x_tcr, y = batch.X_pep.to(
            device), batch.X_tcr.to(device), batch.y.to(device)

        optimizer.zero_grad()
        yhat = model(x_pep, x_tcr)
        y = y.unsqueeze(-1).expand_as(yhat)
        loss = F.binary_cross_entropy(yhat, y)
        loss.backward()
        optimizer.step()

    if epoch % PRINT_EVERY_EPOCH == 1:
        print('[TRAIN] Epoch {} Loss {:.4f}'.format(epoch, loss.item()))


def main():

    parser = argparse.ArgumentParser(description='Prediction of TCR binding to peptide-MHC complexes')

    parser.add_argument('--infile', type=str,
                        help='Input file for training')
    parser.add_argument('--indepfile', type=str, default=None,
                        help='Independent test data file')
    parser.add_argument('--blosum', type=str, default=None,
                        help='File containing BLOSUM matrix to initialize embeddings')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='Training batch size')
    parser.add_argument('--model_name', type=str, default='original.ckpt',
                        help = 'Model name to be saved/loaded for training/independent testing respectively')
    parser.add_argument('--epoch', type=int, default=200, metavar='N',
                        help='The maximum number of epochs to train')
    parser.add_argument('--min_epoch', type=int, default=30,
                        help='The minimum number of epochs to train, early stopping will not be applied until this epoch')
    parser.add_argument('--early_stop', type=str2bool, default=True,
                        help='Use early stopping method')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--cuda', type=str2bool, default=True,
                        help = 'enable cuda')
    parser.add_argument('--seed', type=int, default=1039,
                        help='random seed')
    parser.add_argument('--mode', type=str, default='train',
                        help = 'train or test')
    parser.add_argument('--save_model', type=str2bool, default=True,
                        help = 'save model')
    parser.add_argument('--model', type=str, default='attention',
                        help='Model to import')
    parser.add_argument('--drop_rate', type=float, default=0.25,
                        help='dropout rate')
    parser.add_argument('--lin_size', type=int, default=1024,
                        help='size of linear transformations')
    parser.add_argument('--padding', type=str, default='mid',
                        help='front, end, mid, alignment')
    parser.add_argument('--heads', type=int, default=5,
                        help='Multihead attention head')
    parser.add_argument('--max_len_tcr', type=int, default=20,
                        help='maximum TCR length allowed')
    parser.add_argument('--max_len_pep', type=int, default=22,
                        help='maximum peptide length allowed')
    parser.add_argument('--n_fold', type=int, default=5,
                        help='number of cross-validation folds')
    parser.add_argument('--idx_test_fold', type=int, default=0,
                        help='fold index for test set (0, ..., n_fold-1)')
    parser.add_argument('--idx_val_fold', type=int, default=-1,
                        help='fold index for validation set (-1, 0, ..., n_fold-1). \
                              If -1, the option will be ignored \
                              If >= 0, the test set will be set aside and the validation set is used as test set') 
    parser.add_argument('--split_type', type=str, default='random',
                        help='how to split the dataset (random, tcr, epitope)')
    args = parser.parse_args()

    if args.mode == 'test':
        assert args.indepfile is not None, '--indepfile is missing!'
    assert args.idx_test_fold < args.n_fold, '--idx_test_fold should be smaller than --n_fold'
    assert args.idx_val_fold < args.n_fold, '--idx_val_fold should be smaller than --n_fold'
    assert args.idx_val_fold != args.idx_test_fold, '--idx_val_fold and --idx_test_fold should not be equal to each other'

    # Set Cuda
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device('cuda' if args.cuda else 'cpu')

    # Set random seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load embedding matrix
    embedding_matrix = load_embedding(args.blosum)

    # Read data
    x_pep, x_tcr, y = data_io_tf.read_pTCR(args.infile)
    y = np.array(y)

    # Shuffle data into folds for cross validation
    idx_train, idx_test, idx_test_remove = load_data_split(x_pep, x_tcr, args)
    


    # Define dataloader
    train_loader = define_dataloader(x_pep[idx_train], x_tcr[idx_train], y[idx_train],
                                     args.max_len_pep, args.max_len_tcr,
                                     padding=args.padding,
                                     batch_size=args.batch_size, device=device)
    test_loader = define_dataloader(x_pep[idx_test], x_tcr[idx_test], y[idx_test],
                                    maxlen_pep=train_loader['pep_length'],
                                    maxlen_tcr=train_loader['tcr_length'],
                                    padding=args.padding,
                                    batch_size=args.batch_size, device=device)
    if args.indepfile is not None:
        x_indep_pep, x_indep_tcr, y_indep = data_io_tf.read_pTCR(args.indepfile)
        y_indep = np.array(y_indep)
        indep_loader = define_dataloader(x_indep_pep, x_indep_tcr, y_indep, 
                                         maxlen_pep=train_loader['pep_length'],
                                         maxlen_tcr=train_loader['tcr_length'],
                                         padding=args.padding,
                                         batch_size=args.batch_size, device=device)

    args.pep_length = train_loader['pep_length']
    args.tcr_length = train_loader['tcr_length']

    # Define model
    if args.model == 'attention':
        from attention import Net
    else:
        raise ValueError('unknown model name')

    model = Net(embedding_matrix, args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create Required Directories
    if 'models' not in os.listdir('.'):
        os.mkdir('models')
    if 'result' not in os.listdir('.'):
        os.mkdir('result')

    # eax1it model
    if args.mode == 'train':
        wf_open = open(
            'result/perf_' + os.path.splitext(os.path.basename(args.model_name))[0] + '.csv', 'w')
        wf_colnames = ['loss', 'accuracy',
                       'precision1', 'precision0',
                       'recall1', 'recall0',
                       'f1macro', 'f1micro', 'auc']
        wf = csv.DictWriter(wf_open, wf_colnames, delimiter='\t')

        t0 = time.time()
        lossArraySize = 10
        lossArray = deque([sys.maxsize], maxlen=lossArraySize)
        for epoch in range(1, args.epoch + 1):

            train(model, device, train_loader['loader'], optimizer, epoch)
            perf_test = get_performance_batchiter(
                test_loader['loader'], model, device)

            # Print performance
            if epoch % PRINT_EVERY_EPOCH == 0:
                print('[TEST ] {} ----------------'.format(epoch))
                print_performance(perf_test, printif=False,
                                  writeif=True, wf=wf)

            # Check for early stopping
            lossArray.append(perf_test['loss'])
            average_loss_change = sum(np.abs(np.diff(lossArray))) / lossArraySize
            if epoch > args.min_epoch and average_loss_change < 10 and args.early_stop:
                print('Early stopping at epoch {}'.format(epoch))
                break

        print(os.path.splitext(os.path.basename(args.model_name))[0])
        print(timeSince(t0))

        # evaluate and print independent-test-set performance
        if args.indepfile is not None:
            print('[INDEP] {} ----------------')
            perf_indep = get_performance_batchiter(
                indep_loader['loader'], model, device)

            wf_open = open('result/perf_' + os.path.splitext(os.path.basename(args.model_name))[0] + '_' +
                           os.path.basename(args.indepfile), 'w')
            wf = csv.DictWriter(wf_open, wf_colnames, delimiter='\t')
            print_performance(perf_indep, writeif=True, wf=wf)

            wf_open1 = open('data/pred_' + os.path.splitext(os.path.basename(args.model_name))[0] + '_' +
                            os.path.basename(args.indepfile), 'w')
            wf1 = csv.writer(wf_open1, delimiter='\t')
            write_blackbox_output_batchiter(
                indep_loader, model, wf1, device, ifscore=True)

        # evaluate and print test-set performance
        print('[TEST ] {} ----------------'.format(epoch))
        perf_test = get_performance_batchiter(
            test_loader['loader'], model, device)
        print_performance(perf_test)

        if args.save_model:

            wf_open1 = open(
                'result/pred_' + os.path.splitext(os.path.basename(args.model_name))[0] + '.csv', 'w')
            wf1 = csv.writer(wf_open1, delimiter='\t')
            write_blackbox_output_batchiter(
                test_loader, model, wf1, device, ifscore=True)

            model_name = './models/' + \
                os.path.splitext(os.path.basename(args.model_name))[0] + '.ckpt'
            torch.save(model.state_dict(), model_name)
    
    elif args.mode == 'test':

        model_name = args.model_name

        assert model_name in os.listdir('./models')

        model_name = './models/' + model_name
        model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

        # evaluate and print independent-test-set performance
        print('[INDEP] {} ----------------')
        perf_indep = get_performance_batchiter(
            indep_loader['loader'], model, device)
        print_performance(perf_indep)

        # write blackbox output
        wf_bb_open1 = open('result/pred_' + os.path.splitext(os.path.basename(model_name))[0] + '_' +
                           os.path.basename(args.indepfile), 'w')
        wf_bb1 = csv.writer(wf_bb_open1, delimiter='\t')
        write_blackbox_output_batchiter(
            indep_loader, model, wf_bb1, device, ifscore=True)

    else:
        print('\nError: "--mode train" or "--mode test" expected')

if __name__ == '__main__':
    main()
