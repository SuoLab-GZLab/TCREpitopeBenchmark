import os
gpu_id = '5'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
import torch
import datetime, sys, pickle, random
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
from TcrPepTransform_utils import *
from TcrPepTransform_beta import *
from utils_train import *


parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-b", "--batchsize", default=512, type=int, help="Specify the batch size for minibatch")
parser.add_argument("-r", "--repeat", default=5, type=int, help="Specify the repeat")
parser.add_argument("-f", "--fold", default=5, help="Specify cv fold")
parser.add_argument("-ip", "--inpath", default="../datasets/cross_validation_clustered/", help="Specify input path")
parser.add_argument("-e", "--epochs", default=1000, type=int, help="Specify the number of epochs")
parser.add_argument("-lr", "--learningrate", default=0.001, type=float, help="Specify the learning rate")
parser.add_argument("-n", "--modelname", default='vitTCR', help="Specify the name of model for training")
parser.add_argument("-c", "--chain", default='beta', help="Specify mode of train, alpha/beta")
parser.add_argument("-s", "--seed", default=1234, type=int, help="Specify the random seed")
args = parser.parse_args()


def _init_fn(worker_id): 
    random.seed(10 + worker_id)
    np.random.seed(10 + worker_id)
    torch.manual_seed(10 + worker_id)
    torch.cuda.manual_seed(10 + worker_id)
    torch.cuda.manual_seed_all(10 + worker_id)



# setup_seed(args.seed)
if __name__ =="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for repeat in range(args.repeat): 
        for fold in range(args.fold):
            trainfile = "{}/fold{}/atchleypickle/train.pickle".format(args.inpath, fold)
            validfile = "{}/fold{}/atchleypickle/test.pickle".format(args.inpath, fold)
            
            resultpath1 = "../metrics_repeat{}/Train_model{}_lr{}_fold{}.tsv".format(repeat, args.modelname, args.learningrate, fold)
            resultpath2 = "../metrics_repeat{}/Test_model{}_lr{}_fold{}.tsv".format(repeat, args.modelname, args.learningrate, fold)
        
            modelpath = '../model_repeat{}/fold{}'.format(repeat, fold)
            metricpath = '../metrics_repeat{}'.format(repeat)

            if not os.path.exists(modelpath):
                os.makedirs(modelpath)
            
            if not os.path.exists(metricpath):
                os.makedirs(metricpath)
                
            # Model initializing and compilation
            model = TcrPepTransform_single(input_height=20, 
                                        input_width=12, 
                                        in_chans=5, 
                                        patch_size=4, 
                                        num_classes=2, 
                                        embed_dim=256, 
                                        depth=1, 
                                        num_heads=4, 
                                        mlp_ratio=4, 
                                        qkv_bias=True, 
                                        drop_rate=0.1, 
                                        attn_drop_rate=0.05, 
                                        drop_path_rate=0, 
                                        act_layer=nn.GELU)
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.to(device)
            model.loss_func = torch.nn.CrossEntropyLoss()
            model.acc_func = metrics.accuracy_score
            model.auc_roc_func = metrics.roc_auc_score
            model.auc_pr_func = pr_auc_score
            model.f1_score_func = metrics.f1_score
            model.optimizer = torch.optim.Adam(model.parameters(),lr=args.learningrate)

            # Prepare dataset for training
            with open(trainfile, 'rb') as f:
                dicts = pickle.load(f)
            label_train = np.array(dicts['Binding'])#[index_train]
            label_train_initialized = Labels_Initialization(num_classes=2, labels=label_train)
            if args.chain=='beta':
                print("Beta-related data loading...")
                data_train = np.array(dicts['combined_map_beta'])#[index_train]
                data_train = torch.from_numpy(data_train)
            elif args.chain=='alpha':
                print("Alpha-related data loading...")
                data_train = np.array(dicts['combined_map_alpha'])#[index_train]
                data_train = torch.from_numpy(data_train)
            dataset_train = MyDataset(data=data_train, labl=label_train_initialized)
            dl_train = DataLoader(dataset_train, batch_size=args.batchsize, shuffle=True, drop_last=True, num_workers=3, worker_init_fn=_init_fn)
            now = datetime.datetime.now()
            dfhistory = train_model(model=model,       
                                    epochs=args.epochs, 
                                    dl_train=dl_train, 
                                    trained_model_dir=modelpath,
                                    device=device)
            dfhistory.to_csv(resultpath1, header=True, sep='\t', index=False)

            # Prepare dataset for validating
            with open(validfile, 'rb') as f:
                dicts_valid = pickle.load(f)
            label_valid = np.array(dicts_valid['Binding'])
            label_valid_initialized = Labels_Initialization(num_classes=2, labels=label_valid)
            if args.chain=='beta':
                print("Beta-related data loading...")
                data_valid = np.array(dicts_valid['combined_map_beta'])
                data_valid = torch.from_numpy(data_valid)
            elif args.chain=='alpha':
                print("Alpha-related data loading...")
                data_valid = np.array(dicts_valid['combined_map_alpha'])
                data_valid = torch.from_numpy(data_valid)
            dataset_valid = MyDataset(data=data_valid, labl=label_valid_initialized)
            dl_valid = DataLoader(dataset_valid, batch_size=len(data_valid), shuffle=True, drop_last=True, num_workers=3)
            dfhistory = eval_model(model_initial=model, 
                                epochs=args.epochs,
                                fold=fold,
                                dl_valid=dl_valid, 
                                repeat=repeat, 
                                device=device)
            dfhistory.to_csv(resultpath2, header=True, sep='\t', index=False)