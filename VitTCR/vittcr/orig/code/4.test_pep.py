import os
# gpu_id = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
import datetime, sys, pickle, torch
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
from collections import OrderedDict
from TcrPepTransform_beta import *
from TcrPepTransform_utils import *
import matplotlib.pyplot as plt
from matplotlib import pyplot


#Args parser 
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-r", "--repeat", default=5, type=int, help="Specify the repeat")
parser.add_argument("-f", "--fold", default=5, help="Specify cv fold")
parser.add_argument("-c", "--chain", default='beta', help="Specify the mode")

parser.add_argument("-a", "--aimpath", default='./')

parser.add_argument("-d", "--datapath", default='../unseen_atchleypickle')
parser.add_argument("-od", "--outdir", default='performance_pep_unseen')
parser.add_argument("-of", "--outfile", default='performance_pep_unseen.tsv')
args = parser.parse_args()
chain = args.chain


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum    
    return s


def predicting(model, origin_label, data_test, epoch):
    out_eval = model(data_test)
    out_eval = out_eval.detach().numpy()
    out_eval = softmax(out_eval)
    pred_prob = out_eval[:,1]
    pred_label = np.argmax(out_eval, axis=1)
    fpr, tpr, threshold = metrics.roc_curve(origin_label, pred_prob, pos_label=1, drop_intermediate=False)
    auc = metrics.auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(origin_label, pred_prob)
    pr_auc = metrics.auc(recall, precision)  
    return tpr, fpr, auc, precision, recall, pr_auc, pred_label


if __name__=="__main__":
    datapath = args.datapath
    aimpath = args.aimpath
    
    outdir = '{}/{}'.format(aimpath, args.outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    j = 0
    df_pep = pd.DataFrame(columns = ['pep','epoch', 'val_loss_min_index_AUC_test', 'val_loss_min_index_PR_test'])
    for item in os.listdir(datapath):
        peptide     = item.split('.')[0]
        datadir     = '{}/{}'.format(datapath, item)
        performance = '{}/{}.tsv'.format(outdir, peptide)
        
        # Data loading
        with open(datadir, 'rb') as f:
            dicts_test = pickle.load(f)
        labels_external = np.array(dicts_test['Binding'])
        labels_external = torch.Tensor(labels_external).long()
        
        if chain=='alpha':
            data_external = np.array(dicts_test['combined_map_alpha'])
        elif chain=='beta':
            data_external = np.array(dicts_test['combined_map_beta'])
        data_external = torch.from_numpy(data_external).float()

        # Model testing
        i=0
        log = pd.DataFrame(columns = ['repeat','epoch', 'val_loss_min_index_AUC_test', 'val_loss_min_index_PR_test'])  
        for repeat in range(args.repeat):
            for fold in range(args.fold):
                metricpath = '../metrics_repeat{}'.format(repeat)        
                validfile=[i for i in os.listdir(metricpath) if i.startswith("Test_model") and i.endswith("fold{}.tsv".format(fold))][0]

                validdata = pd.read_csv("{}/{}".format(metricpath,validfile), sep='\t')            
                ind = validdata['loss'].idxmin()
                epoch = validdata.iloc[ind]['epoch']
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
                params = "../model_repeat{}/fold{}/{}_epochs.pt".format(repeat, fold, str(int(epoch)))  
                
                checkpoint = torch.load(params, map_location='cpu')
                model.load_state_dict(checkpoint)
                model.eval()

                tpr, fpr, auc, precision, recall, pr_auc, pred_label = predicting(model=model, origin_label=labels_external, data_test=data_external, epoch=epoch)
                info = (repeat, epoch, auc, pr_auc)
                log.loc[i] = info
                i+=1
                # log.to_csv(performance, sep='\t', index=False)

        mean_df = list(log.iloc[:,1:].mean(axis=0))
        mean_df.insert(0, 'mean')
        mean_df[1] = int(mean_df[1])
        mean_df[-1] = "%.4f" % mean_df[-1]
        mean_df[-2] = "%.4f" % mean_df[-2]
        log.loc[i] = mean_df
        log.to_csv(performance, sep='\t', index=False) 

        pep_mean = mean_df
        pep_mean[0] = peptide
        print(pep_mean)
        df_pep.loc[j] = pep_mean
        j += 1
    
    df_pep.to_csv('{}/{}'.format(aimpath, args.outfile), sep='\t', index=False)