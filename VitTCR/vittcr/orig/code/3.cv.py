import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import os, sys, torch, pickle, re
from argparse import ArgumentParser
from collections import OrderedDict


parser = ArgumentParser(description="Specifying Files")
parser.add_argument("-r", "--repeat", default=5, type=int, help="Specify the repeat")
parser.add_argument("-f", "--fold", default=5, help="Specify cv fold")
parser.add_argument("-o", "--outpath", default='../', help="Specify the path of performance files")
args = parser.parse_args()


def plotting_metrics(metric, df_train, df_valid, fold, repeat, path_of_fig):
    plt.figure()
    epochs = range(1, len(df_train[metric]) + 1)
    max_value = max(df_train[metric].max(), df_valid[metric].max())
    plt.plot(epochs, df_train[metric], '--')
    plt.plot(epochs, df_valid[metric], '--')
    plt.title(metric.upper(), fontsize=20)
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel(metric.upper(), fontsize=15)    
    plt.ylim((0, max_value+0.1))
    plt.xlim((0, df_train.shape[0]+2))
    
    ind = df_valid['loss'].idxmin()
    epoch = df_valid.iloc[ind]['epoch']
    value_of_metric = df_valid[metric][epoch-1]
    plt.plot([epoch,epoch],[0,value_of_metric],'r--',lw=1)
    plt.plot([0,epoch],[value_of_metric, value_of_metric],'r--',lw=1)
    plt.text(epoch, value_of_metric,(epoch,float('%.2e'%value_of_metric)),color='black', fontsize=10)
    plt.legend(["train set", 'valid set'], loc='upper right')
    plt.savefig("{}/{}_repeat{}_fold{}.png".format(path_of_fig, metric, repeat, fold))

    
if __name__=="__main__":
    i=0
    log = pd.DataFrame(columns = ['repeat', 'fold', 
                                  'val_loss_min_index', 'val_loss_min_index_AUC', 'val_loss_min_index_PR', 
                                  'val_auc_max_index', 'val_auc_max_index_AUC',
                                  'val_pr_max_index', 'val_pr_max_index_PR'])
    for repeat in range(args.repeat):
        for fold in range(args.fold):
            metricpath = '{}/metrics_repeat{}'.format(args.outpath,repeat)
            figpath = "{}/figures_repeat{}".format(args.outpath, repeat)
            
            if not os.path.exists(figpath):
                os.makedirs(figpath)
            print(repeat)
            print(fold)
            trainfile=[i for i in os.listdir(metricpath) if i.startswith("Train_model") and i.endswith("fold{}.tsv".format(fold))][0]
            validfile=[i for i in os.listdir(metricpath) if i.startswith("Test_model") and i.endswith("fold{}.tsv".format(fold))][0]

            df_train = pd.read_csv('{}/{}'.format(metricpath,trainfile), header=0, delimiter='\t')
            df_valid = pd.read_csv('{}/{}'.format(metricpath,validfile), header=0, delimiter='\t')
            
            val_loss = list(df_valid['loss'])
            val_auc = list(df_valid['auc_roc'])
            val_pr = list(df_valid['auc_pr'])
            
            ind = val_loss.index(min(val_loss))
            val_loss_min_index = ind+1
            val_loss_min_index_AUC = val_auc[ind]
            val_loss_min_index_PR = val_pr[ind]
            print('val loss min index: ', val_loss_min_index)
            print('val loss min index AUC: ', val_loss_min_index_AUC)
            print('val loss min index PR: ', val_loss_min_index_PR)
            
            ind = val_auc.index(max(val_auc))
            val_auc_max_index = ind+1
            val_auc_max_index_AUC = val_auc[ind]
            print('val auc max index: ', val_auc_max_index)
            print('val auc max index AUC: ', val_auc_max_index_AUC)

            ind = val_pr.index(max(val_pr))
            val_pr_max_index = ind+1
            val_pr_max_index_PR = val_pr[ind]
            print('val auc max index: ', val_pr_max_index)
            print('val auc max index AUC: ', val_pr_max_index_PR)

            info = ("repeat_{}".format(repeat), "fold_{}".format(fold), val_loss_min_index, val_loss_min_index_AUC, val_loss_min_index_PR, 
                                        val_auc_max_index, val_auc_max_index_AUC,
                                        val_pr_max_index, val_pr_max_index_PR)
            log.loc[i] = info
            i+=1
            log.to_csv('{}/performance_summary.tsv'.format(args.outpath), sep='\t', index=False)
            plotting_metrics(metric='loss', df_train=df_train, df_valid=df_valid, fold=fold, repeat=repeat, path_of_fig=figpath)
            plotting_metrics(metric='auc_roc', df_train=df_train, df_valid=df_valid, fold=fold, repeat=repeat, path_of_fig=figpath)
            plotting_metrics(metric='auc_pr', df_train=df_train, df_valid=df_valid, fold=fold, repeat=repeat, path_of_fig=figpath)
            plotting_metrics(metric='acc', df_train=df_train, df_valid=df_valid, fold=fold, repeat=repeat, path_of_fig=figpath)

    for para in ['val_loss_min_index_AUC', 'val_loss_min_index_PR', 'val_auc_max_index_AUC', 'val_pr_max_index_PR']:
        log.loc[i, para]=log.loc[:i, para].mean()
        log.loc[i+1, para]=log.loc[:i, para].var()
    
    log.iloc[i,0]='mean'
    log.iloc[i+1,0]='variance'
    log = log.fillna(0)
    log.to_csv('{}/performance_summary.tsv'.format(args.outpath), sep='\t', index=False)    
