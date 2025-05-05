import os
# gpu_id = '5'
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
parser.add_argument("-itr", "--testinput", default='../datasets/independent_testset/independent_test/independent_testset.pickle', help="Specify the test input files")
parser.add_argument("-tn", "--testname", default='benchmark')
parser.add_argument("-c", "--chain", default='beta', help="Specify the mode")
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


def plot_roc(fprs, tprs, aucs, path_to_fig):
    plt.figure(figsize=(10,10))
    for i in range(len(fprs)):
        plt.plot(fprs[i], tprs[i], lw=1, alpha=0.7, label='ROC (AUC = %0.4f)' % (auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate',fontsize=18)
    plt.ylabel('True Positive Rate',fontsize=18)
    plt.title('ROC of our model on {} dataset'.format(antigen),fontsize=18)
    plt.legend(loc="lower right", prop={'size': 15})
    plt.savefig(path_to_fig)        


if __name__=="__main__":
    resultpath = '../performance_summary_independent_{}.tsv'.format(args.testname) 
    
    # Data loading
    with open(args.testinput, 'rb') as f:
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
            log.to_csv(resultpath, sep='\t', index=False)

    mean_df = list(log.iloc[:,1:].mean(axis=0))
    mean_df.insert(0, 'mean')
    mean_df[1] = int(mean_df[1])
    log.loc[i] = mean_df
    log.to_csv(resultpath, sep='\t', index=False) 