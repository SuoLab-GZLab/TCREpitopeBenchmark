import os, datetime, sys, pickle, torch
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


#Args parser 
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-r", "--repeat", default=5, type=int, help="Specify the repeat")
parser.add_argument("-f", "--fold", default=5, help="Specify cv fold")
parser.add_argument("-c", "--chain", default='beta', help="Specify the mode")
args = parser.parse_args()
chain = args.chain


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1, keepdims = True) #如果是列向量，则axis=0
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
    resultpath = "../PPV_original"
    # resultpath = "../PPV_testset_only"
    data_file = "../datasets/independent_testset/independent_test/independent_testset.pickle"
    # data_file = "../datasets/independent_testset/independent_test_clustered/independent_test_clustered.pickle"

    if not os.path.exists(resultpath):
        os.makedirs(resultpath)
    
    with open(data_file, 'rb') as f:
        dicts_test = pickle.load(f)
    cdr3_external = np.array(dicts_test['cdr3'])
    labels_external = np.array(dicts_test['Binding'])
    labels_external = torch.Tensor(labels_external).long()
    if chain=='alpha':
        data_external = np.array(dicts_test['combined_map_alpha'])
    elif chain=='beta':
        data_external = np.array(dicts_test['combined_map_beta'])
    data_external = torch.from_numpy(data_external).float()


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

            # Calculating...
            namelist = []
            auc_list = []
            antigens = dicts_test['peptide']
            df_all_info = pd.DataFrame()
            for antigen in set(antigens):
                namelist.append(antigen)
                index_sel = [i for i,x in enumerate(antigens) if x==antigen]
                data_test = data_external[index_sel]
                cdr3_test = cdr3_external[index_sel]
                label_test = labels_external[index_sel]
                length_of_dataset = len(index_sel)
                tpr, fpr, auc, precision, recall, pr_auc, pred_label = predicting(model=model, origin_label=label_test, data_test=data_test, epoch=epoch)
                auc_list.append(auc)
                external_data_sub = pd.DataFrame({'cdr3b':cdr3_test, 'antigen':antigen, 'Binding':label_test, 'predicted_label':pred_label})
                external_data_sub = external_data_sub.reset_index(drop=True)
                confusion_list = []
                for idx in external_data_sub.index:
                    if external_data_sub.Binding[idx]==1 and external_data_sub.predicted_label[idx]==1:
                        confusion_list.append('TP')
                    elif external_data_sub.Binding[idx]==0 and external_data_sub.predicted_label[idx]==0:
                        confusion_list.append('TN')
                    elif external_data_sub.Binding[idx]==1 and external_data_sub.predicted_label[idx]==0:
                        confusion_list.append('FN')
                    elif external_data_sub.Binding[idx]==0 and external_data_sub.predicted_label[idx]==1:
                        confusion_list.append('FP')
                external_data_sub['confusion_list'] = confusion_list
                df_all_info = pd.concat([df_all_info, external_data_sub], axis=0)
            
            df_all_info.to_csv('{}/allinfo_repeat{}_fold{}.tsv'.format(resultpath, repeat, fold), sep='\t', index=False)