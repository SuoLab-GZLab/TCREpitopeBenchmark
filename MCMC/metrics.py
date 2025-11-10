import os
os.chdir('/home/wangyuyan/TCREpitope/MCMC/')
import pandas as pd
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score
import data_processing as dp

aa_vec = pk.load(open('atchley.pk', 'rb'))
    
def viz_conf_mat(conf_mat, vc, title, acc_only=False):
    acc = np.trace(conf_mat)/np.sum(conf_mat)
    if acc_only:
        return acc
    normalized = np.divide(conf_mat.transpose(), np.sum(conf_mat, axis=1)).transpose()*100
    f, ax = plt.subplots(1, 2, figsize=(10,5))
    sns.heatmap(conf_mat, annot=True, fmt=".1f", linewidths=.5, ax=ax[0])
    ax[0].set_title(title)
    ax[0].set_xlabel('Predicted Label')
    ax[0].set_ylabel('True Label')
    tick_labels = list(vc.index)
    xticks = np.arange(len(tick_labels)) + 0.5
    yticks = xticks
    ax[0].set_xticks(xticks, tick_labels)
    ax[0].set_xticklabels(tick_labels, rotation=90)
    ax[0].set_yticks(yticks, tick_labels)
    ax[0].set_yticklabels(tick_labels, rotation=0)
    sns.heatmap(normalized, annot=True, fmt=".1f", linewidths=.5, ax=ax[1])
    ax[1].set_title('Normalized ' + title + ' | Acc: {}'.format(str(acc)[:5]))
    ax[1].set_xlabel('Predicted Label')
    tick_labels = list(vc.index)
    xticks = np.arange(len(tick_labels)) + 0.5
    yticks = xticks
    ax[1].set_xticks(xticks, tick_labels)
    ax[1].set_xticklabels(tick_labels, rotation=90)
    ax[1].set_yticks(yticks, tick_labels)
    ax[1].set_yticklabels(tick_labels, rotation=0)
    f.tight_layout()
    plt.show()
    return acc

def assess_task1(data_dict, network_dict):
    ep_dict = {}
    df_tmp = data_dict['df_train'].copy().reset_index(drop=True)
    for i, k in enumerate(data_dict['vc_train'].index):
        idx = df_tmp[df_tmp['antigen.epitope'] == k].index[0]
        tmp = data_dict['y_enc_train'][idx:idx+1]
        ep_dict[i] = tmp

    res_tmp = {}
    for k, v in ep_dict.items():
        ep_tmp = np.tile(v, (data_dict['X_test'].shape[0],1,1))
        nll_out = network_dict['NLL'].predict([data_dict['X_test'], ep_tmp])
        res_tmp[k] = nll_out

    to_stack = []
    for v in res_tmp.values():
        to_stack.append(v)

    stack = np.vstack(to_stack)
    max_likelihood_out = np.argmin(stack, axis=0)

    valid_idx = np.where(data_dict['y_cls_test'] != -1)[0]

    max_l_out_3 = max_likelihood_out[valid_idx]
    cls_test_3 = data_dict['y_cls_test'][valid_idx]

    max_out = confusion_matrix(max_l_out_3, cls_test_3).transpose()
    acc = viz_conf_mat(max_out, data_dict['vc_train'], 'Confusion Matrix', False)
    return acc

def assess_task2(data_dict, network_dict):
    new_dict = {}
    n_classes = data_dict['vc_test'].shape[0]
    roc_dict = {}
    
    ep_dict = {}
    df_tmp = data_dict['df_train'].copy().reset_index(drop=True)
    for i, k in enumerate(data_dict['vc_train'].index):
        idx = df_tmp[df_tmp['antigen.epitope'] == k].index[0]
        tmp = data_dict['y_enc_train'][idx:idx+1]
        ep_dict[i] = tmp
    
    res_tmp = {}
    for k, v in ep_dict.items():
        ep_tmp = np.tile(v, (data_dict['X_test'].shape[0],1,1))
        nll_out = network_dict['NLL'].predict([data_dict['X_test'], ep_tmp])
        res_tmp[k] = nll_out

    to_stack = []
    for v in res_tmp.values():
        to_stack.append(v)

    stack = np.vstack(to_stack)
    
    df_neg_test = data_dict['df_neg_test']
    
    for k in range(n_classes):
        ep_tmp = np.tile(ep_dict[k], (df_neg_test['cdr3'].shape[0],1,1))
        X_neg = dp.encode_seq_array(df_neg_test['cdr3'], aa_vec, True)
        nll_out = -1*network_dict['NLL'].predict([X_neg, ep_tmp])
        neg_score = list(nll_out)
        pos_score = []
        for i, elem in enumerate(data_dict['y_cls_test']):
            if elem == k:
                pos_score.append(-1*stack[elem,i])
        score = np.array(pos_score + neg_score)
        y = [1]*len(pos_score) + [0]*len(neg_score)
        roc_data = roc_curve(y, score, pos_label=1)
        roc_dict[k] = roc_data
        auc_score = auc(roc_data[0], roc_data[1])
        new_dict[data_dict['vc_train'].index[k]] = auc_score
    return new_dict
    #pk.dump(roc_dict, open('n{}_roc_{}.pk'.format(n_classes, j), 'wb'))

def viz_knn(network_dict, train_embed, y_cls_train, test_embed, y_cls_test, knn):
    knn.fit(train_embed, y_test)
    res = knn.predict(test_embed)
    out = confusion_matrix(res, data_dict['y_cls_test']).transpose()