import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import os, datetime, sys, pickle, math, random
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict


def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if seed==0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _init_fn(worker_id): 
    random.seed(10 + worker_id)
    np.random.seed(10 + worker_id)
    torch.manual_seed(10 + worker_id)
    torch.cuda.manual_seed(10 + worker_id)
    torch.cuda.manual_seed_all(10 + worker_id)


class MyDataset(Dataset):
    def __init__(self, data, labl, transform=None):
        self.data = data 
        self.labl = labl
        self.transform = transform        
    def __getitem__(self, idx):
        feature = self.data[idx, :]
        label = self.labl[idx, :]
        if self.transform:
            label = self.transform(label)
            feature = self.transform(feature)
        return feature, label
    def __len__(self):
        return len(self.labl)


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1, keepdims = True) 
    s = x_exp / x_sum    
    return s


def Labels_Initialization(num_classes, labels):
    initialized_labels = torch.IntTensor([])
    for idx in range(len(labels)):
        ones = torch.eye(num_classes)
        tmp = ones.index_select(0, torch.IntTensor([labels[idx]]))
        initialized_labels = torch.cat((initialized_labels, tmp), dim=0)
    return initialized_labels


def train_step(model, features, labels):    
    model.train()    
    model.optimizer.zero_grad()
    predictions = model(features)
    loss = model.loss_func(predictions, labels) # torch type
    pred = softmax(predictions.cpu().detach().numpy()) # numpy type
    labels = labels.cpu().numpy()
    acc = model.acc_func(y_true=labels, y_pred=pred.argmax(1)) # numpy type 
    auc_roc = model.auc_roc_func(y_true=labels, y_score=pred[:,1]) # numpy type
    auc_pr = model.auc_pr_func(y_true=labels, y_score=pred[:,1]) # numpy type
    F1_score = model.f1_score_func(y_true=labels, y_pred=pred.argmax(1)) # numpy type
    loss.backward()
    model.optimizer.step()
    return loss.item(), acc, auc_roc, auc_pr, F1_score


def valid_step(model, features, labels):
    model.eval()
    with torch.no_grad():
        predictions = model(features)
        loss = model.loss_func(predictions, labels) # torch type
        pred = softmax(predictions.cpu().detach().numpy()) # numpy type
        labels = labels.cpu().numpy()
        acc = model.acc_func(y_true=labels, y_pred=pred.argmax(1)) # numpy type 
        auc_roc = model.auc_roc_func(y_true=labels, y_score=pred[:,1]) # numpy type
        auc_pr = model.auc_pr_func(y_true=labels, y_score=pred[:,1]) # numpy type
        F1_score = model.f1_score_func(y_true=labels, y_pred=pred.argmax(1)) # numpy type
    return loss.item(), acc, auc_roc, auc_pr, F1_score


def train_model(model, epochs, dl_train, device):
    dfhistory = pd.DataFrame(columns=['epoch', 'loss', 'acc', 'auc_roc', 'auc_pr', 'f1_score'])
    # dfhistory_eval = pd.DataFrame(columns=['epoch', 'loss', 'acc', 'auc_roc', 'auc_pr', 'f1_score'])
    for epoch in range(1,epochs+1):
        starttime = datetime.datetime.now()    
        loss_sum = 0.0
        acc_sum = 0.0
        auc_roc_sum = 0.0
        auc_pr_sum = 0.0
        F1_score_sum= 0.0
        step = 1
        val_step = 1
        for step, (features, labels) in enumerate(dl_train, 1):
            features, labels = features.to(device), labels.to(device)
            features = features.float()
            labels = labels[:, 1].long()
            loss, acc, auc_roc, auc_pr, F1_score = train_step(model, features=features, labels=labels)
            loss_sum += loss
            acc_sum += acc
            auc_roc_sum += auc_roc
            auc_pr_sum += auc_pr
            F1_score_sum += F1_score


        # #----------------------Valid cycle----------------------
        # print("Start validating ...")
        # for _,(features_valid, labels_valid) in enumerate(dl_valid):
        #     features_valid, labels_valid = features_valid.to(device), labels_valid.to(device)
        #     features_valid = features_valid.float()
        #     labels_valid = labels_valid[:, 1].long()
        #     val_loss, val_acc, val_auc_roc, val_auc_pr, val_F1_score = valid_step(model, features=features_valid, labels=labels_valid)


        #----------------------Recording----------------------
        print("Recording epoch-level info ...")    
        # Train-related metrics
        loss_avrg = loss_sum/step
        acc_avrg = acc_sum/step
        auc_roc_avrg = auc_roc_sum/step
        auc_pr_avrg = auc_pr_sum/step
        F1_score_avrg = F1_score_sum/step
        # # Valid-related metrics
        # val_loss_avrg = val_loss
        # val_acc_avrg = val_acc
        # val_auc_roc_avrg = val_auc_roc
        # val_auc_pr_avrg = val_auc_pr
        # val_F1_score_avrg = val_F1_score

        # Epoch-level information
        info = (epoch, loss_avrg, acc_avrg, auc_roc_avrg, auc_pr_avrg, F1_score_avrg)
        # info_eval = (epoch, val_loss_avrg, val_acc_avrg, val_auc_roc_avrg, val_auc_pr_avrg, val_F1_score_avrg)
        dfhistory.loc[epoch-1] = info
        # dfhistory_eval.loc[epoch-1] = info_eval
        # dfhistory.to_csv("dfhistory_tmp", header=True, sep='\t', index=False)
        # torch.save(model.state_dict(), trained_model_dir + str(epoch) + '_epochs.pt')
        
    #torch.save(model.state_dict(), "{}/{}_epochs.pt".format(trained_model_dir, str(epoch)))
        #torch.save(model.state_dict(),trained_model_dir)
        endtime = datetime.datetime.now()  
        totaltime=endtime-starttime
        print("Epoch: {}      totaltime: {}".format(epoch, totaltime))
    return dfhistory #, dfhistory_eval




def eval_model(model_initial, save_model_path, epochs, fold, dl_valid, repeat, device):
    # Initialize dataframe to store final results
    dfhistory = pd.DataFrame(columns=['CDR3B', 'Epitope', 'labels_valid', 'y_prob', 'y_pred'])

    # Load pre-processed data
    cdr3b_list = []  # Placeholder for storing CDR3B sequences
    epitope_list = []  # Placeholder for storing Epitope sequences

    for epoch in range(1, epochs + 1):
        starttime = datetime.datetime.now()
        model = model_initial

        # Load the saved model state
        checkpoint = torch.load(save_model_path, map_location=device)
        model.load_state_dict(checkpoint)

        all_labels_valid = []
        all_y_prob = []
        all_y_pred = []

        # Iterate over the validation DataLoader
        for _, (features_valid, labels_valid) in enumerate(dl_valid):
            features_valid, labels_valid = features_valid.to(device), labels_valid.to(device)
            features_valid = features_valid.float()
            labels_valid = labels_valid[:, 1].long()

            model.eval()
            with torch.no_grad():
                y_prob = model(features_valid)
                y_prob = torch.sigmoid(y_prob)  # Convert logits to probabilities

            y_pred = (y_prob >= 0.5).long()

            all_labels_valid.extend(labels_valid.cpu().numpy())
            all_y_prob.extend(y_prob.cpu().numpy())
            all_y_pred.extend(y_pred.cpu().numpy())

            # Retrieve original sequences (only once, assuming consistent order)
            if not cdr3b_list and not epitope_list:
                # Assuming features_valid stores CDR3B and Epitope as part of preprocessed features
                cdr3b_list.extend(dl_valid.dataset.cdr3_sequences)  # Replace with actual attribute storing CDR3B
                epitope_list.extend(dl_valid.dataset.epitope_sequences)  # Replace with actual attribute storing Epitope

        # Create a DataFrame for the current epoch
        epoch_results = pd.DataFrame({
            'CDR3B': cdr3b_list,
            'Epitope': epitope_list,
            'labels_valid': all_labels_valid,
            'y_prob': [prob[1] for prob in all_y_prob],  # Extract probability of class 1
            'y_pred': all_y_pred
        })

        # Save epoch results to a cumulative DataFrame
        dfhistory = pd.concat([dfhistory, epoch_results], ignore_index=True)

        # Print progress and runtime
        endtime = datetime.datetime.now()
        totaltime = endtime - starttime
        print(f"Epoch: {epoch}      Total time: {totaltime}")
    return dfhistory



def eval_model_downsample(model_initial, epochs, fold, dl_valid, repeat, device, percent):
    dfhistory = pd.DataFrame(columns=['epoch', 'loss', 'acc', 'auc_roc', 'auc_pr', 'f1_score'])
    for epoch in range(1, epochs+1):
        starttime = datetime.datetime.now()
        model = model_initial
        params = "../model_repeat{}_downsample{}/fold{}/{}_epochs.pt".format(repeat, percent, fold, str(int(epoch)))
    
        checkpoint = torch.load(params, map_location='cpu')
        model.load_state_dict(checkpoint)

        for _,(features_valid, labels_valid) in enumerate(dl_valid):
            features_valid, labels_valid = features_valid.to(device), labels_valid.to(device)
            features_valid = features_valid.float()
            labels_valid = labels_valid[:, 1].long()
            val_loss, val_acc, val_auc_roc, val_auc_pr, val_F1_score = valid_step(model, features=features_valid, labels=labels_valid)

        info = (epoch, val_loss, val_acc, val_auc_roc, val_auc_pr, val_F1_score)
        dfhistory.loc[epoch-1] = info
        dfhistory.to_csv("dfhistory_eval_tmp", header=True, sep='\t', index=False)
        endtime = datetime.datetime.now()
        totaltime = endtime-starttime
        print("Epoch: {}      totaltime: {}".format(epoch, totaltime))
    return dfhistory


def pr_auc_score(y_true, y_score):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true=y_true, probas_pred=y_score)
    aupr = metrics.auc(x=recall, y=precision)
    return aupr
