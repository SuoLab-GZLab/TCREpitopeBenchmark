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
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _init_fn(worker_id): 
    random.seed(10 + worker_id)
    np.random.seed(10 + worker_id)
    torch.manual_seed(10 + worker_id)
    torch.cuda.manual_seed(10 + worker_id)
    torch.cuda.manual_seed_all(10 + worker_id)


class MyDataset(Dataset):
    def __init__(self, data, labl, raw_cdr3=None, raw_epitope=None, transform=None):
        self.data = data 
        self.labl = labl
        self.raw_cdr3 = raw_cdr3  
        self.raw_epitope = raw_epitope  
        self.transform = transform
        
    def __getitem__(self, idx):
        feature = self.data[idx, :]
        label = self.labl[idx, :]
        cdr3_seq = self.raw_cdr3[idx] if self.raw_cdr3 is not None else None
        epitope_seq = self.raw_epitope[idx] if self.raw_epitope is not None else None
        if self.transform:
            label = self.transform(label)
            feature = self.transform(feature)
        return feature, label, cdr3_seq, epitope_seq
        
    def __len__(self):
        return len(self.labl)


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
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
    loss = model.loss_func(predictions, labels)  # torch type
    pred = softmax(predictions.cpu().detach().numpy())  # numpy type
    labels = labels.cpu().numpy()
    acc = model.acc_func(y_true=labels, y_pred=pred.argmax(1))  # numpy type 
    auc_roc = model.auc_roc_func(y_true=labels, y_score=pred[:, 1])  # numpy type
    auc_pr = model.auc_pr_func(y_true=labels, y_score=pred[:, 1])  # numpy type
    F1_score = model.f1_score_func(y_true=labels, y_pred=pred.argmax(1))  # numpy type
    loss.backward()
    model.optimizer.step()
    return loss.item(), acc, auc_roc, auc_pr, F1_score


def valid_step(model, features, labels):
    model.eval()
    with torch.no_grad():
        predictions = model(features)
        loss = model.loss_func(predictions, labels)  # torch type
        pred = softmax(predictions.cpu().detach().numpy())  # numpy type
        labels = labels.cpu().numpy()
        acc = model.acc_func(y_true=labels, y_pred=pred.argmax(1))  # numpy type 
        auc_roc = model.auc_roc_func(y_true=labels, y_score=pred[:, 1])  # numpy type
        auc_pr = model.auc_pr_func(y_true=labels, y_score=pred[:, 1])  # numpy type
        F1_score = model.f1_score_func(y_true=labels, y_pred=pred.argmax(1))  # numpy type
    return loss.item(), acc, auc_roc, auc_pr, F1_score


def train_model(model, dl_train, device, num_epochs):
    for epoch in range(1, num_epochs + 1): 
        loss_sum, acc_sum, auc_roc_sum, auc_pr_sum, F1_score_sum = 0.0, 0.0, 0.0, 0.0, 0.0
        for step, (features, labels, _, _) in enumerate(dl_train, 1):
            features, labels = features.to(device), labels.to(device)
            features = features.float()
            labels = labels[:, 1].long()
            loss, acc, auc_roc, auc_pr, F1_score = train_step(model, features=features, labels=labels)
            loss_sum += loss
            auc_pr_sum += auc_pr
        print(f"Epoch {epoch}: Loss: {loss_sum / step}, AUC-PR: {auc_pr_sum / step}")



import torch.nn.functional as F
def eval_model(model_initial, dl_valid, device):
    dfhistory = pd.DataFrame(columns=['CDR3B', 'Epitope', 'y_true', 'y_prob', 'y_pred'])
    model = model_initial
    all_labels_valid = []
    all_y_prob = []
    all_y_pred = []
    all_cdr3b = []
    all_epitope = []

    for _, (features_valid, labels_valid, cdr3_seq, epitope_seq) in enumerate(dl_valid):
        features_valid, labels_valid = features_valid.to(device), labels_valid.to(device)
        features_valid = features_valid.float()
        labels_valid = labels_valid[:, 1].long()

        with torch.no_grad():
            logits = model(features_valid)  # Get raw logits
            probabilities = F.softmax(logits, dim=1)  # Apply softmax to get probabilities

        y_prob = probabilities[:, 1]  # Extract probabilities for class 1
        y_pred = (y_prob >= 0.5).long()  # Threshold probabilities to get predictions

        all_labels_valid.extend(labels_valid.cpu().numpy())
        all_y_prob.extend(y_prob.cpu().numpy())  # Store class 1 probabilities
        all_y_pred.extend(y_pred.cpu().numpy())  # Store predicted class labels
        all_cdr3b.extend(cdr3_seq)  # Store original CDR3B sequences
        all_epitope.extend(epitope_seq)  # Store original Epitope sequences

    # Store results for this epoch in a DataFrame
    epoch_results = pd.DataFrame({
        'CDR3B': all_cdr3b,
        'Epitope': all_epitope,
        'y_true': all_labels_valid,
        'y_prob': all_y_prob,  # Probabilities for class 1
        'y_pred': all_y_pred  # Predicted class labels
    })

    dfhistory = pd.concat([dfhistory, epoch_results], ignore_index=True)
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

