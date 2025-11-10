import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,KFold
from tqdm import tqdm
import copy
from collections import defaultdict
import Levenshtein
from tcrpeg.TCRpeg import TCRpeg
from model import TEINet
import torch

def split(path,record_dic,fold=5,sep=','):
    '''
    Split the data to k-fold
    @path: path to the file
    @record_dic: the directory to store the processed k fold data
    @sep: seperator for the file type (csv:','; tsv:'\t')
    '''

    data = pd.read_csv(path,sep=sep)    
    kf = KFold(n_splits=fold,shuffle=True,random_state=42)
    i= 1
    for train_idx, test_idx in kf.split(data):
        df_train,df_test = data.iloc[train_idx].reset_index(drop=True),data.iloc[test_idx].reset_index(drop=True)
        df_train.to_csv('{}/train_{}_positive.csv'.format(record_dic,i),index=False)
        df_test.to_csv('{}/test_{}_positive.csv'.format(record_dic,i),index=False)
        i += 1

def epitope_sample_1fold(positive_file,record_file,sample_num=1,fre=True):
    '''
    Negative sampling. For each TCR, sample a epitope as its negative. 
    The output file contains both positive (input) and negative pairs

    @positive_file: path to the file that records the positive pairs
    @record_file: path to the output file
    @sample_num: for each TCR, the number of sampled epitopes
    @fre: epitope sampled randomly (False) or based on their frequency distribution (True)
    '''
    data = pd.read_csv(positive_file)
    cdrs,epitopes = data['CDR3.beta'].values, data['Epitope'].values
    cdrs_new,epitopes_new = [],[]
    cdrs_original,epitopes_original = [],[]
    t2e = defaultdict(set)
    for i in range(len(cdrs)):
        if len(epitopes[i]) <= 15:
            t2e[cdrs[i]].add(epitopes[i])
    for i in tqdm(range(len(cdrs))):
        tcr,e = cdrs[i],epitopes[i]
        cdrs_original.append(tcr)
        epitopes_original.append(e)
        to_remove = t2e[tcr] 
        e2f = defaultdict(int)
        sum_ = 0
        for ee in epitopes: #
            if ee not in to_remove:                
                e2f[ee] += 1
                sum_ += 1
        if not fre:
            sample = np.random.choice(np.array(list(e2f.keys())),sample_num,replace=False)                            
        else :
            sample = np.random.choice(np.array(list(e2f.keys())),sample_num,replace=False,p = [e2f[x]/sum_ for x in e2f.keys()])
        epitopes_new.extend(sample)
        cdrs_new.extend([tcr] * sample_num)
    processed_data = pd.DataFrame(columns=['CDR3.beta','Epitope','Label'])            
    processed_data['CDR3.beta'] = cdrs_original + cdrs_new            
    processed_data['Epitope'] = epitopes_original + epitopes_new
    processed_data['Label'] = [1] * len(cdrs_original) + [0]*len(cdrs_new)
    processed_data.to_csv(record_file,index=False)

def tcr_sample_1fold(positive_file,record_file,sample_num=1,reference_file='None'):
    '''
    Negative sampling. For each epitope, sample a tcr as its negative. 
    The output file contains both positive (input) and negative pairs

    @positive_file: path to the file that records the positive pairs
    @record_file: path to the output file
    @sample_num: for each epitope, the number of sampled tcrs
    @reference_file: Path to reference TCR; If specified, will sample TCR from the reference data
    '''
    data = pd.read_csv(positive_file)
    cdrs,epitopes = data['CDR3.beta'].values, data['Epitope'].values
    cdrs_new,epitopes_new = [],[]
    cdrs_original,epitopes_original = [],[]
    reference=False
    if reference_file != 'None':
        refer_tcr = pd.read_csv(reference_file)['CDR3.beta'].values
        reference=True
    # epitope_reference = set(epitopes)
    e2t = defaultdict(set)
    for i in range(len(cdrs)):
        if len(epitopes[i]) <= 15:
            e2t[epitopes[i]].add(cdrs[i])
    for i in tqdm(range(len(cdrs))):
        tcr,e = cdrs[i],epitopes[i]
        cdrs_original.append(tcr)
        epitopes_original.append(e)
        to_remove = e2t[e] 
        to_sample = set()
        for t in cdrs:
            if t not in to_remove:
                to_sample.add(t)                                               
        if reference:
            sample = np.random.choice(refer_tcr,sample_num,replace=False)
        else :
            sample = np.random.choice(np.array(list(to_sample)),sample_num,replace=False)
        epitopes_new.extend([e] * sample_num)
        cdrs_new.extend(sample)
    processed_data = pd.DataFrame(columns=['CDR3.beta','Epitope','Label'])            
    processed_data['CDR3.beta'] = cdrs_original + cdrs_new            
    processed_data['Epitope'] = epitopes_original + epitopes_new
    processed_data['Label'] = [1] * len(cdrs_original) + [0]*len(cdrs_new)
    processed_data.to_csv(record_file,index=False)

def compute_metric(pos_data,predictions,k=3):
    '''
    Compute Precision, Recall, and NDCG
    @pos_data: dict, key is [tcr]->[e1,e2,...]
    @predictions: dict, key is [tcr] -> [(e1,scores1),(e2,scores2), .... ] 
    '''
    recalls,precisions = [],[]
    ndcgs = []
    for tcr in predictions.keys():
        pres, trues = predictions[tcr], set(pos_data[tcr])
        pres.sort(key=lambda x: -x[1])
        pres = [p[0] for p in pres]
        count = 0
        dcp = 0
        for i,p in enumerate(pres[:k]):
            if p in trues:
                count += 1
                dcp += 1 / np.log2(i+1 + 1)
        idcp = sum([1/np.log2(i+1+1) for i in range(min(len(trues),k))])
        ndcg = dcp / idcp if idcp > 0 else 0
        ndcgs.append(ndcg)
        precisions.append(count / k)
        recalls.append(count / len(trues))
    print('Precision, Recall, NDCG at top {} are:'.format(k))
    print(str(np.mean(precisions)) + ', ',str(np.mean(recalls)) + ', '+str(np.mean(ndcgs)))
    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)
    

def levenstein_filter(path_ref,path_to_filter, threshold,record_path=None):
    '''
    Filter TCRs based on levenstein distance
    @path_ref: path to the reference data (namely the training set)
    @path_to_filter: path to the data that needs filtering (namely the test set)
    @threshold: filter threshold (0 to 1)
    @record_path: if specified, will store the filtered data
    '''
    cdrs_ref = pd.read_csv(path_ref)['CDR3.beta'].values
    data_filter = pd.read_csv(path_to_filter)
    cdrs_to_filter = data_filter['CDR3.beta'].values
    to_filter = []
    for i in tqdm(range(len(cdrs_to_filter))):
        for cdr_ref in cdrs_ref:
            if Levenshtein.ratio(cdr_ref, cdrs_to_filter[i]) >= threshold:            
                to_filter.append(i)
                break
    data = data_filter.drop(to_filter)
    print('The original dataset has {} pairs'.format(len(cdrs_to_filter)))
    print('After filtering there are {} pairs'.format(len(data)))
    if record_path is not None:
        data.to_csv(record_path,index=False)
        
def aa_scan(seq,scan_aa='A',scan_aa_alternate='G',offset = 0):
    seq = list(seq)
    return_seq = []
    for i in range(offset,len(seq)-offset):
        s_ = copy.copy(seq)
        s_[i] = scan_aa
        if s_ != seq:
            return_seq.append(''.join(s_))
        else :
            s_[i] = scan_aa_alternate
            return_seq.append(''.join(s_))
    return return_seq

def contact_index(dis,axis = 0,threshold = 5,offset=1):
    #original is 5 threshold
    #start from 0
    assert axis in [0,1]
    c_index = []
    for i in range(offset,dis.shape[axis]-offset):
        compare = dis[i] if axis == 0 else dis[:,i]
        if sum(compare <= threshold) > 0:
            c_index.append(i)    
    return c_index

def mean_dis(dis,axis = 0,threshold = 5,offset=1):
    #original is 5 threshold
    #start from 0
    assert axis in [0,1]
    mean_distance = []
    for i in range(offset,dis.shape[axis]-offset):
        compare = dis[i] if axis == 0 else dis[:,i]
        # if sum(compare <= threshold) > 0:
        #     c_index.append(i)
        mean_distance.append(np.mean(compare))    
    return mean_distance

def load_teinet(path,emb_path1='encoders/aa_emb_tcr.txt',emb_path2 = 'encoders/aa_emb_tcr.txt',device='cuda:0',normalize=True,weight_decay = 0):
    model_tcr = TCRpeg(hidden_size=768,num_layers = 3,load_data=False,embedding_path=emb_path1,device=device)
    model_tcr.create_model()
    model_epi = TCRpeg(hidden_size=768,num_layers = 3,load_data=False,embedding_path=emb_path2,device=device)
    model_epi.create_model()
    model = TEINet(en_tcr=model_tcr,en_epi = model_epi,cat_size=768*2,dropout=0.1,normalize=True,weight_decay = 0,device=device)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    return model
