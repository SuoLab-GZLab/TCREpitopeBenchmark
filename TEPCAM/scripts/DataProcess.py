import os
import pandas as pd
import numpy as np
import argparse
import warnings
import torch
from torch.utils.data import Dataset
from scripts.utils import get_numbering
warnings.filterwarnings("ignore")

AMINO_ACIDS = "RHKDESTNQCUGPAVILMFYW" #21 amino acid
PAD = "-"
MASK = "."
UNK = "?"
SEP = "|"
CLS = "*"
AA_List = AMINO_ACIDS + PAD + MASK + UNK + SEP + CLS


class TEPDataset(Dataset):
    def __init__(self, sequence, labels, max_tcr_len=20, max_antigen_len=15,align=False):
        self.max_tcr_len = max_tcr_len
        self.max_antigen_len = max_antigen_len
        self.data = self.__initilize_data(sequence, labels, align)
        
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    
    def __initilize_data(self, sequences, labels, align):
        data_list = []
        if align:
            #use ANARCI for alignment
            TCR = sequences['CDR3B']
            TCR_align = get_numbering(TCR)
            sequences.loc[:,'CDR3B'] = TCR_align
        sequences = sequences.values
        labels = labels.values.astype('int')
        for index,sequence in enumerate(sequences):
            tcr, pep = sequence
            tcr_tsfed = self.seq_transform(tcr,self.max_tcr_len)
            pep_tsfed = self.seq_transform(pep,self.max_antigen_len)
            data_list.append((tcr_tsfed,pep_tsfed,labels[index],tcr,pep))
        return data_list
    
    def seq_transform(self, sequence, max_len):
        #Sequence padding
        sequence = sequence[:max_len] if len(sequence) >= max_len else sequence + '-' * (max_len-len(sequence))
        try:
            data_list = [AA_List.index(aa) for aa in sequence]
        except:
            print(f"Invalid AA in seq:{sequence}")
        return torch.tensor(data_list)

def dropInvalid(df_x,df_y):
    df_x.drop(df_x[df_x['CDR3B'].str.contains('-')].index, inplace=True)
    df_y.drop(df_x[df_x['CDR3B'].str.contains('-')].index, inplace=True)
    return df_x,df_y
