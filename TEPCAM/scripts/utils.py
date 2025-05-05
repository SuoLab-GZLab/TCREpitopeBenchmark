import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F 

def BLOSUM_embedding(filename):
    '''
    read in BLOSUM matrix

    parameters:
        - filename : file containing BLOSUM matrix

    returns:
        - blosum embedding matrix: list
    '''
    if filename is None or filename.lower() == 'none':
        filename = './blosum/BLOSUM62'
    
    embedding_file = open(filename, "r")
    lines = embedding_file.readlines()[7:]
    embedding_file.close()

    embedding = [[float(x) for x in l.strip().split()[1:]] for l in lines]
    embedding.append([0.0] * len(embedding[0]))
    
    return embedding

def get_numbering(seqs, ):
    """
    get the IMGT numbering of CDR3 with ANARCI tool
    modified from https://github.com/pengxingang/TEIM/blob/main/scripts/data_process.py
    """
    random_int = random.randint(0,1000000)
    template = ['GVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGTTDQGEVPNGYNVSRSTIEDFPLRLLSAAPSQTSVYF', 'GEGSRLTVL']
    # # save fake tcr file
    save_path = f"tmp_faketcr{random_int}.fasta"
    id_list = []
    seqs_uni = np.unique(seqs)
    print(f"Number of unique sequence:{len(seqs_uni)}")
    with open(save_path, 'w+') as f:
        for i, seq in enumerate(seqs_uni):
            f.write('>'+str(i)+'\n')
            id_list.append(i)
            total_seq = ''.join([template[0], seq ,template[1]])
            f.write(str(total_seq))
            f.write('\n')
    print('Save fasta file to '+save_path + '\n Aligning...')
    df_seqs = pd.DataFrame(list(zip(id_list, seqs_uni)), columns=['Id', 'cdr3'])
    output_file = f"tmp_align{random_int}"
    cmd = f"ANARCI -i {save_path} -o {output_file} --csv -p 10" #please set Number of parallel processes (-p) that fit your device
    # cmd = (cmd)
    res = os.system(cmd)
    try:
        df = pd.read_csv(f'tmp_align{random_int}_B.csv')
    except FileNotFoundError:
        raise FileNotFoundError('Error: ANARCI failed to align')  
    cols = ['104', '105', '106', '107', '108', '109', '110', '111', '111A', '111B', '112C', '112B', '112A', '112', '113', '114', '115', '116', '117', '118']
    seqs_al = []
    for col in cols:
        if col in df.columns:
            seqs_al_curr = df[col].values
            seqs_al.append(seqs_al_curr)
        else:
            seqs_al_curr = np.full([len(df)], '-')
            seqs_al.append(seqs_al_curr)
    seqs_al = [''.join(seq) for seq in np.array(seqs_al).T]
    df_al = df[['Id']]
    df_al['cdr3_align'] = seqs_al
    ## merge
    os.remove(f'tmp_align{random_int}_B.csv')
    os.remove(f'tmp_faketcr{random_int}.fasta')
    df = df_seqs.merge(df_al, how='inner', on='Id')
    df = df.set_index('cdr3')
    return df.loc[seqs, 'cdr3_align'].values


#Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
#https://arxiv.org/abs/1908.08681v1
#implemented for PyTorch / FastAI by lessw2020 
#github: https://github.com/lessw2020/mish

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))

#Swish activation function
class Swish(nn.Module):
    def __init__(
        self,
    ):
        """
        Init method.
        """
        super(Swish, self).__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return input * torch.sigmoid(input)