import pandas as pd
import numpy as np
import random
import sys
# from utils import enccodeListBlosumMaxLen, blosum62_20aa
from imblearn.under_sampling import RandomUnderSampler

random.seed(123456)
MAX_PEPTIDE_LENGTH = 11
MAX_TCR_LENGTH = 19
MAX_MHC_LENGTH = 34

def splitDataframeByPosition(df, splits):
    dataframes = []
    index_to_split = len(df) // splits
    start = 0
    end = index_to_split
    for split in range(splits):
        temporary_df = df.iloc[start:end, :]
        dataframes.append(temporary_df)
        start += index_to_split
        end += index_to_split
    return dataframes

def dataRepresentationBlosum62WithMHCb(X_data):
    encoding = blosum62_20aa
    
    m_cdr3 = enccodeListBlosumMaxLen(X_data.CDR3b, encoding, 19)
    m_cdr3 = m_cdr3.reshape(len(m_cdr3), 380)
    
    m_epitope = enccodeListBlosumMaxLen(X_data.epitope, encoding, 11)
    m_epitope = m_epitope.reshape(len(m_epitope), 220)
    
    m_mhc = enccodeListBlosumMaxLen(X_data.MHC, encoding, 34)
    m_mhc = m_mhc.reshape(len(m_mhc), 680)
    
    df_res1 = pd.DataFrame(m_cdr3)
    df_res2 = pd.DataFrame(m_epitope)
    df_res3 = pd.DataFrame(m_mhc)
    
    res = pd.concat([df_res1, df_res2, df_res3], axis=1)
    res.columns=["F"+str(i+1) for i in range(res.shape[1])]
    
    return res

def dataRepresentationBlosum62WithoutMHCb(X_data):
    encoding = blosum62_20aa
    
    m_cdr3 = enccodeListBlosumMaxLen(X_data.CDR3b, encoding, 19)
    m_cdr3 = m_cdr3.reshape(len(m_cdr3), 380)
    
    m_epitope = enccodeListBlosumMaxLen(X_data.epitope, encoding, 11)
    m_epitope = m_epitope.reshape(len(m_epitope), 220)
    
    df_res1 = pd.DataFrame(m_cdr3)
    df_res2 = pd.DataFrame(m_epitope)
    
    res = pd.concat([df_res1, df_res2], axis=1)
    
    res.columns=["F"+str(i+1) for i in range(res.shape[1])]
    
    return res

def dataRepresentationBlosum62WithoutMHCa(X_data):
    encoding = blosum62_20aa
    
    m_cdr3 = enccodeListBlosumMaxLen(X_data.CDR3a, encoding, 19)
    m_cdr3 = m_cdr3.reshape(len(m_cdr3), 380)
    
    m_epitope = enccodeListBlosumMaxLen(X_data.epitope, encoding, 11)
    m_epitope = m_epitope.reshape(len(m_epitope), 220)
    
    df_res1 = pd.DataFrame(m_cdr3)
    df_res2 = pd.DataFrame(m_epitope)
    
    res = pd.concat([df_res1, df_res2], axis=1)
    
    res.columns=["F"+str(i+1) for i in range(res.shape[1])]
        
    return res

def dataRepresentationBlosum62WithMHCa(X_data):
    encoding = blosum62_20aa
    
    m_cdr3 = enccodeListBlosumMaxLen(X_data.CDR3a, encoding, 19)
    m_cdr3 = m_cdr3.reshape(len(m_cdr3), 380)
    
    m_epitope = enccodeListBlosumMaxLen(X_data.epitope, encoding, 11)
    m_epitope = m_epitope.reshape(len(m_epitope), 220)
    
    m_mhc = enccodeListBlosumMaxLen(X_data.MHC, encoding, 34)
    m_mhc = m_mhc.reshape(len(m_mhc), 680)
    
    df_res1 = pd.DataFrame(m_cdr3)
    df_res2 = pd.DataFrame(m_epitope)
    df_res3 = pd.DataFrame(m_mhc)
    
    res = pd.concat([df_res1, df_res2, df_res3], axis=1)
    res.columns=["F"+str(i+1) for i in range(res.shape[1])]

    return res

def checkLengthEpitope(df):
    df["len_epitope"] = df.epitope.str.len()
    df = df[(df["len_epitope"] <= 11) & (df["len_epitope"] >= 8)]
    df = df.drop(['len_epitope'], axis=1)
    discard = ["\*", '_', '-', 'O', '1', 'y', 'l', 'X', '/', ' ']
    df = df[~df.epitope.str.contains('|'.join(discard))]
    df = df.reset_index(drop=True)
    return df

def checkLengthTCR(df):
    df["len_cdr3"] = df.CDR3b.str.len()
    df = df[(df["len_cdr3"] <= 19) & (df["len_cdr3"] >= 8)]
    df = df.drop(['len_cdr3'], axis=1)
    discard = ["\*", '_', '-', 'O', '1', 'y', 'l', 'X', '/', ' ']
    df = df[~df.CDR3a.str.contains('|'.join(discard))]
    df = df.reset_index(drop=True)
    return df

def checkLengthFulla(df):
    df["len_epitope"] = df.epitope.str.len()
    df = df[(df["len_epitope"] <= 11) & (df["len_epitope"] >= 8)]
    df["len_cdr3"] = df.CDR3a.str.len()
    df = df[(df["len_cdr3"] <= 19) & (df["len_cdr3"] >= 8)]
    df = df.drop(['len_epitope', 'len_cdr3'], axis=1)
    df = df.reset_index(drop=True)
    
    discard = ["\*", '_', '-', 'O', '1', 'y', 'l', 'X', '/', ' ', '#', '\(', '\?']
    df = df[~df.CDR3a.str.contains('|'.join(discard))]
    df = df[~df.epitope.str.contains('|'.join(discard))]
    return df

def checkLengthFullb(df):
    df["len_epitope"] = df.epitope.str.len()
    df = df[(df["len_epitope"] <= 11) & (df["len_epitope"] >= 8)]
    df["len_cdr3"] = df.CDR3b.str.len()
    df = df[(df["len_cdr3"] <= 19) & (df["len_cdr3"] >= 8)]
    df = df.drop(['len_epitope', 'len_cdr3'], axis=1)
    df = df.reset_index(drop=True)
    
    discard = ["\*", '_', '-', 'O', '1', 'y', 'l', 'X', '/', ' ', '#', '\(', '\?']
    df = df[~df.CDR3b.str.contains('|'.join(discard))]
    df = df[~df.epitope.str.contains('|'.join(discard))]
    return df

#def dataRepresentationDownsamplingWithoutMHCb(data):
#    X_train, y_train = data[["CDR3b", "epitope"]], data[["binder"]]

#    nm = RandomUnderSampler(random_state=42)
#    X_res, y_res = nm.fit_resample(X_train, y_train)
    
#    pX_res = dataRepresentationBlosum62WithoutMHCb(X_res)
#    py_res = y_res.copy()
    
 #   return pX_res, py_res



def dataRepresentationDownsamplingWithoutMHCb(data):
    X_train, y_train = data[["CDR3b", "epitope"]], data[["binder"]]
    
    pX_train = dataRepresentationBlosum62WithoutMHCb(X_train)
    py_train = y_train.copy()
    
    return pX_train, py_train



def dataRepresentationDownsamplingWithMHCb(data):
    X_train, y_train = data[["CDR3b", "epitope", "MHC"]], data[["binder"]]

    nm = RandomUnderSampler(random_state=42)
    X_res, y_res = nm.fit_resample(X_train, y_train)
    
    pX_res = dataRepresentationBlosum62WithMHCb(X_res)
    py_res = y_res.copy()
    
    return pX_res, py_res

aas = pd.DataFrame(np.identity(20))
aas.index = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
blank = pd.DataFrame([np.zeros(shape=(20,))])
blank.index = ["-"]
code_dict = pd.concat([aas, blank], axis=0)

def encodeEpitope(record):
    epitope = record["epitope"] 
    epitope_onehot = pd.Series(code_dict.loc[list(epitope)].values.flatten())   
    
    return epitope_onehot

def encodeCDR3b(record):
    CDR3 = record["CDR3b"] 
    CDR3_onehot = pd.Series(code_dict.loc[list(CDR3)].values.flatten())   
    
    return CDR3_onehot

def encodeCDR3a(record):
    CDR3 = record["CDR3a"] 
    CDR3_onehot = pd.Series(code_dict.loc[list(CDR3)].values.flatten())   
    
    return CDR3_onehot

def encodeMHC(record):
    MHC = record["MHC"] 
    MHC_onehot = pd.Series(code_dict.loc[list(MHC)].values.flatten())   
    
    return MHC_onehot

chunk_size = 5

def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

def subtractStringLists(lst1, lst2):
    res = [ ele for ele in lst1]
    for a in lst2:
        if a in lst1:
            res.remove(a)
    return res



blosum62_20aa = {
        'A': np.array(( 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0)),
        'R': np.array((-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3 )),
        'N': np.array((-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3)),
        'D': np.array((-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3)),
        'C': np.array(( 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1)),
        'Q': np.array((-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2)),
        'E': np.array((-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2)),
        'G': np.array(( 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3)),
        'H': np.array((-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3)),
        'I': np.array((-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3)),
        'L': np.array((-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1)),
        'K': np.array((-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2)),
        'M': np.array((-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1)),
        'F': np.array((-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1)),
        'P': np.array((-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2)),
        'S': np.array(( 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2)),
        'T': np.array(( 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0)),
        'W': np.array((-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3)),
        'Y': np.array((-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1)),
        'V': np.array(( 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4))
    }


def enccodeListBlosumMaxLen(aa_seqs, blosum, max_seq_len):
    sequences=[]
    for seq in aa_seqs:
        e_seq=np.zeros((len(seq),len(blosum["A"])))
        count=0
        for aa in seq:
            if aa in blosum:
                e_seq[count]=blosum[aa]
                count+=1
            else:
                sys.stderr.write("Unknown amino acid in peptides: "+ aa +", encoding aborted!\n")
                sys.exit(2)
                
        sequences.append(e_seq)
    n_seqs = len(aa_seqs)
    n_features = sequences[0].shape[1]

    enc_aa_seq = np.zeros((n_seqs, max_seq_len, n_features))
    for i in range(0,n_seqs):
        enc_aa_seq[i, :sequences[i].shape[0], :n_features] = sequences[i]

    return enc_aa_seq
