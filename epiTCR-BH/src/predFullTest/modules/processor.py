import pandas as pd
import numpy as np
import random
import modules.utils as Utils
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
    encoding = Utils.blosum62_20aa
    
    m_cdr3 = Utils.enccodeListBlosumMaxLen(X_data.CDR3b, encoding, 19)
    m_cdr3 = m_cdr3.reshape(len(m_cdr3), 380)
    
    m_epitope = Utils.enccodeListBlosumMaxLen(X_data.epitope, encoding, 11)
    m_epitope = m_epitope.reshape(len(m_epitope), 220)
    
    m_mhc = Utils.enccodeListBlosumMaxLen(X_data.MHC, encoding, 34)
    m_mhc = m_mhc.reshape(len(m_mhc), 680)
    
    df_res1 = pd.DataFrame(m_cdr3)
    df_res2 = pd.DataFrame(m_epitope)
    df_res3 = pd.DataFrame(m_mhc)
    
    res = pd.concat([df_res1, df_res2, df_res3], axis=1)
    
    res.columns=["F"+str(i+1) for i in range(res.shape[1])]
    
    return res

def dataRepresentationBlosum62WithoutMHCb(X_data):
    encoding = Utils.blosum62_20aa
    
    m_cdr3 = Utils.enccodeListBlosumMaxLen(X_data.CDR3b, encoding, 19)
    m_cdr3 = m_cdr3.reshape(len(m_cdr3), 380)
    
    m_epitope = Utils.enccodeListBlosumMaxLen(X_data.epitope, encoding, 11)
    m_epitope = m_epitope.reshape(len(m_epitope), 220)
    
    df_res1 = pd.DataFrame(m_cdr3)
    df_res2 = pd.DataFrame(m_epitope)
    
    res = pd.concat([df_res1, df_res2], axis=1)
    
    res.columns=["F"+str(i+1) for i in range(res.shape[1])]
    
    return res

def dataRepresentationBlosum62WithoutMHCa(X_data):
    encoding = Utils.blosum62_20aa
    
    m_cdr3 = Utils.enccodeListBlosumMaxLen(X_data.CDR3a, encoding, 19)
    m_cdr3 = m_cdr3.reshape(len(m_cdr3), 380)
    
    m_epitope = Utils.enccodeListBlosumMaxLen(X_data.epitope, encoding, 11)
    m_epitope = m_epitope.reshape(len(m_epitope), 220)
    
    df_res1 = pd.DataFrame(m_cdr3)
    df_res2 = pd.DataFrame(m_epitope)
    
    res = pd.concat([df_res1, df_res2], axis=1)
    
    res.columns=["F"+str(i+1) for i in range(res.shape[1])]
        
    return res

def dataRepresentationBlosum62WithMHCa(X_data):
    encoding = Utils.blosum62_20aa
    
    m_cdr3 = Utils.enccodeListBlosumMaxLen(X_data.CDR3a, encoding, 19)
    m_cdr3 = m_cdr3.reshape(len(m_cdr3), 380)
    
    m_epitope = Utils.enccodeListBlosumMaxLen(X_data.epitope, encoding, 11)
    m_epitope = m_epitope.reshape(len(m_epitope), 220)
    
    m_mhc = Utils.enccodeListBlosumMaxLen(X_data.MHC, encoding, 34)
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

def dataRepresentationDownsamplingWithoutMHCb(data):
    X_train, y_train = data[["CDR3b", "epitope"]], data[["binder"]]

    nm = RandomUnderSampler(random_state=42)
    X_res, y_res = nm.fit_resample(X_train, y_train)
    
    pX_res = dataRepresentationBlosum62WithoutMHCb(X_res)
    py_res = y_res.copy()
    
    return pX_res, py_res

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
