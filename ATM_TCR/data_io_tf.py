#!/usr/bin/env python

"""
Functions for data IO for neural network training.

Credit: netTCR
https://github.com/mnielLab/netTCR
"""

from __future__ import print_function
import sys
import numpy as np


def read_pTCR(filename):
    '''
    read data file with MHC-peptide-TCR data

    parameters:
        - filename : file with AA seq of peptide and TCR
    returns:
        - peptides : list of peptide sequences
        - tcrs : list of TCRb-CDR3 sequences
    '''

    # initialize variables:
    peptides=[]
    tcrs=[]
    bound=[]

    # read data:
    infile=open(filename,"r")
    for l in infile:
        if l[0] != "#":
            data = l.strip().split("\t")
            if len(data) < 2:
                data = l.strip().split(",")
            if len(data) < 2:
                sys.stderr.write("Problem with input file format!\n")
                sys.stderr.write(l)
                sys.exit(2)
            else:
                if l[0] != "peptide":
                    peptides.append(data[0])
                    tcrs.append(data[1])
                    bound.append(data[2])
    infile.close()

    # return data:
    return np.array(peptides), np.array(tcrs), np.array(bound)

def read_pTCR_peplist(filename,peplist):
    '''
    read data file with MHC-peptide-TCR data

    parameters:
        - filename : file with AA seq of TCRs
        - peplist : list of peptides
    returns:
        - peptides : list of peptide sequences
        - tcrs : list of TCRb-CDR3 sequences
    '''

    # initialize variables:
    peptides=[]
    tcrs=[]

    # read data:
    infile=open(filename,"r")
    for l in infile:
        if l[0] != "#":
            l=l.strip()
            if len(l.split("\t"))>1 or len(l.split(","))>1:
                sys.stderr.write("Problem with input file format!\n")
                sys.stderr.write(l)
                sys.exit(2)
            else:
                tcrs.extend(len(peplist)*[l])
                peptides.extend(peplist)
    infile.close()

    # return data:
    return peptides, tcrs


def read_blosum_MN(filename):
    '''
    read in BLOSUM matrix

    parameters:
        - filename : file containing BLOSUM matrix

    returns:
        - blosum : dictionnary AA -> blosum encoding (as list)
    '''

    # read BLOSUM matrix:
    blosumfile = open(filename, "r")
    blosum = {}
    B_idx = 99
    Z_idx = 99
    star_idx = 99

    for l in blosumfile:
        l = l.strip()

        if l[0] != '#':
            l= list(filter(None,l.strip().split(" ")))

            if (l[0] == 'A') and (B_idx==99):
                B_idx = l.index('B')
                Z_idx = l.index('Z')
                star_idx = l.index('*')
            else:
                aa = str(l[0])
                if (aa != 'B') &  (aa != 'Z') & (aa != '*'):
                    tmp = l[1:len(l)]
                    # tmp = [float(i) for i in tmp]
                    # get rid of BJZ*:
                    tmp2 = []
                    for i in range(0, len(tmp)):
                        if (i != B_idx) &  (i != Z_idx) & (i != star_idx):
                            tmp2.append(float(tmp[i]))

                    #save in BLOSUM matrix
                    #tmp2 = [i * 0.2 for i in tmp2] #scale (divide by 5)
                    blosum[aa]=tmp2
    blosumfile.close()
    blosum["B"]=np.ones(21)*0.1
    blosum["E"]=np.ones(21)*(-0.1)
    return(blosum)

def enc_list_bl(aa_seqs, blosum):
    '''
    blosum encoding of a list of amino acid sequences with padding

    parameters:
        - aa_seqs : list with AA sequences
        - blosum : dictionnary: key= AA, value= blosum encoding
    returns:
        - enc_aa_seq : list of np.ndarrays containing padded, encoded amino acid sequences
    '''

    # encode sequences:
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

    # pad sequences:
    max_seq_len = max([len(x) for x in aa_seqs])
    n_seqs = len(aa_seqs)
    n_features = sequences[0].shape[1]

    enc_aa_seq = np.zeros((n_seqs, max_seq_len, n_features))
    for i in range(0,n_seqs):
        enc_aa_seq[i, :sequences[i].shape[0], :n_features] = sequences[i]

    return enc_aa_seq

def enc_list_bl_start_stop(aa_seqs, blosum):
    '''
    blosum encoding of a list of amino acid sequences with padding

    parameters:
        - aa_seqs : list with AA sequences
        - blosum : dictionnary: key= AA, value= blosum encoding
    returns:
        - enc_aa_seq : list of np.ndarrays containing padded, encoded amino acid sequences
    '''

    # encode sequences:
    sequences=[]
    for seq in aa_seqs:
        # replace O (pyrrolysine) with lysine (K):
        #seq.replace("0", "K")
        e_seq=np.zeros((len(seq)+2,len(blosum["A"])))
        # start encoding:
        e_seq[0]=np.ones(len(blosum["A"]))*0.1
        # sequence encoding:
        count=1
        for aa in seq:
            if aa == "O":
                aa ="K"
            if aa in blosum:
                e_seq[count]=blosum[aa]
                count+=1
            else:
                sys.stderr.write("Unknown amino acid: "+ aa +", encoding aborted!\n")
                sys.exit(2)
        # end encoding:
        e_seq[count]=np.ones(len(blosum["A"]))*(-0.1)
        # save:
        sequences.append(e_seq)

    # pad sequences:
    max_seq_len = max([len(x) for x in aa_seqs]) +2
    n_seqs = len(aa_seqs)
    n_features = sequences[0].shape[1]

    enc_aa_seq = np.zeros((n_seqs, max_seq_len, n_features))
    for i in range(0,n_seqs):
        enc_aa_seq[i, :sequences[i].shape[0], :n_features] = sequences[i]

    return enc_aa_seq