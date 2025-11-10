from tcrpeg.TCRpeg import TCRpeg
import pandas as pd
import numpy as np
import pickle
import copy
import numpy as np
from tqdm import tqdm
from model import TEINet
from scipy.special import expit
import argparse
from predict import *
from utils import aa_scan, contact_index,load_teinet
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_path',type=str,default='results/model.pth')
    parser.add_argument('--cat_size',type=int,default=768*2)
    parser.add_argument('--offset',type=int,default=0)
    parser.add_argument('--scan_aa',type=str,default='A',help='The scan AA')
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--scan_aa_alternate',type=str,default='G',help='The alternative scan AA; If the aa=scan_aa, will then use this alternative AA for subsitution')
    parser.add_argument('--scan_axis',type=int,default=0)
    parser.add_argument('--threshold',type=float,default=5.0,help='distance threshold')
    parser.add_argument('--permuted_path',type=str,default='None',help='The path to store the permuted sequences')
    parser.add_argument('--store_path',type=str,default='None',help='The file store the score difference for each permuted sequence')
    parser.add_argument('--pdb_original',type=str,default='data/PDB_distance/complex_data_original.csv')
    parser.add_argument('--distance_matrix',type=str,default='data/PDB_distance/distance_matrices.p')
    args = parser.parse_args()

    #load model
    model_tcr = TCRpeg(hidden_size=args.cat_size // 2,num_layers = 3,load_data=False,embedding_path='encoders/aa_emb_tcr.txt',device=args.device)
    model_tcr.create_model()
    model_epi = TCRpeg(hidden_size=args.cat_size // 2,num_layers = 3,load_data=False,embedding_path='encoders/aa_emb_epi.txt',device=args.device)
    model_epi.create_model()
    model = TEINet(en_tcr=model_tcr,en_epi = model_epi,cat_size=args.cat_size,normalize=True,device=args.device)

    model.load_state_dict(torch.load(args.model_path))
    # model = load_teinet(args.model_path,device=args.device)
    model.eval()
    model = model.to(args.device)
    #load model


    pdb_data  = pd.read_csv(args.pdb_original)
    with open(args.distance_matrix, 'rb') as f:
        pdb_distances = pickle.load(f)
    cdrs,epis,pdbs = pdb_data['cdr3'].values, pdb_data['antigen.epitope'].values, pdb_data['PDB_ID'].values
    contact_diff, non_contact_diff = [],[]

    record_cdrs,record_epis, record_labels, sources = [],[],[],[]
    index = 0
    for i in range(len(cdrs)):
        cdr,epi,pdb = cdrs[i],epis[i],pdbs[i]
        if pdb not in pdb_distances.keys():
            continue
        distances = pdb_distances[pdb]
        initial_score = predict_only([cdr],[epi],model)[0]
        record_cdrs.append(cdr)
        record_epis.append(epi)
        record_labels.append(str(index) + 'R') #reference  
        sources.append(pdb)      
        contact_idx = contact_index(distances,offset=args.offset,axis=args.scan_axis,threshold=args.threshold) #from 0-end        
        if len(contact_idx) == 0:
            continue
        contact_idx = [x - args.offset for x in contact_idx]
        seq = cdr if args.scan_axis == 0 else epi
        permuted_seqs = aa_scan(seq,scan_aa=args.scan_aa,scan_aa_alternate=args.scan_aa_alternate, offset=args.offset)
        record_cdrs.extend(permuted_seqs)
        record_epis.extend([epi] * len(permuted_seqs))
        sources.extend([pdb] * len(permuted_seqs))
        if args.scan_axis == 0:
            permuted_scores = predict_only(permuted_seqs,[epi]*len(permuted_seqs),model)
        else :
            permuted_scores = predict_only([cdr]*len(permuted_seqs),permuted_seqs,model)
        diff = np.abs(initial_score - permuted_scores) #len=len(seq)-2*offset
        contact_diff.extend([diff[k] for k in contact_idx])
        if i == 0:
            print(contact_idx)
        ref = np.array(['{}D'.format(index)]* len(diff))
        ref[contact_idx] = '{}C'.format(index)
        record_labels.extend(ref)
        
        non_contact_diff.extend([diff[k] for k in range(len(diff)) if k not in contact_idx])
        print('current diff for ',pdb)
        print('contact : ',np.mean([diff[k] for k in contact_idx]))
        print('non contact : ', np.mean([diff[k] for k in range(len(diff)) if k not in contact_idx]))
        index += 1
    print('mean contact : ',np.mean(contact_diff))
    print('mean noncontact : ',np.mean(non_contact_diff))
    
    if args.permuted_path != 'None':
        #will produce a file that record all the permuted sequences
        res_df = pd.DataFrame(columns = ['CDR3.beta','Epitope','Labels','Source'])
        res_df['CDR3.beta'] = record_cdrs
        res_df['Epitope'] = record_epis
        res_df['Labels'] = record_labels        
        res_df['Source'] = sources
        res_df.to_csv(args.permuted_path,index=False)
        
    if args.store_path != 'None':
        #store the score difference
        res_df = pd.DataFrame(columns = ['contact','non_contact'])
        if len(contact_diff) > len(non_contact_diff):
            non_contact_diff = non_contact_diff + [0] * (len(contact_diff) - len(non_contact_diff))
        else :
            contact_diff = contact_diff + [0] * np.abs(len(contact_diff) - len(non_contact_diff))
        res_df['contact'] = contact_diff
        res_df['non_contact'] = non_contact_diff
        res_df.to_csv(args.store_path,index=False)
