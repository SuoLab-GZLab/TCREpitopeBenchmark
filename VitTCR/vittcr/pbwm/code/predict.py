import os
# gpu_id = '5'
# os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
import datetime, sys, pickle, torch, re, random
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
from collections import OrderedDict
from TcrPepTransform_beta import *
from TcrPepTransform_utils import *
import matplotlib.pyplot as plt
from matplotlib import pyplot


#Args parser 
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-r", "--repeat", default=5, type=int, help="Specify the repeat")
parser.add_argument("-f", "--fold", default=5, help="Specify cv fold")
parser.add_argument("-i", "--input", default='input.tsv', help="Specify the test input files")
parser.add_argument("-o", "--output", default='results.tsv', help="Specify the test input files")
args = parser.parse_args()


Atchley = {
        'A': np.array((-0.591,-1.302,-0.733,1.57,-0.146)),
        'C': np.array((-1.343,0.465,-0.862,-1.02,-0.255)),
        'D': np.array((1.05,0.302,-3.656,-0.259,-3.242)),
        'E': np.array((1.357,-1.453,1.477,0.113,-0.837)),
        'F': np.array((-1.006,-0.59,1.891,-0.397,0.412)),
        'G': np.array((-0.384,1.652,1.33,1.045,2.064)),
        'H': np.array((0.336,-0.417,-1.673,-1.474,-0.078)),
        'I': np.array((-1.239,-0.547,2.131,0.393,0.816)),
        'K': np.array((1.831,-0.561,0.533,-0.277,1.648)),
        'L': np.array((-1.019,-0.987,-1.505,1.266,-0.912)),
        'M': np.array((-0.663,-1.524,2.219,-1.005,1.212)),
        'N': np.array((0.945,0.828,1.299,-0.169,0.933)),
        'P': np.array((0.189,2.081,-1.628,0.421,-1.392)),
        'Q': np.array((0.931,-0.179,-3.005,-0.503,-1.853)),
        'R': np.array((1.538,-0.055,1.502,0.44,2.897)),
        'S': np.array((-0.228,1.399,-4.76,0.67,-2.647)),
        'T': np.array((-0.032,0.326,2.213,0.908,1.313)),
        'V': np.array((-1.337,-0.279,-0.544,1.242,-1.262)),
        'W': np.array((-0.595,0.009,0.672,-2.128,-0.184)),
        'Y': np.array((0.26,0.83,3.097,-0.838,1.512))        
    }


def generate_interaction_map(df, query, length_cdr3, length_pep, chain):
    dict_combined = defaultdict(list)
    dict_intermap = defaultdict(list)
    features = list(query.columns)
    
    cdr3s=[]
    peptides=[]
    Bindings=[]

    for order in range(5):
        feature=features[order]
        for idx in range(df.shape[0]):
            if chain=="beta":
                cdr3 = df.loc[idx]['cdr3b'].upper()
            elif chain=='alpha':
                cdr3 = df.loc[idx]['cdr3a'].upper()
            peptide = df.loc[idx]['peptide'].upper()
            
            if 'Binding' in df.columns:
                binding = df.loc[idx]['Binding']
            intermap = np.zeros((length_cdr3, length_pep))
            for i,m in enumerate(cdr3):
                feature_cdr3 = query[feature][m]
                for j,n in enumerate(peptide):
                    feature_peptide = query[feature][n]
                    intermap[i,j] = abs(feature_cdr3-feature_peptide)
            dict_intermap[feature].append(intermap)
            
            if order==0:
                cdr3s.append(cdr3)
                peptides.append(peptide)
                if 'Binding' in df.columns:
                    Bindings.append(binding)
    
    dict_combined['cdr3']=cdr3s
    dict_combined['peptide']=peptides
    if 'Binding' in df.columns:
        dict_combined['Binding']=Bindings
    
    # Combine the intermaps
    data_0 = np.array(dict_intermap[features[0]])
    data_1 = np.array(dict_intermap[features[1]])
    data_2 = np.array(dict_intermap[features[2]])
    data_3 = np.array(dict_intermap[features[3]])
    data_4 = np.array(dict_intermap[features[4]])
    combinedmap = np.concatenate((np.expand_dims(data_0, 1), 
                                  np.expand_dims(data_1, 1), 
                                  np.expand_dims(data_2, 1),
                                  np.expand_dims(data_3, 1),
                                  np.expand_dims(data_4, 1),), 1)

    combinedmap = combinedmap.tolist()
    if chain=='alpha':
        dict_combined['combined_map_alpha'] = combinedmap
    elif chain=='beta':
        dict_combined['combined_map_beta'] = combinedmap

    return dict_combined


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1, keepdims = True) 
    s = x_exp / x_sum    
    return s
    

def predicting(model, data_test):
    out_eval = model(data_test)
    out_eval = out_eval.detach().numpy()
    out_eval = softmax(out_eval)
    pred_prob = out_eval[:,1]
    return pred_prob


if __name__=="__main__":
    # Data loading
    data = pd.read_csv(args.input, sep='\t')

    # Encoding
    queryfile=Atchley
    queryfile=pd.DataFrame(queryfile).T
    queryfile.columns=['f1','f2','f3','f4','f5']

    dict_combined_beta = generate_interaction_map(df=data, query=queryfile, length_cdr3=20, length_pep=12, chain='beta')   
    data_external = np.array(dict_combined_beta['combined_map_beta'])
    data_external = torch.from_numpy(data_external).float()
    
    # Predicting ...
    sums = [0] * data.shape[0]
    lists = []
    for repeat in range(args.repeat):
        for fold in range(args.fold):
            metricpath = '../metrics_repeat{}'.format(repeat)        
            validfile=[i for i in os.listdir(metricpath) if i.startswith("Test_model") and i.endswith("fold{}.tsv".format(fold))][0]

            validdata = pd.read_csv("{}/{}".format(metricpath,validfile), sep='\t')            
            ind = validdata['loss'].idxmin()
            epoch = validdata.iloc[ind]['epoch']
            model = TcrPepTransform_single(input_height=20, 
                                            input_width=12, 
                                            in_chans=5, 
                                            patch_size=4, 
                                            num_classes=2, 
                                            embed_dim=256, 
                                            depth=1, 
                                            num_heads=4, 
                                            mlp_ratio=4, 
                                            qkv_bias=True, 
                                            drop_rate=0.1, 
                                            attn_drop_rate=0.05, 
                                            drop_path_rate=0, 
                                            act_layer=nn.GELU)
            params = "../model_repeat{}/fold{}/{}_epochs.pt".format(repeat, fold, str(int(epoch)))  
            
            checkpoint = torch.load(params, map_location='cpu')
            model.load_state_dict(checkpoint)
            model.eval()

            pred_prob = predicting(model=model, data_test=data_external)
            
            for item in range(data.shape[0]):
                sums[item] += pred_prob[item]

    averages = [s / (int(args.repeat)*int(args.fold)) for s in sums]
    data['probability'] = averages
    data.to_csv(args.output, sep='\t', index=False)