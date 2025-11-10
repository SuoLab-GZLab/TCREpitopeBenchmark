model_name='TCR-H'

import peptides
import pandas as pd

import sys,os
import numpy as np
import scipy as scipy
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
# import shap
import urllib
#import requests
import zipfile
#import seaborn
from io import StringIO
import warnings
warnings.filterwarnings('ignore')
from scipy import sparse
from pandas.plotting import scatter_matrix
from datetime import datetime
from pprint import pprint
from sklearn import tree

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve, auc, f1_score, accuracy_score
from sklearn.metrics import precision_recall_curve
#from sklearn.externals import joblib
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix

def fix(data_ori):
    data=pd.read_csv(data_ori)
    #'CDR3.beta', 'antigen_epitope','mhc.a','label','negative.source','license'
    data.rename(columns={'CDR3B':'CDR3.beta','Epitope':'antigen_epitope','MHC':'mhc.a','Affinity':'label'},inplace=True)
    df=data[['CDR3.beta', 'antigen_epitope','mhc.a','label']]
    
    df_epi = pd.DataFrame([peptides.Peptide(s).descriptors() for s in df.antigen_epitope])
    df_epi.columns='epitope_'+df_epi.columns
    df_cdrb = pd.DataFrame([peptides.Peptide(s).descriptors() for s in df['CDR3.beta']])
    df_cdrb.columns='cdr3_'+df_cdrb.columns
    df=pd.concat([df, df_cdrb, df_epi],axis=1)
    return df

def Model_retraining(trainfile, testfile, save_model_path, resultfile_path):
    df_train =fix(trainfile)
    df_test = fix(testfile)
    var_columns = [c for c in df_train.columns if c not in('CDR3.beta', 'antigen_epitope','mhc.a','label','negative.source','license')]
    X_train = df_train.loc[:, var_columns]
    y_train = df_train.loc[:, 'label']
    
    X_test= df_test.loc[:, var_columns]
    y_test = df_test.loc[:, 'label']
    def correlation(dataset, threshold):
        col_corr = set()  # Set of all the names of correlated columns
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        return col_corr
    
    corr_features = correlation(X_train, 0.8)
    corr_features = list(corr_features)
    X_train = X_train.drop(corr_features,axis=1)
    X_test = X_test.drop(corr_features,axis=1)
    X_train_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)    
    roc_scorer = make_scorer(roc_auc_score) #note, can use this or 'f1' for scoring below
    parameters= {'C': [1.0], 'gamma': ['scale'], 'kernel':['rbf'], 'class_weight':['balanced']}
    classifier=GridSearchCV(SVC(probability=True), parameters, cv=None, scoring=roc_scorer, n_jobs=-1)
    clf=classifier.fit(X_train_scaled, y_train)
    print("\nThe best CV parameters for rbf model are [" + str(classifier.best_params_) + "] with a score on train data of [" + str(classifier.best_score_) + "]")
    joblib.dump(clf, f'{save_model_path}.pkl')
    print("\nEvaluating...")
    preds=clf.predict_proba(X_test_scaled)
    data = pd.read_csv(testfile)
    probability = data[['Epitope', 'CDR3B', 'Affinity']]
    probability = probability.rename(columns={'Affinity': 'y_true'})
    probability['y_prob'] = preds[:,1]
    probability['y_pred'] = probability['y_prob'].apply(lambda x: 1 if x >= 0.5 else 0)
    probability.to_csv(f'{resultfile_path}probability.csv', index=False)
    print("Saving done")

import pandas as pd
database=['as','hy','pt_all','as_epi']
name1=['1','2','3','4','5']
name=['1_1']
for i in database:
    for j in name:
        for k in name1:
            trainfile_path ="../data/retrain/ce_seen/"+i+"/"+k +'_'+j+"train.csv"
            testfile_path="../data/retrain/ce_seen/"+i+"/"+k+'_'+"1_1test.csv"
            save_modle_path="/home/bingxing2/gpuuser907/models/TCR_H/retrain/ce_seen/"+i+"/" +k+'_'+j
            result_path="/home/bingxing2/gpuuser907/all_result/TCR_H/retrain/ce_seen/"+i+"/"+k+'_'+j+'test_'
            os.makedirs(save_modle_path,exist_ok=True)
            os.makedirs(result_path,exist_ok=True)
            Model_retraining(trainfile_path,testfile_path,save_modle_path,result_path)
            