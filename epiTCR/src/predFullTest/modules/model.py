import sklearn.metrics as metrics

from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,classification_report,roc_curve,auc, f1_score
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import functools

# lst_models = [ ('Random Forest - without MHC', RandomForestClassifier(bootstrap=False, max_features=15,
#                          n_estimators=300, n_jobs=-1, random_state=42)),
#                ('Random Forest - with MHC', RandomForestClassifier(max_features=20,
#                          n_estimators=300, n_jobs=-1, random_state=42))]

def train(lst_models, X, y, cv):
    res_table = []
    for mdl_name, model in tqdm(lst_models):
        tic = time.time()
        cv_res = cross_validate(model, X, y, cv=cv, return_train_score=True, scoring=['accuracy', 'roc_auc'])
        res_table.append([mdl_name, 
                          cv_res['train_accuracy'].mean(), 
                          cv_res['test_accuracy'].mean(), 
                          np.abs(cv_res['train_accuracy'].mean() - cv_res['test_accuracy'].mean()),
                          cv_res['train_accuracy'].std(),
                          cv_res['test_accuracy'].std(),
                          cv_res['train_roc_auc'].mean(),
                          cv_res['test_roc_auc'].mean(),
                          np.abs(cv_res['train_roc_auc'].mean() - cv_res['test_roc_auc'].mean()),
                          cv_res['train_roc_auc'].std(),
                          cv_res['test_roc_auc'].std(),
                          cv_res['fit_time'].mean()
        ])
        toc = time.time()
        print('\tModel {} has been trained in {:,.2f} seconds'.format(mdl_name, (toc - tic)))
    
    res_table = pd.DataFrame(res_table, columns=['model', 'train_acc', 'test_acc', 'diff_acc',
                                                 'train_acc_std', 'test_acc_std', 'train_roc_auc', 'test_roc_auc',
                                                 'diff_roc_auc', 'train_roc_auc_std', 'test_roc_auc_std', 'fit_time'])
    res_table.sort_values(by=['test_acc', 'test_roc_auc'], ascending=False, inplace=True)

    return res_table.reset_index(drop=True)

def confusionMatrix(y_true, y_pred):
    target_names = ['Non-binder', 'Binder']
    print(classification_report(y_true, y_pred, target_names=target_names))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=target_names, columns=target_names)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted values')
    plt.ylabel('Actual values')
    plt.show()
    
def rocAuc(model, X, y_true):
    plot_roc_curve(model, X, y_true)
    plt.show()
    
def _rocAuc(y_true, y_score):
    y_pred01_proba = y_score.to_numpy()
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred01_proba)
    auc = metrics.roc_auc_score(y_true, y_pred01_proba)
    plt.plot(fpr,tpr,label="AUC = "+str(auc))
#     print ("AUC : ", auc)
    plt.legend(loc=4)
    plt.show()
    
def predicMLModel(model, data, X_test, y_test, path):
    y_rf_test_proba = model.predict_proba(X_test)
    df_test_rf = pd.DataFrame(data = y_rf_test_proba, columns = ["tmp", "predict_proba"])

    df_test_rf = df_test_rf[["predict_proba"]]
    df_prob_test_rf = pd.concat([data, df_test_rf], axis=1)
    df_prob_test_rf['binder_pred'] = np.where(df_prob_test_rf['predict_proba'] >= 0.5, 1, 0)

    y_test = y_test["binder"].to_numpy()
    y_test_pred = df_prob_test_rf["binder_pred"].to_numpy()

    confusionMatrix(y_test, y_test_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    accuracy = float(accuracy_score(y_test, y_test_pred).ravel())
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    auc = metrics.roc_auc_score(y_test, df_prob_test_rf["predict_proba"])
    print ("AUC : ", round(auc, 3))
    print ("Accuracy score  : ", round(accuracy, 3))
    print('Sensitivity (TPR): ', round(sensitivity, 3))
    print('Specificity (TNR): ', round(specificity, 3))

#     rocAuc(model, X_test, y_test)
    df_prob_test_rf.to_csv(path, index=False)
    
    return round(auc, 3), round(accuracy, 3), round(sensitivity, 3), round(specificity, 3)
    
def trainTunningModel(lst_models, X, y, cv):
    models_final = []
    for model_name, model, params in tqdm(lst_models):
        tic     = time.time()
        search = GridSearchCV(estimator=model, param_grid=params, cv=cv, scoring='roc_auc', n_jobs=-1)
        search.fit(X, y)
        model_tunned = model.set_params(**search.best_params_)
        models_final.append((model_name, model_tunned))
        toc = time.time()
        print('Model {} has been tunned in {:,.2f} seconds'.format(model_name, (toc - tic)))

    return models_final

def rocAuc(model, X, y_true):
    plot_roc_curve(model, X, y_true)
    plt.show()
    
def modelRun(algo, pX_res, py_res, pX_test, py_test):

    algo.fit(pX_res, np.ravel(py_res))
    y_pred = algo.predict(pX_test)
    y_pred_proba = algo.predict_proba(pX_test)[:,1]
    
    accuracy = accuracy_score(py_test, y_pred).ravel()
    classify_metrics = classification_report(py_test, y_pred)
    f1 = f1_score(py_test, y_pred).ravel()
    
    fpr, tpr, thresholds = roc_curve(py_test, y_pred_proba)
    auc_score = auc(fpr, tpr)
    tn, fp, fn, tp = confusion_matrix(py_test, y_pred).ravel()
    confusionMatrix(py_test, y_pred)
    
    print (algo)
    print ("Accuracy score        : ", accuracy)
    print ("F1 score              : ", f1)
    print ("AUC                   : ", auc_score)
    print ('Sensitivity (TPR)     : ', tp/(tp+fn))
    print ('Specificity (TNR)     : ', tn/(tn+fp))
#     print ("classification report :\n", classify_metrics)

    rocAuc(algo, pX_test, py_test)
    return accuracy, classify_metrics, fpr , tpr, auc_score, f1
    
def predictModel(path, path_out):
    data_predict = pd.read_csv(path)
    data_predict['binder'] = data_predict['binder'].astype(int)
    data_predict['binder_pred'] = np.where(data_predict['prediction'] >= 0.5, 1, 0)

    y_test = data_predict["binder"].to_numpy()
    y_test_pred = data_predict["binder_pred"].to_numpy()

    confusionMatrix(y_test, y_test_pred)

    tn, fp, fn, tp= confusion_matrix(y_test, y_test_pred).ravel()
    accuracy = float(accuracy_score(y_test, y_test_pred).ravel())
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    auc = metrics.roc_auc_score(y_test, data_predict["prediction"])
    print ("AUC : ", auc)
    print ("Accuracy score  : ", accuracy)
    print('Sensitivity (TPR): ', sensitivity)
    print('Specificity (TNR): ', specificity)

    _rocAuc(y_test, data_predict["prediction"])
    data_predict.to_csv(path_out, index=False)
    
    return round(auc, 3), round(accuracy, 3), round(sensitivity, 3), round(specificity, 3)

def saveByPickle(object, path):
    pickle.dump(object, open(path, "wb"))
    print(f"{object} has been saved at {path}.")

def evaluation(tunning_models, X_train, y_train, X_test, y_test):
    res = []
    for name, model in tqdm(tunning_models):
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_roc_auc = roc_auc_score(y_train, y_train_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_pred)
        res.append([name, train_acc, test_acc, train_roc_auc, test_roc_auc])
        
    res = pd.DataFrame(res, columns=['model', 'train_acc', 'test_acc', 'train_roc_auc', 'test_roc_auc'])
    res.sort_values(by=['test_acc', 'test_roc_auc'], ascending=False, inplace=True)
    
    return res.reset_index(drop=True)
