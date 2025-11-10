import csv
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.base import clone
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_decomposition import CCA

from sklearn.feature_selection import SelectKBest, chi2

import warnings

warnings.filterwarnings('ignore')

"""
readFile：读取原始TCR-epitope数据函数
filePath:原始数据文件路径
key_fields:要存储的键值List，从前往后范围减小
value_fields:要存储的内容List，内容不分先后
delimiter:csv.DictReader参数，换行符标志
返回值：返回一个dict
"""
def readFile(filePath, key_fields, value_fields, delimiter = '\t'):
    retDict = {}
    with open(filePath, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in reader:
            keys = [row[k] for k in key_fields]
            values = [row[s] for s in value_fields]
            sub_dict = retDict
            for key in keys[:-1]:
                if key not in sub_dict:
                    sub_dict[key] = {}
                sub_dict = sub_dict[key]
            key = keys[-1]
            if key not in sub_dict:
                sub_dict[key] = []
            sub_dict[key].append(values)

    return retDict

"""
dataPreprocess:对数进行预处理，统计不同表位的可用数据并加以筛选
dataDict:输入的原始Dict格式数据
species:要处理的种族数据，如HomoSapiens或者MusMusculus
gene:要处理的基因类型，如TRB或TRA
miniValue:筛选epitope的最小数据量
返回值：返回一个dict
"""
def dataPreprocess(dataDict, species, gene, miniValue=50):
    retDict = {}
    for s in species:
        for g in gene:
            epitopeList = list(dataDict[s][g].keys())
            for epi in epitopeList:
                epiSize = len(dataDict[s][g][epi])
                if epiSize >= miniValue:
                    retDict[epi] = dataDict[s][g][epi]
                    for i in range(epiSize):
                        meta = ast.literal_eval(retDict[epi][i][4])
                        sub_id = meta['subject.id']
                        reference = retDict[epi][i][3]
                        retDict[epi][i][3] = reference + '_' + sub_id
                        retDict[epi][i] = retDict[epi][i][:-1]
    return retDict


def statisticsEpitope(epiDict, path='data/epiDict.csv'):
    print('{:22s} {:s}'.format('Epitope', 'Number'))
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['epitope', 'info'])
        for epi in epiDict:
            print('{:22s} {:d}'.format(epi, len(epiDict[epi])))
            for item in epiDict[epi]:
                writer.writerow([epi, item])

    print("epiDict have been saved in: {:s}".format(path))


def removeDuplicates(epiDict):
    ret_dict = {}
    for epi in epiDict:
        item_list = []
        item_dict = set()
        for item in epiDict[epi]:
            if item[0] not in item_dict:
                item_list.append(item)
                item_dict.add(item[0])
        ret_dict[epi] = item_list
    return ret_dict


def saveEpitope(epiDict, path):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['epitope', 'cdr3b', 'vb_gene', 'vj_gene'])
        for epi in epiDict:
            for item in epiDict[epi]:
                writer.writerow([epi, item[0], item[1], item[2]])

    print("Epitope-TCR pairs have been saved in: {:s}".format(path))

def splitCDR(CDRseq, k = 3):
    retDict = []
    for i in range(len(CDRseq) - k + 1):
        retDict.append(CDRseq[i:i+k])
    return retDict


def statisticsKmer(epiDict, k=3):
    kmerDict = {}
    for epi in epiDict:
        for i in range(len(epiDict[epi])):
            splitList = splitCDR(epiDict[epi][i], k)
            for split in splitList:
                if split not in kmerDict:
                    kmerDict[split] = 1
                else:
                    kmerDict[split] += 1
    return kmerDict


def buildFeatures(epiDict, kmerDict, k=3,
                  return_kmers=False):
    counter = 0
    for epi in epiDict:
        counter += len(epiDict[epi])
    retArr = np.zeros((counter, len(kmerDict)))

    kmerList = list(kmerDict.keys())
    retLabel = []

    iter = 0
    epinum = 0
    for epi in epiDict:
        for cdr in range(len(epiDict[epi])):
            splitlist = splitCDR(epiDict[epi][cdr], k)
            retLabel.append(epinum)
            i = 0
            for kmer in kmerList:
                retArr[iter][i] = splitlist.count(kmer)
                i += 1
            iter += 1
        epinum += 1

    if not return_kmers:
        return np.array(retArr), np.array(retLabel)
    else:
        return np.array(retArr), np.array(retLabel), kmerList


def _cal_micro_ROC(y_test, y_score):
    """Calculate the micro ROC value"""
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
    return fpr, tpr, auc(fpr, tpr)


def _cal_macro_ROC(y_test, y_score, fpr, tpr, n_classes):
    """Calculate the macro ROC value"""
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)


def _plot_roc_curves(fpr, tpr, roc_auc, epi_list, title):
    """PLot the ROC curve"""
    mean_fpr = np.linspace(0, 1, 200)
    tprs = list()
    aucs = list()
    for i in range(len(epi_list)):
        tprs.append(interp(mean_fpr, fpr[i], tpr[i]))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc[i])
        cur_auc = round(roc_auc[i], 3)
        plt.plot(fpr[i], tpr[i], lw=1, alpha=0.5, label='{0}({1})'.format(epi_list[i], str(cur_auc)))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC({})'.format(round(mean_auc, 3)), lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()


def _cal_roc_auc(y_test, y_score, y_pred, epi_list, draw_roc_curve=True, title="ROC curves"):
    """"Calculate the AUROC value and draw the ROC curve."""
    fpr = dict()
    tpr = dict()
    precision = list()
    recall = list()
    roc_auc = dict()
    y_test = label_binarize(y_test, classes=np.arange(len(epi_list)))
    y_pred = label_binarize(y_pred, classes=np.arange(len(epi_list)))
    for i in range(len(epi_list)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precision.append(precision_score(y_test[:, i], y_pred[:, i]))
        recall.append(recall_score(y_test[:, i], y_pred[:, i]))

    # micro-average ROC
    fpr["micro"], tpr["micro"], roc_auc["micro"] = _cal_micro_ROC(y_test, y_score)

    # macro-average ROC
    fpr["macro"], tpr["macro"], roc_auc["macro"] = _cal_macro_ROC(y_test, y_score, fpr, tpr, len(epi_list))

    # plot all ROC curves
    if draw_roc_curve:
        _plot_roc_curves(fpr, tpr, roc_auc, epi_list, title)

    return roc_auc, np.mean(precision), np.mean(recall)


def predict_auc(X, y, classifier, cv, epi_list, draw_roc_curve=True, title="ROC curves"):
    auc_dict = {}
    acc_list, precision_list, recall_list = [], [], []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=666)
    cur_fold = 1
    for train_index, test_index in skf.split(X, y):
        # split cross-validation folds
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train, X_test = pca_analyse(X_train, X_test, 0.9)

        clf = clone(classifier)
        clf.fit(X_train, y_train)

        acc_list.append(clf.score(X_test, y_test))

        y_prob = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)

        auc_dict[cur_fold], precision, recall = _cal_roc_auc(y_test, y_prob, y_pred, epi_list, draw_roc_curve)

        precision_list.append(precision)
        recall_list.append(recall)
        cur_fold += 1

    return auc_dict, acc_list, precision_list, recall_list


def pca_analyse(X_train, X_test, rate=0.9):
    """Perform PCA for the train set and test set."""
    pca = PCA(n_components=rate).fit(X_train)
    return pca.transform(X_train), pca.transform(X_test)


def cca_analyse(X_train, y_train, X_test, n_components=100):
    """Perform CCA for the train set and test set."""
    cca = CCA(n_components=n_components).fit(X_train, y_train)
    return cca.transform(X_train), cca.transform(X_test)


def kernel_pca_analyse(X_train, X_test, kernel, rate=0.9):
    """Perform kernel PCA for the train set and test set."""
    pca = KernelPCA(n_components=rate, kernel=kernel)
    return pca.transform(X_train), pca.transform(X_test)


def data_preprocess(file, k=3, remove_duplicate=False,
                    return_kmers=False, min_tcrs_amount=10):
    print("Reading file: ", file)
    df = pd.read_csv(file)

    if remove_duplicate:
        head_list = df.columns.values.tolist()
        assert 'epitope' in head_list and 'cdr3b' in head_list and 'vb_gene' in head_list
        subset = ['epitope', 'cdr3b', 'vb_gene']
        if 'vb_gene' in head_list:
            df.drop_duplicates(subset=subset, inplace=True)

    epiDict = {}
    for index, row in df.iterrows():
        if row['epitope'] not in epiDict:
            epiDict[row['epitope']] = []
        epiDict[row['epitope']].append(row['cdr3b'])

    # Threshold：10, only epitopes with binding tcr sequences over 10 are remained.
    epiDict_filtered = {}
    for epi in epiDict:
        if len(epiDict[epi]) > min_tcrs_amount:
            epiDict_filtered[epi] = epiDict[epi]
    epiDict = epiDict_filtered

    statistics_epi = []
    statistics_num = []
    print('{:22s} {:s}'.format('Epitope', 'Number'))
    for epi in epiDict:
        statistics_epi.append(epi)
        statistics_num.append(len(epiDict[epi]))
        print('{:22s} {:d}'.format(epi, len(epiDict[epi])))

    kmerDict = statisticsKmer(epiDict, k)

    if not return_kmers:
        X, y = buildFeatures(epiDict, kmerDict, k)
        return X, y, list(epiDict.keys())
    else:
        X, y, kmers_list = buildFeatures(epiDict, kmerDict, k, return_kmers=True)
        return X, y, list(epiDict.keys()), kmers_list



def draw_heatmap(X, y):
    print('ploting heatmap ...')
    sns.heatmap(np.c_[X, y], cmap=sns.cm.rocket_r)
    plt.show()


def make_table(auc_dict, acc_list, precision_list, recall_list, cv=5):
    # record auc result for each fold of cross-validation
    table = np.array([0]*(4*(cv+1)), dtype=np.float16).reshape((cv+1, 4))
    for fold in auc_dict:
        table[fold-1][0] = auc_dict[fold]['macro']
    for i in range(cv):
        table[i][1] = acc_list[i]
        table[i][2] = precision_list[i]
        table[i][3] = recall_list[i]
    for i in range(4):
        table[cv][i] = np.mean(table[:-1,i])
    return table
