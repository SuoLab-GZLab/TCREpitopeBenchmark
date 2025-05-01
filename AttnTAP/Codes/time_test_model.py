import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from torch.utils.data import DataLoader
from network import AttentionBiLSTMClassifier, AttnBLSTMDataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve
torch.manual_seed(2021)
np.random.seed(2021)

def test_model(file_path,model,output_file):
    df = pd.read_csv(file_path)
    new_column_names = {'CDR3B': 'tcr',
                    'Epitope': 'antigen',
                   'Affinity':'label'}
    df = df.rename(columns=new_column_names)
    x, y = df[['tcr', 'antigen']], df['label']
    #x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2021, test_size=test_size, stratify=y)
    x_test, y_test=x,y
    test_results = _evaluate_network(model, x_test, y_test,output_file,plot_roc_curve=False)

def _evaluate_network(model, x_test, y_test,output_file,plot_roc_curve=False):
    """
    Evaluate network's performance.
    Parameters
    ----------
    model
    x_test
    y_test
    plot_roc_curve

    Returns
    -------

    """
    x_test, y_test = x_test.values, y_test.values.astype('int')
    test_loader = _prepare_data_loader(x_test, y_test)

    model.eval()
    y_true, y_pred, y_prob = [], [], []
    tcrs_list,antigens_list=[],[]
    for tcrs, antigens, labels, tcrs_seq, antigens_seq in test_loader:
        output = model(tcrs, antigens)
        y_true.extend(labels.numpy())
        y_prob.extend(torch.softmax(output, dim=1)[:, 1].detach().numpy())
        tcrs_list.extend(tcrs_seq)
        antigens_list.extend(antigens_seq)
    y_pred = [1 if prob > 0.5 else 0 for prob in y_prob]
 
    data1 = {'tcr': tcrs_list,'antigen': antigens_list,'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob}
    #pd.DataFrame(data1).to_csv(output_file)
  
    
    if plot_roc_curve:
        _plot_roc_curve(y_true, y_prob)

    return {
        "ACC": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }

def _prepare_data_loader(x, y, batch_size=64, shuffle=True, data_type="aa", embedding_file=None):
    """
    DataLoader for torch
    Parameters
    ----------
    x
    y
    batch_size
    shuffle

    Returns
    -------

    """
    if data_type == "aa":
        dataset = AttnBLSTMDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def _plot_roc_curve(y_true, y_prob):
    """
    Parameters
    ----------
    y_true
    y_prob

    Returns
    -------

    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr)
    plt.show()

def str2bool(str):
	return True if str == 'True' else False

def create_parser():
    parser = argparse.ArgumentParser(description="Dataset build",formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument("--input_file",dest="input_file",type=str,help="The input file in .csv format.",required=True)
    parser.add_argument("--output_file", dest="output_file", type=str, help="The output file in .csv format.",required=True)
    parser.add_argument("--load_model_file", dest="load_model_file", type=str, help="The path to load the model.",required=True)
    parser.add_argument("--plot_roc_curve", dest="plot_roc_curve", type=str2bool,default=False,help="The plot of roc.", required=False)

    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = create_parser()
    test_model(
        file_path=args.input_file,
        load_model_file=args.load_model_file,
        plot_roc_curve=args.plot_roc_curve,
        output_file=args.output_file
    )
    print("The work is done!")
    #python./AttnTAP_code/AttnTAP_test.py --input_file=./AttnTAP_data/McPAS-TCR/mcpas_crossvalid_data/0/mcpas_0.csv --output_file=./AttnTAP_result/cv_0_mcpas_0.csv --load_model_file=./AttnTAP_model/cv_model_0_mcpas_0.pt
