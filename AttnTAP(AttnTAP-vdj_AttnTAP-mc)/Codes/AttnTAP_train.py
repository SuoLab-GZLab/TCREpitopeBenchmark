import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
from network import AttentionBiLSTMClassifier, AttnBLSTMDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve
torch.manual_seed(2021)
np.random.seed(2021)


def train_model(file_path, epoch,valid_set, plot_train_curve, learning_rate, embedding_dim, dropout_rate,hidden_dim,plot_roc_curve, pretrained_embedding=None,save_model_file=None):
    df = pd.read_csv(file_path)
    new_column_names = {'CDR3B': 'tcr',
                    'Epitope': 'antigen',
                   'Affinity':'label'}
    df = df.rename(columns=new_column_names)
    x, y = df[['tcr', 'antigen']], df['label']
    #x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2021, test_size=test_size, stratify=y)
    x_train, y_train=x,y
    model = _train_network(
        x_train, y_train,
        epoch=epoch,
        learning_rate=learning_rate,
        valid_set=valid_set,
        plot_train_curve=plot_train_curve,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        bidirectional=True,
        hidden_dim=hidden_dim,
        pretrained_embedding=pretrained_embedding
    )
    if save_model_file:
        torch.save(model, save_model_file)
    train_results = _evaluate_network(model, x_train, y_train,plot_roc_curve=plot_roc_curve)
    print("-----train results-----")
    print("ACC:\t\t", '{:.3f}'.format(train_results['ACC']))
    print("AUC:\t\t", '{:.3f}'.format(train_results['AUC']))
    print("Recall:\t\t", '{:.3f}'.format(train_results['Recall']))
    print("Precision:\t", '{:.3f}'.format(train_results['Precision']))
    print("F1:\t\t", '{:.3f}'.format(train_results['F1']))


def _train_network(x_train, y_train, epoch, learning_rate, valid_set=False, plot_train_curve=False,
                   embedding_dim=10, dropout_rate=0.1, bidirectional=True, hidden_dim=50,
                   pretrained_embedding=None):
    """
    Train the TCR-Antigen network.
    Parameters
    ----------
    x_train: Training dataset.
    y_train: Labels for x_train
    aa2idx: Dictionary that maps amino acid into index.
    epoch: Training epoch.
    learning_rate: Learning rate.
    valid_set: Boolean, whether split the X_train into X_train' and X_valid. The default value is False.

    Returns
    -------

    """
    x_train, y_train = x_train.values, y_train.values.astype('int')

    if valid_set:
        # Split the whole training set into training set and validation set.
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=0.2, random_state=2021, stratify=y_train)
        train_loader = _prepare_data_loader(x_train, y_train)
        valid_loader = _prepare_data_loader(x_valid, y_valid)
    else:
        train_loader = _prepare_data_loader(x_train, y_train)

    model = AttentionBiLSTMClassifier(
        embedding_dim=embedding_dim,
        bidirectional=bidirectional,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim,
        pretrained_embeddings=pretrained_embedding
    )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    train_loss_list, test_loss_list = [], []
    for epoch in range(epoch):
        total_loss = 0
        batches = 0
        model.train()
        for tcrs, antigens, labels, tcrs_seq, antigens_seq in train_loader:
            output = model(tcrs, antigens)
            labels = labels.long()
            loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        train_loss = round(total_loss / batches, 4)
        train_loss_list.append(train_loss)
        print("Epoch-{} Train loss {}".format(epoch + 1, train_loss))
        # scheduler.step()

        if valid_set:
            # evaluate loss on test data.
            total_loss = 0
            batches = 0
            model.eval()
            for tcrs, antigens, labels, tcrs_seq, antigens_seq in valid_loader:
                output = model(tcrs, antigens)
                labels = labels.long()
                loss = loss_fn(output, labels)

                total_loss += loss.item()
                batches += 1
            test_loss = round(total_loss / batches, 4)
            test_loss_list.append(test_loss)
            print("Epoch-{} Test loss {}".format(epoch + 1, test_loss))

    if plot_train_curve:
        _plot_train_curve(train_loss_list, test_loss_list)
    return model


def _evaluate_network(model, x_test, y_test, plot_roc_curve=False):
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
    for tcrs, antigens, labels, tcrs_seq, antigens_seq in test_loader:
        output = model(tcrs, antigens)
        y_true.extend(labels.numpy())
        y_prob.extend(torch.softmax(output, dim=1)[:, 1].detach().numpy())

    y_pred = [1 if prob > 0.5 else 0 for prob in y_prob]

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
    dataset = AttnBLSTMDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def _plot_train_curve(train_loss_list, test_loss_list):
    """
    Plot the train and valid curve
    Parameters
    ----------
    train_loss_list
    test_loss_list

    Returns
    -------

    """
    # Plot loss curve
    plt.plot(np.arange(len(train_loss_list)), train_loss_list, marker="s", label="Train Loss")
    plt.plot(np.arange(len(test_loss_list)), test_loss_list, marker="o", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

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
    parser.add_argument("--save_model_file", dest="save_model_file", type=str, help="The path to save the model.",required=False)
    parser.add_argument("--valid_set",dest="valid_set",type=str2bool,default=False,help="The valid set.",required=False)
    parser.add_argument("--epoch",dest="epoch",type=int,help="epoch",default=15, required=False)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float,default=0.005, required=False)
    parser.add_argument("--dropout_rate", dest="dropout_rate", type=float, default=0.1, required=False)
    parser.add_argument("--embedding_dim", dest="embedding_dim", type=int,default=10, required=False)
    parser.add_argument("--hidden_dim", dest="hidden_dim", type=int,default=50,required=False)
    parser.add_argument("--plot_train_curve", type=str2bool,dest="plot_train_curve", default=False,help="The plot of training loss.", required=False)
    parser.add_argument("--plot_roc_curve",type=str2bool, dest="plot_roc_curve", default=False, help="The plot of roc.",required=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_parser()
    train_model(
        file_path=args.input_file,
        save_model_file=args.save_model_file,
        epoch=args.epoch,
        valid_set=args.valid_set,
        plot_train_curve=args.plot_train_curve,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        dropout_rate=args.dropout_rate,
        hidden_dim=args.hidden_dim,
        plot_roc_curve=args.plot_roc_curve
        )
    print("The work is done!")
#python  ./AttnTAP_code/AttnTAP_train.py --input_file=./AttnTAP_data/McPAS-TCR/mcpas_train.csv --save_model_file=./AttnTAP_model/mcpas_train.pt --valid_set=False --epoch=15 --learning_rate=0.005 --dropout_rate=0.1 --embedding_dim=10 --hidden_dim=50 --plot_train_curve=False --plot_roc_curve=False

