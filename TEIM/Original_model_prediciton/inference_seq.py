import numpy as np
import pandas as pd
import os
import sys
sys.path.append('.')

import torch
import pytorch_lightning as pl
from data_process import Tokenizer, PretrainedEncoder, encode_cdr3
from models import NewCNN2dBaseline
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score,precision_score,f1_score,matthews_corrcoef,roc_curve,auc
import numpy as np


import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BdTrainerSystem(pl.LightningModule):
    def __init__(self, data_mode, model_args, lr, weight_decay, model=None, setting='', *args, **kwargs):
        super().__init__()
        self.data_mode = data_mode
        self.save_hyperparameters('data_mode', 'model_args', 'lr', 'weight_decay')
        if model is None:
            self.model = NewCNN2dBaseline(model_types=data_mode['model_type'].split('|'), **model_args)
        else:
            self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.train()
        loss, y, y_hat, mask = self.minimum_step(batch)
        self.log('train_loss', loss)
        return {
            'loss': loss,
            'y_hat': y_hat,
            'y': y
        }

    def training_epoch_end(self, training_step_outputs):
        loss, scores, _, _, _ = self.evaluate_model(self.val_dataloader())
        self.log('val_loss', loss)
        self.log_dict({
            'val_corr': scores[0],
            'val_mse': scores[1],
            'val_mape': scores[2],
            'val_auc': scores[3],
        })
        self.scores_epochs.append(scores)
        # self.mse_epochs.append(mse)

    def minimum_step(self, batch):
        x, y = batch[0:3], batch[3].float().to(self.device)
        mask = batch[4].float().to(self.device)
        y_hat = self([x_.to(self.device) for x_ in x])

        y_bind_hat = y_hat
        y_bind_true = y

        loss = self.get_loss(y_bind_hat, y_bind_true, mask)
        return loss, y_bind_true, y_bind_hat, mask

    def evaluate_model(self, data_loader=None, plot_model=False, device=None):
        if device is not None:
            self.to(device)
        else:
            device = self.device
        self.eval()

        y_true = []
        y_pred = []
        y_mask = []
        for i, batch in enumerate(data_loader):
            # x.extend(batch[1].detach().cpu().numpy())
            loss, y, y_hat, mask = self.minimum_step(batch)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_hat.detach().cpu().numpy())
            y_mask.extend(mask.detach().cpu().numpy())

        scores, scores_samples = self.get_scores(y_true, y_pred, y_mask)
        return loss, scores, scores_samples, y_true, y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return {
                'optimizer': optimizer,
                'lr_scheduler': StepLR(optimizer, 200, gamma=0.5, last_epoch=-1, ),
                }

    def get_loss(self, y_hat, y, ):
        loss = F.binary_cross_entropy(y_hat, y, weight=None, reduction='mean')
        return loss

    def get_scores(self, y_true, y_pred, y_mask):
        auc = roc_auc_score(y_true, y_pred)
        aupr = average_precision_score(y_true, y_pred)
        return auc, aupr


class BindPredictor:
    def __init__(self, path='./ckpt/teim_seq.ckpt'):
        self.model = BdTrainerSystem.load_from_checkpoint(path)
        self.model = self.model.to(device)
        self.model.eval()
        self.tokenizer = Tokenizer()

    def predict(self, seqs_cdr3, seqs_epi):
        encoding_cdr3, encoding_epi, epi_vec = self.encode_data(seqs_cdr3, seqs_epi)
        predictions = self.forward(encoding_cdr3, encoding_epi, epi_vec)
        return predictions

    def encode_data(self, seqs_cdr3, seqs_epi):
        encoding_cdr3 = encode_cdr3(seqs_cdr3, self.tokenizer)
        epi_encoder = PretrainedEncoder(self.tokenizer)
        encoding_epi, epi_vec = epi_encoder.encode_pretrained_epi(seqs_epi)

        return [encoding_cdr3, encoding_epi, epi_vec]

    def forward(self, encoding_cdr3, encoding_epi, epi_vec):
        bs = 128
        pred = []
        for batch in np.arange(0, len(encoding_cdr3) // bs + 1):
            idx_end = min((batch+1)*bs, len(encoding_cdr3))
            input_cdr3 = torch.Tensor(encoding_cdr3[batch*bs:idx_end, ...]).to(device)
            input_epi = torch.Tensor(encoding_epi[batch*bs:idx_end, ...]).to(device)
            input_vec = torch.Tensor(epi_vec[batch*bs:idx_end, ...]).to(device)
            pred_batch = self.model([input_cdr3, input_epi, input_vec])
            pred_batch = pred_batch.cpu().detach().numpy()
            pred.extend(pred_batch)
        return np.array(pred)


def get_seqs(input_path):
    df = pd.read_csv(input_path)
    #seqs_cdr3 = df['cdr3'].values
    #seqs_epi = df['epitope'].values
    seqs_cdr3 = df['CDR3B'].values
    seqs_epi = df['Epitope'].values
    y_true=df['Affinity'].values
    return seqs_cdr3, seqs_epi,y_true


# def predict_binding(input_path, save_dir='./outputs', batch_size=128):
#     # # load data
#     seqs_cdr3, seqs_epi = get_seqs(input_path)

#     # # build predictor
#     predictor = BindPredictor()
#     pred_list = []
#     for i in range(0, len(seqs_epi), batch_size):
#         print('Predict batch', i // batch_size)
#         # # get batch data
#         idx_end = min(i+batch_size, len(seqs_epi))
#         seqs_cdr3_batch = seqs_cdr3[i:idx_end]
#         seqs_epi_batch = seqs_epi[i:idx_end]

#         # # predict distance and site
#         pred = predictor.predict(seqs_cdr3_batch, seqs_epi_batch)[:, 0]
#         pred_list.extend(pred)

#     # # save
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     df = pd.DataFrame(dict(cdr3=seqs_cdr3, epitope=seqs_epi, binding=pred_list))
#     df.to_csv(os.path.join(save_dir, 'sequence_level_binding.csv'))
#     print('Done. The predictions are in', save_dir)

# if __name__ == '__main__':
#     input_path ='./inputs/inputs_bd.csv'
#     output_dir = './outputs'
#     predict_binding(input_path, output_dir)
    
    
    
def predict_binding(testfile_path, result_path, batch_size=128):
    # # load data
    seqs_cdr3, seqs_epi,y_true = get_seqs(testfile_path)
    # # build predictor
    predictor = BindPredictor()
    pred_list = []
    for i in range(0, len(seqs_epi), batch_size):
        print('Predict batch', i // batch_size)
        # # get batch data
        idx_end = min(i+batch_size, len(seqs_epi))
        seqs_cdr3_batch = seqs_cdr3[i:idx_end]
        seqs_epi_batch = seqs_epi[i:idx_end]

        # # predict distance and site
        pred = predictor.predict(seqs_cdr3_batch, seqs_epi_batch)[:, 0]
        pred_list.extend(pred)

    # # save
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    df = pd.DataFrame(dict(cdr3=seqs_cdr3, epitope=seqs_epi, binding=pred_list))
    df['y_prob']=df['binding']
    df['y_true']=y_true
    
    df['y_pred'] = (df['y_prob'] >= 0.5).astype(int)
    df.rename(columns={'Affinity': 'y_true'}, inplace=True)
    metrics = []
    fpr_all = []
    tpr_all = []
    roc_auc_all = []
    for cls in df['epitope'].unique():
        class_df = df[df['epitope'] == cls]
        y_true = class_df['y_true']
        y_pred = class_df['y_pred']
        y_prob = class_df['y_prob']
        acc = accuracy_score(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_prob)if len(np.unique(y_true)) != 1 else 0
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)  # Added precision calculation
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)if len(np.unique(y_true)) != 1 else 0
        metrics.append({
            'Class': cls,
            'ACC': acc,
            'AUC': auc_score,
            'Recall': recall,
            'Precision': precision,  # Added Precision metric
            'F1': f1,
            'MCC': mcc
        })
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fpr_all.append(fpr)
        tpr_all.append(tpr)
        roc_auc_all.append(roc_auc)
    metrics_class = pd.DataFrame(metrics)
    # Calculate overall metrics
    y_true_all = df['y_true']  # Modified from df to result
    y_pred_all = df['y_pred']  # Modified from df to result
    y_prob_all = df['y_prob']  # Modified from df to result

    acc_all = accuracy_score(y_true_all, y_pred_all)
    auc_all = roc_auc_score(y_true_all, y_prob_all)
    recall_all = recall_score(y_true_all, y_pred_all)
    precision_all = precision_score(y_true_all, y_pred_all)  # Added Precision calculation
    f1_all = f1_score(y_true_all, y_pred_all)
    mcc_all = matthews_corrcoef(y_true_all, y_pred_all)
    metrics_all = pd.DataFrame({
        'Metric': ['ACC', 'AUC', 'Recall', 'Precision', 'F1', 'MCC'],
        'Value': [acc_all, auc_all, recall_all, precision_all, f1_all, mcc_all]})
    all_value_row = pd.DataFrame({
        'Class': ['all_Value'],
        'ACC': [metrics_all.loc[0, 'Value']],
        'AUC': [metrics_all.loc[1, 'Value']],
        'Recall': [metrics_all.loc[2, 'Value']],
        'Precision': [metrics_all.loc[3, 'Value']],
        'F1': [metrics_all.loc[4, 'Value']],
        'MCC': [metrics_all.loc[5, 'Value']]
    })
    metrics_class = pd.concat([metrics_class, all_value_row], ignore_index=True)
    metrics_class.to_csv(result_path + 'all_result.csv')
    df.rename(columns={'epitope': 'Epitope','cdr3': 'CDR3B'}, inplace=True)
    result=df[['Epitope','CDR3B','y_true','y_prob','y_pred']]
    result.to_csv(result_path +'probability.csv')
   
    
name=['1','2','3','4','5','6']
me=['all','withCF']
epi=['_four','_three']
for i in name:
    for j in me:
        for k in epi:
            testfile_path="/home/luyanping/data/TCR_epitope_prediction/Compare_models_same_data/database/benchmark_dataset/10cross_validation/Prediction/five/"+j+'/'+i+k+'.csv'
            result_path="/home/luyanping/data/TCR_epitope_prediction/Compare_models_same_data/TEIM/evaluate/repeat10/Prediction/repeat/"+j+'/'+i+k
            predict_binding(testfile_path,result_path)     
    
    
    
    
