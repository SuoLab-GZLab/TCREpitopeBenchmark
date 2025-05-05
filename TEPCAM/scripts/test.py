import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from DataProcess import TEPDataset,dropInvalid
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, precision_recall_curve, roc_curve, auc
from tqdm import tqdm


save_attn = False #Set save_attn=True if attn matrix is in need
batch_size = 128
n_heads = 6
tcr_length = 20
pep_length = 15

def test_model(file_path,model_path,output_file,metric_file,GPU_num=0):
    device = torch.device(f'cuda:{GPU_num}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():print(f'Testing model on GPU{GPU_num}.') 
    df = pd.read_csv(file_path)
    x, y = df[['TCR', 'epitope']], df['Label']
    x, y = dropInvalid(x, y)
    dataset_test = TEPDataset(x,y,align=False)
    test_loader = DataLoader(dataset_test,batch_size=batch_size,shuffle=False,drop_last=True) 
    model_name = model_path.split('/')[-1].split('.')[0]
    model = torch.load(model_path,map_location='cpu')
    model.to(device)
    model.eval()
    y_true, y_prob = [], []
    tcrs_list,epitope_list=[],[]
    if save_attn == True:
        attn_tcr_list, attn_pep_list,attn_tcr_calist, attn_pep_calist = torch.empty([0,n_heads//2,tcr_length,tcr_length]), torch.empty([0,n_heads//2,pep_length,pep_length]), torch.empty([0,n_heads,tcr_length,pep_length]),torch.empty([0,n_heads,pep_length,tcr_length])
    for tcr, peptide, labels, tcr_seq, pep_seq in tqdm(test_loader):
        tcr, peptide = tcr.to(device), peptide.to(device)
        attn_tcr, attn_pep, attn_tcr_ca, attn_pep_ca, fusedoutput, output = model(tcr,peptide)
        labels = labels.long()
        y_true.extend(labels.numpy())
        y_prob.extend(torch.softmax(output, dim=1)[:, 1].to('cpu').detach().numpy())
        tcrs_list.extend(tcr_seq)
        epitope_list.extend(pep_seq)
        if save_attn == True:
            if not os.path.exists(f'./attn/pt'):
                os.makedirs(f'./attn/pt')
            attn_tcr_list = torch.cat((attn_tcr_list,attn_tcr.to('cpu').detach()),dim=0)
            attn_pep_list = torch.cat((attn_pep_list,attn_pep.to('cpu').detach()),dim=0)
            attn_tcr_calist = torch.cat((attn_tcr_calist,attn_tcr_ca.to('cpu').detach()),dim=0)
            attn_pep_calist = torch.cat((attn_pep_calist,attn_pep_ca.to('cpu').detach()),dim=0)
            if "HQ" in file_path:
                torch.save(attn_tcr_list,f"./attn/pt/tcr_sa_{model_name}_HQ.pt")
                torch.save(attn_pep_list,f"./attn/pt/pep_sa_{model_name}_HQ.pt")
                torch.save(attn_tcr_calist,f"./attn/pt/tcr_ca_{model_name}_HQ.pt")
                torch.save(attn_pep_calist,f"./attn/pt/pep_ca_{model_name}_HQ.pt")
            else:
                torch.save(attn_tcr_list,f"./attn/pt/tcr_sa_{model_name}_{file_path.split('.')[0]}_test.pt")
                torch.save(attn_pep_list,f"./attn/pt/pep_sa_{model_name}_{file_path.split('.')[0]}_test.pt")
                torch.save(attn_tcr_calist,f"./attn/pt/tcr_ca_{model_name}_{file_path.split('.')[0]}_test.pt")
                torch.save(attn_pep_calist,f"./attn/pt/pep_ca_{model_name}_{file_path.split('.')[0]}_test.pt")
    y_pred = [1 if prob > 0.5 else 0 for prob in y_prob]
    output_data = {'tcr': tcrs_list, 'epitope': epitope_list, 'y_prob': y_prob, 'y_pred': y_pred, 'Label':y_true}
    df = pd.DataFrame(output_data)
    df.to_csv(output_file)

    #Metrics
    ACC = accuracy_score(y_true, y_pred)
    AUC = roc_auc_score(y_true, y_prob)
    Recall = recall_score(y_true, y_pred)
    Precision = precision_score(y_true, y_pred)
    F1 = f1_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    AUPR = auc(recall, precision)
    
    write_to_csv(metric_file,"-----------------{}--------------\n".format(model_path))
    write_to_csv(metric_file,"ACC,{:.3f}\n".format(ACC))
    write_to_csv(metric_file,"AUC,{:.3f}\n".format(AUC))
    write_to_csv(metric_file,"AUPR,{:.3f}\n".format(AUPR))
    write_to_csv(metric_file,"Recall,{:.3f}\n".format(Recall))
    write_to_csv(metric_file,"Precision,{:.3f}\n".format(Precision))
    write_to_csv(metric_file,"F1,{:.3f}\n".format(F1))
    

def write_to_csv(file_path,result):
    f = open(file_path, "a", encoding="gbk", newline="")
    f.write(result)
    f.close()       
    
def create_parser():
    parser = argparse.ArgumentParser(description="Predict binding by TEPCAM",formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--file_path",dest="file_path",type=str,help="The input test file in .csv format.",required=True)
    parser.add_argument("--model_path",dest="model_path",type=str, help="Model path.",required=True)
    parser.add_argument("--output_file",dest="output_file",type=str,help="File for saving the output result",required=False)
    parser.add_argument("--metric_file", dest="metric_file", type=str,required=False)
    args = parser.parse_args()
    return args
    
if __name__=="__main__":
    args = create_parser()
    test_model(
        file_path = args.file_path,
        model_path = args.model_path,
        output_file = args.output_file,
        metric_file = args.metric_file
        )  
    print(f"Model test finish!")
    
