import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from DataProcess import TEPDataset,dropInvalid
from torch.utils.data import DataLoader
from model import TEPCAM
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

batch_size = 128

def train_model(input_file,model_name,epoch,learning_rate,d_model,n_heads,random_seed=1000,GPU_num=0,modelseed=1000):
    device = torch.device(f'cuda:{GPU_num}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():print(f'The code runs on GPU{GPU_num}.') 
    df = pd.read_csv(input_file)
    x, y = df[['TCR', 'epitope']], df['Label']
    
    '''for strict split'''
    #epi = x['epitope']
    tcr = x['TCR']
    gss = GroupShuffleSplit(n_splits=1,train_size=.8,random_state=random_seed)
    for i,(train_idx,_) in enumerate(gss.split(x,y,groups=tcr)):
        x_train,y_train = x.iloc[train_idx],y.iloc[train_idx]
        x_train, y_train = dropInvalid(x_train, y_train) #drop sequence contained invalid char
    dataset_train = TEPDataset(x_train,y_train,align=True)
    train_loader = DataLoader(dataset_train,batch_size,shuffle=True,drop_last=True)
    
    model = TEPCAM(batch_size=batch_size,d_model=d_model,modelseed=modelseed,n_heads=n_heads)
    model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(),lr=learning_rate,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20],gamma=0.25)
    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = Mish()
    loss_fn.to(device)
    train_loss_list = []
    
    for epoch in range(epoch):
        total_loss = 0
        batches = 0
        model.train()
        for tcrs, peptides, labels, _, _ in tqdm(train_loader):
            tcrs,peptides = tcrs.to(device),peptides.to(device)
            _, _, _, _, _, output = model(tcrs,peptides)
            labels = labels.long().to(device)
            optimizer.zero_grad()
            loss = loss_fn(output, labels) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        train_loss = round(total_loss/batches,4)
        train_loss_list.append(train_loss)
        print(f"Loss for Epoch{epoch + 1}:{train_loss}")
        scheduler.step()
    
    save_model_file = f'./model/{model_name}.pt'
    torch.save(model,save_model_file) 
    #Visualization
    if not os.path.exists(f'./result/pic'):
        os.makedirs(f'./result/pic')
    epochs = range(1, len(train_loss_list) + 1)   
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss_list, 'b', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"./result/pic/{model_name}.png")

    
def str2bool(str):
	return True if str == 'True' else False

def create_parser():
    parser = argparse.ArgumentParser(description="Model Training for TEPCAM",formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--input_file",dest="input_file",type=str,help="The input file in .csv format.",required=True)
    parser.add_argument("--model_name",dest="model_name",type=str, help="model name and the prefix for saving model.",required=True)
    parser.add_argument("--epoch",dest="epoch",type=int,help="training epoch",default=30, required=False)
    parser.add_argument("--d_model",dest="d_model",type=int,help="The dimension of model",default=32, required=False)
    parser.add_argument("--n_heads",dest="n_heads",type=int,help="Number of heads for attention module",default=6, required=False)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float,default=0.0005, required=False)
    parser.add_argument("--random_seed",dest="random_seed",type=int,default=1000,help="Random Seed used in this trial")
    parser.add_argument("--GPU_num",type=int, dest="GPU_num", default=0, help="which GPU to run the code",required=False)
    parser.add_argument("--modelseed",type=int, dest="modelseed", default=1000, help="seed for reproductbility",required=False)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = create_parser() 
    train_model(
        input_file = args.input_file,
        model_name = args.model_name,
        epoch = args.epoch,
        learning_rate = args.learning_rate,
        d_model = args.d_model,
        n_heads = args.n_heads,
        random_seed = args.random_seed,
        # EarlyStopping = args.EarlyStopping,
        GPU_num = args.GPU_num,
        modelseed = args.modelseed
        ) 
    print(f"The model training is done! Model saved in ./model/{args.model_name}.pt")
    
    


    