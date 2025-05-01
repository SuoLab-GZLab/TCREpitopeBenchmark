from tcrpeg.TCRpeg import TCRpeg
import pandas as pd
import numpy as np
import argparse
import torch.optim as optim
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import average_precision_score as AUPRC
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sampler import Sampler
from model import TEINet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dropout',type=float,default=0.0)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--cat_size',type=int,default=1536,help='the size of the concatenated embedding')
    parser.add_argument('--batch_size',type=int,default=48)
    parser.add_argument('--sample_num',type=int,default=10, help='The number of negative samples for a given positive pair')
    parser.add_argument('--fre',type=int,default=1,help='1: sample epitope based on frequency distribution; 0: randomly')
    parser.add_argument('--info',type=str,default='information',help='the information text in the output file')
    parser.add_argument('--reference_tcr',type=str,default='None',help='Path to reference TCR if using Reference TCR for negative sampling')
    parser.add_argument('--aa_tcr',type=str,default='encoders/aa_emb_tcr.txt', help='path to the aa embedding for TCRs')
    parser.add_argument('--aa_epi',type=str,default= 'encoders/aa_emb_epi.txt',help='path to the aa embedding for epitopes')
    parser.add_argument('--pretrain_tcr',type=str,default='encoders/encoder_tcr.pth',help='path to the pretrained TCRpeg for TCRs')
    parser.add_argument('--pretrain_epi',type=str,default='encoders/encoder_epi.pth',help='path to the pretrained TCRpeg for epitopes')
    parser.add_argument('--weight',type=float,default=4,help='class weight for positive data') 
    parser.add_argument('--weight_decay',type=float,default=0.0) 
    parser.add_argument('--normalize',type=int,default=1,help='whether to use layer norm;1 to use and 0 not to use')   
    parser.add_argument('--step1',type=int,default=21,help='the epoch that reduces the learning rate')  
    parser.add_argument('--step2',type=int,default=27)  
    parser.add_argument('--step3',type=int,default=30)              
    parser.add_argument('--lr',type=float,default=0.001)                 
    parser.add_argument('--record_path',type=str,default='results/predictions.txt',help='record the predictions for test set')   
    parser.add_argument('--output_path',type=str,default='results/training.txt',help='record the information text and the test AUROC to this file')   
    parser.add_argument('--model_path',type=str,default='None',help='path to store the model')
    parser.add_argument('--train_file',type=str,default='data/train_pos.csv', help='Training set. Note that it should contain only the positive pairs') 
    parser.add_argument('--static',type=int,default=0,help='whether to use static dataset, i.e., sample negatives before the training process')   
    parser.add_argument('--test_file',type=str,default='None',help='The path to the test file.')
    parser.add_argument('--pretrain',action='store_false',default=True,help='whether to use the pretrained encoder')    
    parser.add_argument('--sample_strategy',type=str,default='sample_epi',help='The negative sampling strategy used; options=[\'sample_epi\',\'sample_tcr\']')    
    '''
    The four negative sampling strategies, you need to specify the following settings:
    @Random Epitope: --sample_strategy sample_epi --fre 0
    @Uniform Epitope: --sample_strategy sample_epi --fre 1
    @Reference TCR: --sample_strategy sample_tcr --reference_tcr path_to_reference_tcr
    @Random TCR: --sample_strategy sample_tcr
    '''
    args = parser.parse_args()
    fre = True if args.fre == 1 else False    
    normalize = False if args.normalize == 0 else True     
    hidden_size = args.cat_size // 2 
    cat_size = args.cat_size        
    static = True if args.static == 1 else False

    model_tcr = TCRpeg(hidden_size=hidden_size,num_layers = 3,load_data=False,embedding_path=args.aa_tcr)
    #load pretrained TCR model
    if args.pretrain:
        print('using the pretrained model')
        model_tcr.create_model(load=True,path=args.pretrain_tcr)
    else :
        model_tcr.create_model()
    model_tcr.model.train()

    #construct TCRpeg for epi
    model_epi = TCRpeg(hidden_size=hidden_size,num_layers = 3,load_data=False,embedding_path=args.aa_epi)
    if args.pretrain:                             
        model_epi.create_model(load=True,path = args.pretrain_epi)
    else :
        model_epi.create_model()
    model_epi.model.train()


    sampler = Sampler()
    sampler.construct_neg(args.train_file,static)              
    if args.reference_tcr != 'None':
        assert args.sample_strategy == 'sample_tcr', 'Sampling method should be sample tcr'
        sampler.reference_tcr = pd.read_csv(args.reference_tcr)['CDR3.beta'].values        

    dropout = args.dropout
    cat_size = args.cat_size
    model = TEINet(en_tcr=model_tcr,en_epi = model_epi,cat_size=cat_size,dropout=dropout,normalize=normalize,weight_decay = args.weight_decay).to('cuda:0')
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if args.test_file != 'None':
        data = pd.read_csv(args.test_file)   
        cs_test,es_test,ls_test = data['CDR3.beta'].values,data['Epitope'].values,data['Label'].values
    pos_weight = (torch.ones([1])*args.weight).to('cuda:0')    
    loss_fcn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    epochs = args.epochs

    print('Begin training')
    batch_size = args.batch_size
    sample_num = args.sample_num    
    print('Epoch = {}'.format(epochs))
    print('sample num = {}'.format(sample_num))
    print('dropout = {}'.format(dropout))
    print('Using frequency to sample = {}'.format(str(fre)))
    print('cat_size = {}'.format(cat_size))
    # batch_num_total = len(cs_train) // batch_size if len(cs_train) % batch_size == 0 else len(cs_train) // batch_size + 1    
    record_aucs = []
    for e in range(epochs):
        batch_num_total = len(sampler.tcrs) // batch_size if len(sampler.tcrs) % batch_size == 0 else len(sampler.tcrs) // batch_size + 1
        # batch_num_total = len(sampler.tcrs) // batch_size if len(sampler.tcrs) % batch_size == 0 else len(sampler.tcrs) // batch_size + 1
        infer = np.random.permutation(len(sampler.tcrs))
        # tcrs,epitopes = sampler.tcrs[infer],sampler.epitopes[infer]
        tcrs,epitopes = sampler.tcrs[infer],sampler.epitopes[infer] #shuffle
        if static:
            print('enter static')
            labels = sampler.labels[infer]
        # labels = ls_train[infer]
        for batch_num in tqdm(range(batch_num_total)):
            end = (batch_num+1)*batch_size if (batch_num+1)*batch_size <= len(tcrs) else len(tcrs)                
            ts,es = tcrs[batch_num*batch_size:end],epitopes[batch_num * batch_size:end]             
            if static: 
                ls = labels[batch_num*batch_size:end]                
            else :     
                if args.sample_strategy == 'sample_epi':
                    if fre:
                        ts,es,ls = sampler.sample_neg_fre(ts,es,sample_num)   #by fre 
                    else :                
                        ts,es,ls = sampler.sample_neg_whole(ts,es,sample_num) #randomly
                elif args.sample_strategy == 'sample_tcr':                      
                    if args.reference_tcr != 'None':
                        ts,es,ls = sampler.sample_neg_tcr(ts,es,sample_num,True) #Reference TCR                   
                    else :                        
                        ts,es,ls = sampler.sample_neg_tcr(ts,es,sample_num,False) # Random TCR           
            ls = torch.FloatTensor(ls).to('cuda:0')  
            output = model(ts,es)

            if args.weight_decay == 0.0:
                loss = loss_fcn(output.view(-1),ls)
            else :
                loss = loss_fcn(output[0].view(-1),ls) + output[1]   

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #calculate AUC  
        # batch_size = 64
        y_pres = []
        y_trues = []
        # batch_num = len(cs_train) // batch_size
        if args.test_file != 'None':
            with torch.no_grad():
                y_pres = []
                y_trues = []
                batch_num = len(cs_test) // batch_size if len(cs_test) % batch_size == 0 else len(cs_test) // batch_size + 1        
                for i in tqdm(range(batch_num)):
                    end = (i+1)*batch_size if (i+1)*batch_size <= len(cs_test) else len(cs_test)                
                    cs_batch,es_batch = cs_test[i*batch_size :end], es_test[i*batch_size : end]
                    score = model(cs_batch,es_batch) 
                    if args.weight_decay !=0.0:
                        score = score[0]           
                    y_pres.extend(score.view(-1).detach().cpu().numpy())  
                    y_trues.extend(ls_test[i*batch_size : end])
                test_auc = AUC(y_trues,y_pres)            
                print('Epoch: ',e)
                print('Test AUC: ',AUC(y_trues,y_pres))
                record_aucs.append(AUC(y_trues,y_pres))
            with open(args.output_path,'a') as f:
                f.write('\n')
                f.write(args.info + '\n')
                f.write(str(test_auc)) 

        if e == args.step1 or e == args.step2:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.01
            print('change the learning rate to 1e-4')
        if e == args.step3:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.02
            print('change')



    if args.model_path != 'None':
        if not args.model_path.endswith('.pth'):
            args.model_path = args.model_path + '.pth'
        torch.save(model.state_dict(),args.model_path)
    if args.record_path != 'None' and args.test_file != 'None':
        with open(args.record_path,'w') as f:
            for i in range(len(y_trues)):
                f.write(str(y_pres[i]) + ',' + str(y_trues[i]) + '\n')

