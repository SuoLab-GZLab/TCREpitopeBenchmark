import torch
import torch.nn as nn 
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from tqdm import tqdm
import numpy as np

#note that, the padding_idx is 0, the idx for START token is 1
#the idx for AAs start from 2 and the idx for END stoken is 22


class TCRpeg_model(nn.Module):
    def __init__(self,embedding_layer,embedding_size,hidden_size,dropout,max_length=30,sos_idx=1,pad_idx=0,eos_idx=22,device='cuda:0',num_layers=1):
        super(TCRpeg_model,self).__init__()
        self.max_length = max_length
        self.sos_idx=sos_idx
        self.eos_idx=  eos_idx
        self.pad_idx = pad_idx
        # self.embedding_dropout= embedding_dropout
        self.embedding = embedding_layer
        self.embedding_size=  embedding_size
        self.hidden_size=hidden_size
        
        self.decoder_rnn = nn.GRU(embedding_size,hidden_size,num_layers,dropout=dropout)
        self.num_layers = num_layers
        self.out_layer=  nn.Linear(hidden_size, 22)
        self.device = device        
    
    def forward(self,seqs,length,need_hidden=False):
        #note that, for the RNN layer in torch, the batch_num is not 
        #at the first dimension (batch_first=False), which results in those permute' below
        self.batch_size = seqs.size(0)
        sorted_lengths, sorted_idx = torch.sort(length,descending=True) #sorting is used for pack_padded_sequence
      
        seqs = seqs[sorted_idx]
      
        input_emb = self.embedding(seqs).to(torch.float32).permute([1,0,2]) 
   
        packed_input = pack_padded_sequence(input_emb,sorted_lengths.data.tolist())

        outputs,h_n= self.decoder_rnn(packed_input)
        
        padded_outputs = pad_packed_sequence(outputs)[0].permute([1,0,2]) 
        h_n = h_n.permute([1,0,2])

        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        h_n = h_n[reversed_idx] 
    
        logp = nn.functional.log_softmax(self.out_layer(padded_outputs),dim=-1) # batch_size x max_length x 22
        # print(length)
        # print(logp[:,length-1,:])
        # print(logp[:,length-1,:].size())
        # exit()
        #logp = logp.permute([0,2,1]) #batch_size x 22 x max_length
        if need_hidden:
            return logp,h_n
        else :
            return logp

class TCRpeg_vj_model(nn.Module):
    def __init__(self,embedding_layer,embedding_size,hidden_size,dropout,max_length=30,sos_idx=1,pad_idx=0,eos_idx=22,device='cuda:0',num_layers=1,num_v=43,num_j=12):
        super(TCRpeg_vj_model,self).__init__()
        self.max_length = max_length
        self.sos_idx=sos_idx
        self.eos_idx=  eos_idx
        self.pad_idx = pad_idx
        # self.embedding_dropout= embedding_dropout
        self.embedding = embedding_layer
        self.embedding_size=  embedding_size
        self.hidden_size=hidden_size
        self.decoder_rnn = nn.GRU(embedding_size,hidden_size,num_layers,dropout=dropout)
        self.num_layers = num_layers
        self.out_layer=  nn.Linear(hidden_size, 22)
        self.device = device
        #self.v_layer = nn.Sequential(nn.Linear(hidden_size * num_layers,128),nn.ReLU(),nn.Linear(128,num_v)) #try a simple one first
        #self.j_layer = nn.Sequential(nn.Linear(hidden_size * num_layers,64),nn.ReLU(),nn.Linear(64,num_j))
        self.v_layer = nn.Sequential(nn.Linear(hidden_size * num_layers,num_v)) #try a simple one first
        self.j_layer = nn.Sequential(nn.Linear(hidden_size * num_layers,num_j))
    def forward(self,seqs,length,need_hidden=False):        
        self.batch_size = seqs.size(0)
        sorted_lengths, sorted_idx = torch.sort(length,descending=True)


        seqs = seqs[sorted_idx]
    
        input_emb = self.embedding(seqs).to(torch.float32).permute([1,0,2])
   
        packed_input = pack_padded_sequence(input_emb,sorted_lengths.data.tolist())

        outputs,h_n= self.decoder_rnn(packed_input)
        #h_n is L x B x Hout
      

        padded_outputs = pad_packed_sequence(outputs)[0].permute([1,0,2]) 
        h_n = h_n.permute([1,0,2]) #B x layer X Hout
        #print(padded_outputs.size()) #batch_size x max_length x hidden_size
        #padded_outputs = padded_outputs.contiguous() # check this 
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        h_n = h_n[reversed_idx] 
    
        logp = nn.functional.log_softmax(self.out_layer(padded_outputs),dim=-1) # batch_size x max_length x 22

        #logp = logp.permute([0,2,1]) #batch_size x 22 x max_length        
        h_n_ = h_n.view(self.batch_size,-1)
    
        v_pre = self.v_layer(h_n_) #check the size of h_n
        j_pre = self.j_layer(h_n_)
        if need_hidden:
            return logp,v_pre,j_pre,h_n
        else :
            return logp,v_pre,j_pre

#Models below are used for classification task (to show the usefulness of the embedding of tcrpeg)
class FC_NN_small(nn.Module):
    def __init__(self,embedding_size,last_layer=True,device='cuda:0'):
        #the model should be a vae_nlp model
        super(FC_NN_small,self).__init__()
        #self.vae_model= model
        #self.vae_model.model.eval()
        self.embeeding_size=  embedding_size
        self.device= device
        self.last_layer = last_layer #whether to use the last layer of GRU hidden units
        self.f1 = nn.Linear(embedding_size,1)
        self.embedding_size = embedding_size
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,embedding):
        return self.sigmoid(self.f1(embedding))


class FC_NN_medium(nn.Module):
    def __init__(self,embedding_size,dropout,last_layer=True,device='cuda:0',batch_norm=False):

        super(FC_NN_medium,self).__init__()
        self.embeeding_size=  embedding_size
        self.dropout = nn.Dropout(p=dropout)
        self.device= device
        self.last_layer = last_layer #whether to use the last layer of GRU hidden units
        self.f1 = nn.Linear(embedding_size,embedding_size//4)
        self.relu = nn.ReLU()
        self.f2 = nn.Linear(embedding_size // 4,1)
        self.embedding_size = embedding_size
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(embedding_size // 4)
        self.batch_norm = batch_norm
    
    def forward(self,embedding):
        if self.batch_norm:
            f1 = self.dropout(self.relu(self.bn(self.f1(embedding))))
        else :
            f1 = self.dropout(self.relu(self.f1(embedding)))
        f2 = self.sigmoid(self.f2(f1))
        return f2

class FC_NN_large(nn.Module):
    def __init__(self,embedding_size,dropout,last_layer=True,device='cuda:0',batch_norm=False):
        #the model should be a vae_nlp model
        super(FC_NN_large,self).__init__()
        #self.vae_model= model
        #self.vae_model.model.eval()
        self.embeeding_size=  embedding_size
        self.dropout = nn.Dropout(p=dropout)
        self.device= device
        self.last_layer = last_layer #whether to use the last layer of GRU hidden units
        self.f1 = nn.Linear(embedding_size,embedding_size // 4)
        self.f2 = nn.Linear(embedding_size // 4 , embedding_size // 16)
        self.f3 = nn.Linear(embedding_size // 16, 1)
        self.embedding_size = embedding_size
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.batch_norm = batch_norm
        self.bn1 = nn.BatchNorm1d(embedding_size//4)
        self.bn2 = nn.BatchNorm1d(embedding_size//16)
        
    
    def forward(self,embedding):
        # embedding = self.dropout(embedding)
        if self.batch_norm:
            # s1 = self.dropout(self.relu(self.bn1(self.f1(embedding))))
            s1 = self.dropout(self.bn1(self.relu(self.f1(embedding))))
            # s2 = self.dropout(self.relu(self.bn2(self.f2(s1))))
            s2 = self.dropout(self.bn2(self.relu(self.f2(s1))))
        else :
            s1 = self.dropout(self.relu(self.f1(embedding)))
            s2 = self.dropout(self.relu(self.f2(s1)))
        return self.sigmoid(self.f3(s2))

class FC_NN_huge(nn.Module):
    def __init__(self,embedding_size,dropout,last_layer=True,device='cuda:0',batch_norm=False):
        #the model should be a vae_nlp model
        super(FC_NN_huge,self).__init__()
        #self.vae_model= model
        #self.vae_model.model.eval()
        self.embeeding_size=  embedding_size
        self.dropout = nn.Dropout(p=dropout)
        self.device= device
        self.last_layer = last_layer #whether to use the last layer of GRU hidden units
        self.f1 = nn.Linear(embedding_size,embedding_size // 2)
        self.f2 = nn.Linear(embedding_size//2,embedding_size //8)
        self.f3 = nn.Linear(embedding_size // 8 , embedding_size // 25)
        self.f4 = nn.Linear(embedding_size // 25, 1)
        self.embedding_size = embedding_size
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.batch_norm = batch_norm
        self.bn1 = nn.BatchNorm1d(embedding_size//2)
        self.bn2 = nn.BatchNorm1d(embedding_size//8)
        self.bn3 = nn.BatchNorm1d(embedding_size//25)
        
    
    def forward(self,embedding):
        # embedding = self.dropout(embedding)
        if self.batch_norm:
            # s1 = self.dropout(self.relu(self.bn1(self.f1(embedding))))
            s1 = self.dropout(self.bn1(self.relu(self.f1(embedding))))
            # s2 = self.dropout(self.relu(self.bn2(self.f2(s1))))
            s2 = self.dropout(self.bn2(self.relu(self.f2(s1))))
            s3 = self.dropout(self.bn3(self.relu(self.f3(s2))))
        else :
            s1 = self.dropout(self.relu(self.f1(embedding)))
            s2 = self.dropout(self.relu(self.f2(s1)))
            s3 = self.dropout(self.relu(self.f3(s2)))
        return self.sigmoid(self.f4(s3))
