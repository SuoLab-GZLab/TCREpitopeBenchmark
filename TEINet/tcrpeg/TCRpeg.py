import torch
import numpy as np
import pandas as pd
import torch.nn as nn
#from tcrpeg.model import TCRpeg_model, TCRpeg_vj_model
# from model import TCRpeg_model,TCRpeg_vj_model
from tqdm import tqdm
import time
from datetime import datetime
import argparse
import os
from math import ceil

class TCRpeg:
    def __init__(
        self,
        max_length=30,
        embedding_size=32,
        hidden_size=64,
        dropout=0,
        num_layers=1,
        device="cuda:0",
        load_data=False,
        path_train=None,
        evaluate=0,
        path_test=None,
        embedding_path="data/embedding_32.txt",
        vs_list=None,
        js_list=None,
        vj=False
    ):
        '''
        @max_length: the maximum length of CDR3 seqs you want to deal with
        @embedding_size: length of the amino acid embedding vectors
        @hidden_size: The number of features in the hidden state of GRU layers
        @dropout: dropout rate
        @device: gpu or cpu you want the software run on; GPU: cuda:0' ('cuda:x') / CPU: 'cpu'
        @load_data & path_train: If you want to train tcrpeg, you should set load_data=True,path_train='path_to_training_file'
                                Also, it accepts list: [AAs1, AAs2,.....]
                                if using vj model, the format should be [[AAs seqs],[v genes],[j genes]]

        @evaluate & path_test : If you want to evaluate on test set, you should set evaluate=1, path_test='path_to_test_set'
                                Also accepts list as input. Same with 'path_train'
        @embedding_path: Recorded trained embeddings for amino acids (22 x embedding_size)
        @vs_list & js_list: The list of v and j genes. Only used when you're using the tcrpeg_vj model; 
                           The default v and j gene lists contain 43 and 12 genes respectively. If your TCR dataset
                           contains v,or j genes that are not included the default vs_list/js_list, yould should provide it.
                           type(vs_list) should be a list
        '''

        self.max_length = max_length
        self.embedding_size = embedding_size
        self.device = device
        self.dropout = dropout

        emb = np.array(
            pd.read_csv(embedding_path, sep=",", names=list(range(embedding_size + 1)))
        )
        embedding = np.zeros((23, embedding_size))  # 23 x letent_dim
        embedding[1:, :] = emb[:, 1:].astype(float)  # 22 x latent_dim
        self.embedding_layer = nn.Embedding.from_pretrained(
            torch.tensor(embedding), padding_idx=0,freeze=False
        ).to(device) #the embedding layer for AAs
        self.num_layers = num_layers

        self.aas = "sACDEFGHIKLMNPQRSTVWYe"  # 0 index is the padding,1 is START, 22 is END
        self.aa2idx = {self.aas[i]: i for i in range(len(self.aas))}
        self.idx2aa = {v: k for k, v in self.aa2idx.items()}
        
        vs_default = ['TRBV10-1','TRBV10-2','TRBV10-3','TRBV11-1','TRBV11-2','TRBV11-3','TRBV12-5', 'TRBV13', 'TRBV14', 'TRBV15', 
        'TRBV16', 'TRBV18','TRBV19','TRBV2', 'TRBV20-1', 'TRBV25-1', 'TRBV27', 'TRBV28', 'TRBV29-1', 'TRBV3-1', 'TRBV30',
'TRBV4-1', 'TRBV4-2','TRBV4-3',
 'TRBV5-1', 'TRBV5-4', 'TRBV5-5', 'TRBV5-6', 'TRBV5-8','TRBV6-1', 'TRBV6-4', 'TRBV6-5', 'TRBV6-6','TRBV6-8', 'TRBV6-9', 'TRBV7-2', 'TRBV7-3',
  'TRBV7-4', 'TRBV7-6', 'TRBV7-7', 'TRBV7-8', 'TRBV7-9', 'TRBV9']
        js_default = ['TRBJ1-1', 'TRBJ1-2', 'TRBJ1-3','TRBJ1-4', 'TRBJ1-5', 'TRBJ1-6','TRBJ2-1', 'TRBJ2-2', 'TRBJ2-3', 'TRBJ2-4', 'TRBJ2-6', 'TRBJ2-7']
        vs_default.sort() #to keep the order
        js_default.sort()

        if vs_list is not None:
            #here, to load the v and j genes you specify
            #the inputed vs and js needed to be sorted 
            vs_list.sort()
            js_list.sort()
            self.vs_list = vs_list
            self.js_list = js_list 
        else :
            self.vs_list = vs_default
            self.js_list = js_default
        self.v2idx = {self.vs_list[i]: i for i in range(len(self.vs_list))}
        self.idx2v = {v: k for k, v in self.v2idx.items()}
        self.j2idx = {self.js_list[i]: i for i in range(len(self.js_list))}
        self.idx2j = {v: k for k, v in self.j2idx.items()}
        self.loss_v,self.loss_j = nn.CrossEntropyLoss(reduction='sum'),nn.CrossEntropyLoss(reduction='sum')

        self.NLL = torch.nn.NLLLoss(
            ignore_index=-1, reduction="sum"
        ) #-1 index is the padding index in the training process, so need to ignore it  

        self.hidden_size = hidden_size

        self.evaluate = True if evaluate != 0 else False
        if load_data:
            if vj:
                if type(path_train) is not str:
                    self.aas_seqs_train = np.array([x[0] for x in path_train])
                    self.vs_train,self.js_train=[x[1] for x in path_train],[x[2] for x in path_train]
                else :
                    self.aas_seqs_train,self.vs_train,self.js_train = self.load_data(path_train,vj=True)
                self.vs_train,self.js_train = self.gene2embs(self.vs_train,'v'),self.gene2embs(self.js_train,'j')
            else :
                if type(path_train) is not str:
                    self.aas_seqs_train = path_train
                else :
                    self.aas_seqs_train = self.load_data(path_train)
            if self.evaluate:
                #remember to add!!
                if vj is not None:
                    if type(path_train) is not str:
                        self.aas_seqs_test = np.array([x[0] for x in path_test])
                        self.vs_test,self.js_test=[x[1] for x in path_test],[x[2] for x in path_test]
                    else :
                        self.aas_seqs_test,self.vs_test,self.js_test = self.load_data(path_test)
                    self.vs_test,self.js_test = self.gene2embs(self.vs_test,'v'),self.gene2embs(self.js_test,'j')
                else :
                    if type(path_test) is not str:
                        self.aas_seqs_test = path_test
                    else :
                        self.aas_seqs_test = self.load_data(path_test)
            print("Have loaded the data, total training seqs :", len(self.aas_seqs_train))

    def aas2embs(self, seqs):
        """
        @seqs: list of AAs

        #return [[1,2,3,0,0],[2,3,4,0].....], the padded 
        """
        seqs = [[self.aa2idx[k] + 1 for k in seq] for seq in seqs]
        # lengths = [len(seq)+2 for seq in seqs] #contains start and stop
        lengths = [len(seq) + 1 for seq in seqs]  # contains the start token
        inputs = [[1] + seq for seq in seqs]
        # inputs = [[1] + seq + [22] for seq in seqs]
        targets = [seq + [22] for seq in seqs]
        inputs = [seq + [0] * (self.max_length - len(seq)) for seq in inputs] #pad
        targets = [seq + [0] * (self.max_length - len(seq)) for seq in targets]#pad

        return inputs, targets, lengths  # targets are N x max_length

    def gene2embs(self, genes, gene_type='v'):
        '''
        @genes: a list containing genes
        '''
        if gene_type == 'v':
            return [self.v2idx[g] for g in genes]
        else :
            assert gene_type == 'j', 'the gene type should only be v or j'
            return [self.j2idx[g] for g in genes]

    def load_data(self, path,vj=False,sep='\t'):
        if vj:
            if type(path) == list:
                return path[0],path[1],path[2]
            
            data = pd.read_csv(path,sep)
            if data.columns[0] != 'seq':
                return data['amino_acid'].values,data['v_gene'].values,data['j_gene'].values    
            else :
                return data['seq'].values,data['v'].values,data['j'].values
        else :
            if type(path) == list:
                return path

            if not path.endswith("txt"):
                if 'vdj' in path: #this is used for loading epitope-tcrs
                    data = pd.read_csv(path)['cdr3b'].values
                    return data
                data = pd.read_csv(path, sep="\t")
            else:
                data = pd.read_csv(path, names=["seq"])
        return data["seq"].values

    def loss_fn(self, logp, target, lengths):
        '''
        Loss function for training a tcrpeg model
        '''
        logp = logp.view(-1, 22)
        target = target - 1
        # target is b x
        # lengths = lengths - 1 #remember to change this!
        target = target[:, : torch.max(lengths).item()].contiguous()

        target = target.view(-1)
        nll_loss = self.NLL(logp, target)

        return nll_loss

    def create_model(self, load=False, path=None,vj=False, bidirectional=False):
        '''
        Create the TCRpeg (TCRpeg_vj) model 
        @load: whether to load from pre-trained model
        @path: the path to pre-trained model, only need to specify when @load=True
        @vj: if set to True, will create the TCRpeg_vj model
        '''

        if vj:
            model = TCRpeg_vj_model(
                self.embedding_layer,
                self.embedding_size,
                self.hidden_size,
                self.dropout,
                self.max_length,
                num_layers=self.num_layers  ,
                num_v = len(self.vs_list),
                num_j = len(self.js_list)
            )
            self.vj=True
        else :
            model = TCRpeg_model(
                self.embedding_layer,
                self.embedding_size,
                self.hidden_size,
                dropout=self.dropout,
                num_layers=self.num_layers,
                bidirectional=bidirectional
                )
            self.vj=False
        model.train()
        if load:
            model.load_state_dict(torch.load(path))
            model.eval()
        self.model = model.to(self.device)
        # return model    

    def save(self, path):
        '''
        Save the model to the @path you specify
        '''
        torch.save(self.model.state_dict(), path)

    def sampling_tcrpeg(self, seqs):
        '''        
        @seqs: list containing CDR3 sequences
        #return: the log_prob of the input sequences
        '''
        with torch.no_grad():
            batch_size = len(seqs)

            inputs, targets, lengths = self.aas2embs(seqs)
            inputs, targets, lengths = (
                torch.LongTensor(inputs).to(self.device),
                torch.LongTensor(targets).to(self.device),
                torch.LongTensor(lengths).to(self.device),
            )
            logp = self.model(inputs, lengths)  
            
            logp = logp.detach().cpu()
            targets = targets.detach().cpu()

            targets = targets - 1

            target = targets[:, : torch.max(lengths).item()].contiguous()  # 0 is pad
            # logpx_z = torch.zeros(batch_size,torch.max(lengths).items())
            target = torch.where(target >= 0, target, 0)  # B x max_length ,0-21
            target_onehot = torch.FloatTensor(batch_size, torch.max(lengths).item(), 22)

            target_onehot.zero_()
            target_onehot.scatter_(
                2, target.view(batch_size, torch.max(lengths).item(), 1), 1
            )  # B x l x 22\

            probs = target_onehot * logp

            probs = probs.sum(dim=-1)  # B x L

            probs = torch.where(target > 0, probs, torch.FloatTensor([0.0])) #mask out the probs of padding positions
            logpx_given_z = probs.sum(dim=-1).numpy()
            return logpx_given_z  # B, log probability

    def sampling_tcrpeg_batch(self, seqs,batch_size=10000):
        '''
        @seqs: list containing CDR3 sequences
        #return: the log_prob of the input sequences
        Inferring in batch
        '''
        logpx_given_z = np.zeros(len(seqs))
        with torch.no_grad():
            #batch_size = len(seqs)
            for i in tqdm(range(int(len(seqs)/batch_size)+1)):
                end = len(seqs) if (i+1) * batch_size > len(seqs) else (i+1) * batch_size
                seq_batch = seqs[i * batch_size : end]
                if len(seq_batch) == 0:
                    continue
                log_probs = self.sampling_tcrpeg(seq_batch) #change here
                logpx_given_z[i*batch_size : end] = log_probs
            return logpx_given_z  # B, log probability



    def train_tcrpeg(self, epochs, batch_size, lr, info=None, model_name=None,record_dir=None):
        '''
        @epochs: epochs
        @batch_size: batch_size
        @lr: initial learning rate; The learning rate will reduced by lr=lr*0.2 at the middle of training
        @info: the information you want to record at the top of the log file (only activated when you specify the record dir)
        @model_name: the models will be saved as model_name.pth (only activated when you specify the record_dir)
        @record_dir: the directory you want to record your models; if not provided, the trained models will not be saved
        '''
        print("begin the training process")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        batch_size = batch_size
        epochs = epochs
        record = True if record_dir is not None else False
        if info is None:
            info = 'Not provided'
        if model_name is None:
            model_name = 'Not provided'
        

        # start_time = datetime.now().strftime("%m_%d_%Y-%H:%M:%S")
        if record:
            # dir = record_dir + "/" + start_time.split("-")[1]
            dir = record_dir 
            # if not os.path.exists(dir):
            #     os.mkdir(dir)
            with open(dir + "/logs.txt", "a") as f:
                f.write(info + "\n")
        aas_seqs = self.aas_seqs_train

        
        for epoch in range(epochs):
            print("begin epoch :", epoch + 1)
            self.model.train()
            aas_seqs = np.random.permutation(aas_seqs) #shuffle at the beginning of each epoch
            nll_loss_ = []
            num_batch = len(aas_seqs) // batch_size
            for iter in tqdm(range(len(aas_seqs) // batch_size)):
                # seqs = aas_seqs[infer_arr[iter * batch_size : (iter+1) * batch_size]]
                seqs = aas_seqs[iter * batch_size : (iter + 1) * batch_size]
                inputs, targets, lengths = self.aas2embs(seqs) #embeds the sequences
                inputs, targets, lengths = (
                    torch.LongTensor(inputs).to(self.device),
                    torch.LongTensor(targets).to(self.device),
                    torch.LongTensor(lengths).to(self.device),
                )
                logp = self.model(inputs, lengths)

                nllloss = self.loss_fn(logp, targets, lengths)
                nll_loss_.append(nllloss.item() / batch_size)
                loss = nllloss / batch_size  # remember to multiply the beta!
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("nll_loss: {0}".format(sum(nll_loss_) / num_batch))
            if record:
                with open(dir + "/logs.txt", "a") as f:
                    f.write(
                        "For {} model, trained at {}_th epoch with nll (train) are {}\n".format(
                            model_name,
                            epoch + 1,
                            sum(nll_loss_) / num_batch,
                        )
                    )
        
                if epoch % 3 == 0:
                    # early_stopping.save_checkpoint(sum(elbo_test)/num_batch,model)
                    print('wrong enter')
                    self.save(
                        dir
                        + "/"
                        + model_name
                        + "_{}.pth".format(str(epoch + 1))
                    )
    
            # if epoch != 0 and epoch % ((epochs) // 2) == 0 :
            if epoch == 10:
                print('The learning rate has beed reduced')
                optimizer.param_groups[0]["lr"] = lr * 0.2

        print("Done training")
        if record:
            self.save(
                dir + "/" + model_name + "_{}.pth".format(str(epoch + 1))
            )
        return sum(nll_loss_) / num_batch
        
    
    def evaluate(self,batch_size,vj=False):
        '''
        Evaluation on test set for tcrpeg model
        @batch_size: batch_size for loading data
        @vj: whether the model is TCRpeg or TCRpeg_vj

        @return: (nll_loss,v_loss,j_loss) if vj is True else return the nll_loss
        '''
        aas_val = self.aas_seqs_test
        with torch.no_grad():
            nll_test = []
            v_test_,j_test_ = [],[]
            num_batch = len(aas_val) // batch_size
            for iter in tqdm(range(num_batch)):
                seqs = aas_val[iter * batch_size : (iter + 1) * batch_size]
                if vj:
                    vs,js = self.vs_test[iter * batch_size : (iter + 1) * batch_size],self.js_test[iter * batch_size : (iter + 1) * batch_size]
                    vs,js = torch.LongTensor(vs).to(self.device),torch.LongTensor(js).to(self.device)
                inputs, targets, lengths = self.aas2embs(seqs)
                inputs, targets, lengths = (
                    torch.LongTensor(inputs).to(self.device),
                    torch.LongTensor(targets).to(self.device),
                    torch.LongTensor(lengths).to(self.device),
                )
                if vj:
                    logp,v_pre,j_pre = self.model(inputs,lengths)
                else :
                    logp = self.model(inputs, lengths)

                nllloss = self.loss_fn(
                    logp, targets, lengths, decoder_only=True
                )
                if vj:
                    v_loss,j_loss = self.loss_v(v_pre,vs),self.loss_j(j_pre,js)
                    v_test_.append(v_loss.item()/batch_size)
                    j_test_.append(j_loss.item()/batch_size)
                nll_test.append(nllloss.item() / batch_size)
        if vj:
            print("nll_loss,v_loss,j_loss for test set: {0}, {1}, {2}".format(sum(nll_test) / num_batch,sum(v_test_) / num_batch,sum(j_test_) / num_batch))
            return sum(nll_test) / num_batch,sum(v_test_) / num_batch,sum(j_test_) / num_batch

        else :
            print("nll_loss for val: {0}".format(sum(nll_test) / num_batch))
            return sum(nll_test) / num_batch
        

    def train_tcrpeg_vj(self, epochs, batch_size, lr, info=None, model_name=None,record_dir=None):
        '''
        Train a TCRpeg with v j usage
        Same parameter setting as @train_tcrpeg
        '''
        print("begin the training process")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        batch_size = batch_size
        epochs = epochs
        aas_seqs = self.aas_seqs_train
        vs_,js_ = np.array(self.vs_train),np.array(self.js_train)
        record = True if record_dir is not None else False
        #start_time = datetime.now().strftime("%m_%d_%Y-%H:%M:%S")
        if info is None:
            info = 'Not provided'
        if model_name is None:
            model_name = 'Not provided'

        if record:
            #dir = record_dir + "/" + start_time.split("-")[1]
            dir = record_dir
            # if not os.path.exists(dir):
            #     os.mkdir(dir)
            with open(dir + "/logs.txt", "a") as f:
                f.write(info + "\n")
        infer = np.arange(len(self.aas_seqs_train)) #used to shuffle the training data

        
    
        for epoch in range(epochs):
            np.random.shuffle(infer)
            print("begin epoch :", epoch + 1)
            self.model.train()
            
            aas_seqs = aas_seqs[infer]
            vs_,js_ = vs_[infer],js_[infer] #shuffle
            nll_loss_ = []
            j_loss_ = []
            v_loss_ = []
            num_batch = len(aas_seqs) // batch_size
            for iter in tqdm(range(len(aas_seqs) // batch_size)):
                # seqs = aas_seqs[infer_arr[iter * batch_size : (iter+1) * batch_size]]
                seqs = aas_seqs[iter * batch_size : (iter + 1) * batch_size]
                vs = vs_[iter * batch_size : (iter + 1) * batch_size]
                js = js_[iter * batch_size : (iter + 1) * batch_size]
                vs ,js = torch.LongTensor(vs).to(self.device),torch.LongTensor(js).to(self.device)

                inputs, targets, lengths = self.aas2embs(seqs)
                inputs, targets, lengths = (
                    torch.LongTensor(inputs).to(self.device),
                    torch.LongTensor(targets).to(self.device),
                    torch.LongTensor(lengths).to(self.device),
                )
                logp,v_pre,j_pre = self.model(inputs, lengths)

                nllloss = self.loss_fn(logp, targets, lengths)
                v_loss,j_loss = self.loss_v(v_pre,vs),self.loss_j(j_pre,js)
                nll_loss_.append(nllloss.item() / batch_size)
                v_loss_.append(v_loss.item()/batch_size)
                j_loss_.append(j_loss.item()/batch_size)
                loss = nllloss / batch_size  + 1.5 * v_loss/batch_size + 1.5 * j_loss/batch_size # remember to multiply the beta!
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("nll_loss: {0}".format(sum(nll_loss_) / num_batch))
            print("v_loss: {0}".format(sum(v_loss_) / num_batch))
            print("j_loss: {0}".format(sum(j_loss_) / num_batch))
            if record:
                with open(dir + "/logs.txt", "a") as f:
                    f.write(
                        "For {} model, trained at {}_th epoch with nll (train),v,j are {},{},{}\n".format(
                            model_name,
                            epoch + 1,
                            sum(nll_loss_) / num_batch,
                            sum(v_loss_) / num_batch,
                            sum(j_loss_) / num_batch
                        )
                    )
                if epoch % 3 == 0:
                    # early_stopping.save_checkpoint(sum(elbo_test)/num_batch,model)
                    self.save(
                        dir
                        + "/"
                        + model_name
                        + "_{}.pth".format(str(epoch + 1))
                    )
            if epoch != 0 and epoch % ((epochs) // 2) == 0:
                print('The learning rate has beed reduced')
                optimizer.param_groups[0]["lr"] = lr * 0.2

        print("Done training")
        if record:
            self.save(
                dir + "/" + model_name + "_{}.pth".format( str(epoch + 1))
            )

    def generate_decoder(self, num_to_gen, record_path=None):
        # need to change this to batch!!!  20k seqs need ~2 mins
        SOS_token = 1
        EOS_token = 22
        self_ = self
        self = self.model
        refer = list(range(22))  # 0~21
        # self = self.model
        seqs = []
        batch_size = 1
        with torch.no_grad():
            for _ in tqdm(range(num_to_gen)):
                decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)]).to("cuda:0")
                seq = []
                for t in range(30):
                    decoder_input = (
                        self.embedding(decoder_input).to(torch.float32).permute([1, 0, 2]))
                    if t != 0:
                        decoder_output, decoder_hidden = self.decoder_rnn(decoder_input, decoder_hidden)
                    else:
                        decoder_output, decoder_hidden = self.decoder_rnn(decoder_input)
                    decoder_output = self.out_layer(decoder_output)
                    decoder_output = nn.functional.softmax(decoder_output, dim=-1).view(
                        -1
                    )  # 22,
                    next_idx = np.random.choice(refer, p=decoder_output.detach().cpu().numpy())
                    if next_idx == EOS_token - 1:  ##
                        break
                    seq.append(self_.idx2aa[next_idx])
                    decoder_input = torch.LongTensor([[next_idx + 1] for _ in range(batch_size)]).to("cuda:0")
                seqs.append("".join(seq))

        if record_path is not None:
            with open(record_path, "w") as f:
                for s in seqs:
                    f.write(s + "\n")

    def generate_tcrpeg(self, num_to_gen, batch_size,record_path=None):
        '''
        Generating new sequences by the trained model; Note that (num_to_gen % batch_size) should be 0
        @num_to_gen: number of sequences need to generate
        @batch_size: batch_size
        @record_path: if specified, will record the generated seqs 
        #return: a list containing the generated seqs
        '''
        # need to change this to batch!!!  20k seqs need ~2 mins
        SOS_token = 1
        EOS_token = 22
        self_ = self
        self = self.model
        
        # self = self.model
        seqs = []
        assert num_to_gen % batch_size == 0, 'The num_to_gen have to be a multiple of batch_size'
        steps = num_to_gen // batch_size
        with torch.no_grad():
            for _ in tqdm(range(steps)):
                decoded_batch = torch.zeros((batch_size, self.max_length))
                decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)]).to(self.device)
                EOS_position = np.ones(batch_size,dtype=int) * -1
                
                for t in range(self.max_length):
                    decoder_input = self.embedding(decoder_input).to(torch.float32).permute([1,0,2])
                    if t != 0:
                        decoder_output, decoder_hidden = self.decoder_rnn(decoder_input, decoder_hidden)
                    else:
                        decoder_output, decoder_hidden = self.decoder_rnn(decoder_input)
                    decoder_output = self.out_layer(decoder_output)
                    decoder_output = nn.functional.softmax(decoder_output, dim=-1).view(batch_size,22)  # 1 xB x 22,
                    #print(decoder_output.size())
                    next_idx = torch.multinomial(decoder_output,1,True).view(-1) #B,
                    for k in range(batch_size):
                        if next_idx[k] == EOS_token - 1 and EOS_position[k] == -1:
                            EOS_position[k] = t
                    # if next_idx == EOS_token - 1:  ##
                    #     break
                    decoded_batch[:,t] = next_idx
                    decoder_input = (next_idx + 1).view(-1,1)
                decoded_batch = decoded_batch.cpu().numpy().astype(int)
                decoded = []
                for k in range(batch_size):
                #print(EOS_position[k])
                    decoded.append(list(decoded_batch[k,:EOS_position[k]]))
                decoded = [[self_.idx2aa[a] for a in seq] for seq in decoded]
                decoded = [''.join(seq) for seq in decoded]
                seqs = seqs + decoded
            
        if record_path is not None:
            with open(record_path, "w") as f:
                for s in seqs:
                    f.write(s + "\n")
        return seqs

    def generate_tcrpeg_vj(self, num_to_gen, batch_size,record_path=None):
        '''
        #return: a list containing the generated seqs,vs,js in the format of [[seqs],[vs],[js]]
        '''
        # need to change this to batch!!!  20k seqs need ~2 mins
        SOS_token = 1
        EOS_token = 22
        self_ = self
        self = self.model
        
        # self = self.model
        seqs = []
        assert num_to_gen % batch_size == 0, 'The num_to_gen have to be a multiple of batch_size'
        steps = num_to_gen // batch_size
        with torch.no_grad():
            for _ in tqdm(range(steps)):
                decoded_batch = torch.zeros((batch_size, self.max_length))
                decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)]).to(self.device)
                EOS_position = np.ones(batch_size,dtype=int) * -1
                
                for t in range(self.max_length):
                    decoder_input = self.embedding(decoder_input).to(torch.float32).permute([1,0,2])
                    if t != 0:
                        decoder_output, decoder_hidden = self.decoder_rnn(decoder_input, decoder_hidden)
                    else:
                        decoder_output, decoder_hidden = self.decoder_rnn(decoder_input)
                    decoder_output = self.out_layer(decoder_output)
                    decoder_output = nn.functional.softmax(decoder_output, dim=-1).view(batch_size,22)  # 1 xB x 22,
                    #print(decoder_output.size())
                    next_idx = torch.multinomial(decoder_output,1,True).view(-1) #B,
                    for k in range(batch_size):
                        if next_idx[k] == EOS_token - 1 and EOS_position[k] == -1:
                            EOS_position[k] = t
                    # if next_idx == EOS_token - 1:  ##
                    #     break
                    decoded_batch[:,t] = next_idx
                    decoder_input = (next_idx + 1).view(-1,1)
                decoded_batch = decoded_batch.cpu().numpy().astype(int)
                decoded = []
                for k in range(batch_size):
                #print(EOS_position[k])
                    decoded.append(list(decoded_batch[k,:EOS_position[k]]))
                decoded = [[self_.idx2aa[a] for a in seq] for seq in decoded]
                decoded = [''.join(seq) for seq in decoded]
                seqs = seqs + decoded
            
            #used the generated seqs to generate the v and j gene:
            vs_whole,js_whole = [],[]
            for iter in tqdm(range(steps)):
                seqs_batch = seqs[iter * batch_size : (iter + 1) * batch_size]
                inputs, targets, lengths = self_.aas2embs(seqs_batch)
                inputs, targets, lengths = (
                    torch.LongTensor(inputs).to(self_.device),
                    torch.LongTensor(targets).to(self_.device),
                    torch.LongTensor(lengths).to(self_.device),
                )
                _,v_pre,j_pre = self_.model(inputs, lengths)
                v_pre,j_pre = nn.functional.softmax(v_pre,-1),nn.functional.softmax(j_pre,-1)
                v_pre_id = torch.multinomial(v_pre,1,True).view(-1).detach().cpu().numpy() #B,
                j_pre_id = torch.multinomial(j_pre,1,True).view(-1).detach().cpu().numpy()

                v_pres = [self_.idx2v[i] for i in v_pre_id]
                j_pres = [self_.idx2j[i] for i in j_pre_id]
                vs_whole = vs_whole + list(v_pres)
                js_whole = js_whole + list(j_pres)
            
        if record_path is not None:
            with open(record_path, "w") as f:
                for i,s in enumerate(seqs):
                    f.write(s + ',' + vs_whole[i] + ',' + js_whole[i] + "\n")
        return [seqs,vs_whole,js_whole]

    def get_embedding(self,seqs,last_layer=False):
        '''
        Get the embedding of CDR3 sequences
        @seqs: a list containing the CDR3 sequences
        @batch_size: batch_size
        @last_layer: if set to True, will return the hidden features of the last GRU layer. Otherwise, hidden features of all layers are used

        #return: embedding of CDR3 sequences. The shape would be (B,num_layers*hidden_size) if last_layer=False, (B,hidden_size) if last_layer=True
        '''
        nums = len(seqs)
        self.model.eval()
        with torch.no_grad():
            inputs,targets,lengths = self.aas2embs(seqs)
            inputs,targets,lengths = torch.LongTensor(inputs).to(self.device),torch.LongTensor(targets).to(self.device),torch.LongTensor(lengths).to(self.device)
            if self.vj:                
                _,_,_,embedding = self.model(inputs,lengths,True)
            else :
                _,embedding= self.model(inputs,lengths,True)
            if last_layer:
                embedding = embedding[:,-1,:] #B x 64
            else :
                embedding = embedding.view(nums,-1)
            #embedding = embedding.view(batch_size,-1)
        return embedding.detach().cpu().numpy()

import torch
import torch.nn as nn 
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from tqdm import tqdm
import numpy as np

#note that, the padding_idx is 0, the idx for START token is 1
#the idx for AAs start from 2 and the idx for END stoken is 22


class TCRpeg_model(nn.Module):
    def __init__(self,embedding_layer,embedding_size,hidden_size,dropout,max_length=30,sos_idx=1,pad_idx=0,eos_idx=22,device='cuda:0',num_layers=1,bidirectional=False):
        super(TCRpeg_model,self).__init__()
        self.max_length = max_length
        self.sos_idx=sos_idx
        self.eos_idx=  eos_idx
        self.pad_idx = pad_idx
        # self.embedding_dropout= embedding_dropout
        self.embedding = embedding_layer
        self.embedding_size=  embedding_size
        self.hidden_size=hidden_size
        
        self.decoder_rnn = nn.GRU(embedding_size,hidden_size,num_layers,dropout=dropout,bidirectional=bidirectional)
        self.num_layers = num_layers
        self.D = 2 if bidirectional else 1
        self.out_layer=  nn.Linear(hidden_size * self.D, 22)
        self.device = device 
        self.bidirectional = bidirectional       
    
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

