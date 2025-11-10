import torch
import math
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")

def set_modelseed(modelseed=23):
    random.seed(modelseed)
    np.random.seed(modelseed)
    torch.manual_seed(modelseed)
    torch.cuda.manual_seed(modelseed)
    torch.cuda.manual_seed_all(modelseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True #Set for reproducibility

'''
Default model parameters for TEPCAM are listed below:
    d_model = 32
    n_heads = 6
'''

max_len = 35 # max_len_tcr:20 + max_len_pep:15

#embedding_matrix = BLOSUM_embedding("None")

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len, dropout_rate=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class Embeddings(nn.Module):
    #Vanilla embedding + positional embedding :)
    def __init__(self, vocab_size, hidden_size, max_len):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = PositionalEncoding(hidden_size, max_len, dropout_rate=0.1)
    
    def forward(self, input_ids):
        words_embeddings = self.word_embeddings(input_ids)    
        embeddings = self.position_embeddings(words_embeddings.transpose(0,1)).transpose(0,1)
        return embeddings

 
def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(21):PAD token for TEPdata
    pad_attn_mask = seq_k.data.eq(21).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, len_q, len_k] 
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) #[batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9) #Automatically brocast to n_head on dimension 1
        attn = nn.Softmax(dim=-1)(scores) #calculate total attntion value of a single position
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model_1, d_model_2, n_heads,d_model):
        super(MultiHeadAttention, self).__init__()
        self.d_model_1 = d_model_1
        self.d_model_2 = d_model_2
        self.d_model = d_model
        self.n_heads = n_heads
        self.W_Q_dense = nn.Linear(self.d_model_1, self.d_model * self.n_heads, bias=False) 
        self.W_K_dense = nn.Linear(self.d_model_2, self.d_model * self.n_heads, bias=False)
        self.W_V_dense = nn.Linear(self.d_model_2, self.d_model * self.n_heads, bias=False)
        
        self.scale_product = ScaledDotProductAttention(self.d_model)
        self.out_dense = nn.Linear(self.n_heads * self.d_model, self.d_model, bias=False)  # self.n_heads * self.d_dim = const
        self.LN = nn.LayerNorm(self.d_model)
        
    def forward(self, Q, K, V, attn_mask):
        Q_residual, batch_size = Q, Q.size(0)
        # (B, S[seqlen], D[d_model]) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q_dense(Q).view(batch_size, -1, self.n_heads, self.d_model).transpose(1,2)
        k_s = self.W_K_dense(K).view(batch_size, -1, self.n_heads, self.d_model).transpose(1,2)
        self.v_s = self.W_V_dense(V).view(batch_size, -1, self.n_heads, self.d_model).transpose(1,2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_q, seq_k]
        context, attn = self.scale_product(q_s, k_s, self.v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_model)
        context = self.out_dense(context)
        output = context + Q_residual
        output = self.LN(output)
        return output, attn

class CrossAttention(nn.Module):
    def __init__(self, feature_size,n_heads,d_model):
        super(CrossAttention,self).__init__()
        self.feature_size = feature_size
        self.n_heads = n_heads
        self.cross_attention = MultiHeadAttention(feature_size,feature_size,self.n_heads,d_model) #two inputs must keep same dimension 
    def forward(self, input_1, input_2, attn_mask1, attn_mask2):
        cr_1_q, cr_1_k, cr_1_v = input_1, input_2, input_2 
        cr_2_q, cr_2_k, cr_2_v = input_2, input_1, input_1 
        cr_out_1,attn1 = self.cross_attention(cr_1_q, cr_1_k, cr_1_v, attn_mask=attn_mask1) #tcr:q,pep:k
        cr_out_2,attn2 = self.cross_attention(cr_2_q, cr_2_k, cr_2_v, attn_mask=attn_mask2) #pep:q,tcr:k
        return cr_out_1, cr_out_2, attn1, attn2


class TEPDA(nn.Module):
    def __init__(self, d_model, batch_size, modelseed, n_heads):
        super(TEPDA, self).__init__()
        set_modelseed(modelseed)
        self.d_model = d_model
        self.batch_size = batch_size
        self.tcr_embedding = Embeddings(vocab_size=26, hidden_size=self.d_model, max_len=20)
        self.pep_embedding = Embeddings(vocab_size=26, hidden_size=self.d_model, max_len=15)
        self.n_heads = n_heads
        self.SALayer = MultiHeadAttention(self.d_model, self.d_model, self.n_heads // 2, self.d_model)
        self.CALayer = CrossAttention(self.d_model, self.n_heads, self.d_model)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.GELU(),
            nn.Conv2d(in_channels=4, out_channels=d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.GELU(),
            nn.Conv2d(d_model // 2, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )
        self.FFN = nn.Sequential(
            nn.Linear(2240, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Linear(16, 2),
        )

    def forward(self, tcr, peptide):
        pep_embed = self.pep_embedding(peptide)
        tcr_embed = self.tcr_embedding(tcr)
        tcr_attn_mask = get_attn_pad_mask(tcr, tcr)
        pep_attn_mask = get_attn_pad_mask(peptide, peptide)
        tcr2pep_mask = get_attn_pad_mask(tcr, peptide)
        pep2tcr_mask = get_attn_pad_mask(peptide, tcr)

        # Self-attention layers
        tcr_sa, attn_tcr = self.SALayer(tcr_embed, tcr_embed, tcr_embed, tcr_attn_mask)
        pep_sa, attn_pep = self.SALayer(pep_embed, pep_embed, pep_embed, pep_attn_mask)

        # Cross-attention layers
        CA1, CA2, attn_tcr_ca, attn_pep_ca = self.CALayer(tcr_sa, pep_sa, tcr2pep_mask, pep2tcr_mask)
        CA1_conv = self.conv_layer(CA1.unsqueeze(1))
        CA1_conv = torch.mean(CA1_conv, dim=1, keepdim=False)
        CA2_conv = self.conv_layer(CA2.unsqueeze(1))
        CA2_conv = torch.mean(CA2_conv, dim=1, keepdim=False)

        # Dynamically adjust shapes
        CA1_conv = CA1_conv.view(CA1.size(0), -1)
        CA2_conv = CA2_conv.view(CA2.size(0), -1)
        CA1 = CA1.view(CA1.size(0), -1)
        CA2 = CA2.view(CA2.size(0), -1)

        # Align batch sizes using zero padding
        max_batch_size = max(CA1.size(0), CA1_conv.size(0), CA2_conv.size(0), CA2.size(0))

        # Apply zero padding to align all tensors to the max batch size
        CA1 = torch.cat([CA1, torch.zeros(max_batch_size - CA1.size(0), CA1.size(1)).to(CA1.device)], dim=0)
        CA1_conv = torch.cat([CA1_conv, torch.zeros(max_batch_size - CA1_conv.size(0), CA1_conv.size(1)).to(CA1_conv.device)], dim=0)
        CA2_conv = torch.cat([CA2_conv, torch.zeros(max_batch_size - CA2_conv.size(0), CA2_conv.size(1)).to(CA2_conv.device)], dim=0)
        CA2 = torch.cat([CA2, torch.zeros(max_batch_size - CA2.size(0), CA2.size(1)).to(CA2.device)], dim=0)

        # Concatenate tensors
        fusedoutput = torch.cat([CA1, CA1_conv, CA2_conv, CA2], dim=1)
        output = self.FFN(fusedoutput)

        return attn_tcr, attn_pep, attn_tcr_ca, attn_pep_ca, fusedoutput, output




#class TEPCAM(nn.Module):
# class TEPDA(nn.Module):
#     def __init__(self,d_model,batch_size,modelseed,n_heads):
#         super(TEPDA,self).__init__()
#         set_modelseed(modelseed)
#         self.d_model = d_model
#         self.batch_size = batch_size
#         self.tcr_embedding = Embeddings(vocab_size = 26, hidden_size = self.d_model, max_len =20)
#         self.pep_embedding = Embeddings(vocab_size = 26, hidden_size = self.d_model, max_len =15)
#         self.n_heads = n_heads
#         self.SALayer = MultiHeadAttention(self.d_model,self.d_model,self.n_heads//2,self.d_model)
#         self.CALayer = CrossAttention(self.d_model,self.n_heads,self.d_model)
#         self.conv_layer = nn.Sequential(
#             nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,padding=1),
#             nn.BatchNorm2d(4),
#             nn.GELU(),
#             nn.Conv2d(in_channels=4,out_channels=d_model//2,kernel_size=3,padding=1),
#             nn.BatchNorm2d(d_model//2),
#             nn.GELU(),
#             nn.Conv2d(d_model//2,d_model,3,padding=1),
#             nn.BatchNorm2d(d_model),
#             nn.GELU(),
#             )
#         #self.FFN = nn.Sequential(nn.Linear(self.d_model*max_len*2,1024),
#         self.FFN = nn.Sequential(nn.Linear(2240,1024),        
#                          nn.BatchNorm1d(1024),
#                                  nn.GELU(),
#                                  nn.Dropout(p=0.5),
#                                  nn.Linear(1024,128),
#                                  nn.BatchNorm1d(128),
#                                  nn.GELU(),
#                                  nn.Dropout(p=0.25),
#                                  nn.Linear(128,16),
#                                  nn.BatchNorm1d(16),
#                                  nn.GELU(),
#                                  nn.Linear(16,2),)
                
#    def forward(self, tcr, peptide):
 #       pep_embed = self.pep_embedding(peptide)
  #      tcr_embed = self.tcr_embedding(tcr) 
   #     tcr_attn_mask = get_attn_pad_mask(tcr,tcr) 
    #    pep_attn_mask = get_attn_pad_mask(peptide,peptide) 
     #   tcr2pep_mask = get_attn_pad_mask(tcr,peptide) 
      #  pep2tcr_mask = get_attn_pad_mask(peptide,tcr)
       # tcr_sa,attn_tcr = self.SALayer(tcr_embed,tcr_embed,tcr_embed,tcr_attn_mask)
       # pep_sa,attn_pep = self.SALayer(pep_embed,pep_embed,pep_embed,pep_attn_mask)
       # CA1,CA2,attn_tcr_ca,attn_pep_ca = self.CALayer(tcr_sa,pep_sa,tcr2pep_mask,pep2tcr_mask)
       # CA1_conv = self.conv_layer(CA1.unsqueeze(1))
       # CA1_conv = torch.mean(CA1_conv,dim=1,keepdim=False)
       # CA2_conv = self.conv_layer(CA2.unsqueeze(1))
       # CA2_conv = torch.mean(CA2_conv,dim=1,keepdim=False)
       # CA1_conv = CA1_conv.view(self.batch_size, -1)
        #CA2_conv = CA2_conv.view(self.batch_size, -1)
       # CA1 = CA1.view(self.batch_size,-1)
        #CA2 = CA2.view(self.batch_size,-1)

       # fusedoutput = torch.cat([CA1,CA1_conv,CA2_conv,CA2],dim=1)
       # output = self.FFN(fusedoutput)
       # return attn_tcr, attn_pep, attn_tcr_ca, attn_pep_ca, fusedoutput, output

