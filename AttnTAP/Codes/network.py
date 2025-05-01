import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
import numpy as np

from utils import sequencePadding


class TCRAntigenClassifier(nn.Module):
    """
    The Basic version of TCR-Antigen dual-input neural network.
    Note that this network use one-hot encoded inputs.
    """
    def __init__(self, max_tcr_len=18, max_antigen_len=9):
        super(TCRAntigenClassifier, self).__init__()
        self.max_tcr_len = max_tcr_len
        self.max_antigen_len = max_antigen_len
        self.fc1 = nn.Sequential(
            nn.Linear(21 * self.max_tcr_len, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(21 * self.max_antigen_len, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(96, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, tcr, antigen):
        tcr = tcr.view(-1, 21 * self.max_tcr_len)
        antigen = antigen.view(-1, 21 * self.max_antigen_len)
        tcr_embedding = self.fc1(tcr)
        antigen_embedding = self.fc2(antigen)
        combine = torch.cat([tcr_embedding, antigen_embedding], 1)
        output = self.fc3(combine)
        return output


class TCRAntigenDataset(Dataset):
    def __init__(self, sequences, labels, aa2idx, max_tcr_len=18, max_antigen_len=9):
        self.aa2idx = aa2idx
        self.max_tcr_len = max_tcr_len
        self.max_antigen_len = max_antigen_len
        self.data = self.__init_data(sequences, labels)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __init_data(self, sequences, labels):
        ret_list = []
        for idx, sequence in enumerate(sequences):
            tcr_padding = sequencePadding(sequence[0], max_len=self.max_tcr_len, aa2idx=self.aa2idx)
            antigen_padding = sequencePadding(sequence[1], max_len=self.max_antigen_len, aa2idx=self.aa2idx)
            ret_list.append((tcr_padding, antigen_padding, labels[idx]))
        return ret_list


class DoubleLSTMClassifier(nn.Module):
    """
    The two layers stacked-LSTM based TCR-Antigen dual-input neural network.
    Note that this network is the original network provided by ERGO which did not use word2vec pretrained embedding.
    """
    def __init__(self, embedding_dim, lstm_dim, dropout_rate, bidirectional=True):
        super(DoubleLSTMClassifier, self).__init__()

        # Dimension parameters
        self. embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim * 2 if bidirectional else lstm_dim
        self.dropout_rate = dropout_rate

        # Embedding metrics - 20 amino acidas + padding
        self.tcr_embedding = nn.Embedding(21, embedding_dim, padding_idx=0)
        self.pep_embedding = nn.Embedding(21, embedding_dim, padding_idx=0)

        # LSTM layers
        # nn.LSTM(input_embedding_dim, hidden_dim, number_of_layers, batch_first)
        self.tcr_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_dim, num_layers=2, batch_first=True, dropout=dropout_rate, bidirectional=bidirectional)
        self.pep_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_dim, num_layers=2, batch_first=True, dropout=dropout_rate, bidirectional=bidirectional)

        # Fully connected layers
        self.hidden_layer = nn.Linear(self.lstm_dim * 2, lstm_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(lstm_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def lstm_pass(self, lstm, padded_embeds, lengths):
        # Before using PyTorch pack_padded_sequence we need to order the sequences batch by descending sequence length
        lengths, perm_idx = lengths.sort(0, descending=True)
        padded_embeds = padded_embeds[perm_idx]

        # Pack the batch and ignore the padding
        padded_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths, batch_first=True)

        # Feed into LSTM
        lstm_out, hidden = lstm(padded_embeds)

        # Unpack the batch after the LSTM
        lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Remember that outputs are sorted but we want the original ordering
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        lengths = lengths[unperm_idx]

        return lstm_out

    def forward(self, tcr, tcr_lens, pep, pep_lens):
        tcr_embed = self.tcr_embedding(tcr)
        # Decode hidden state of last time step for tcr sequence
        tcr_lstm_out = self.lstm_pass(self.tcr_lstm, tcr_embed, tcr_lens)
        tcr_last_cell = torch.cat([tcr_lstm_out[i, j.data - 1] for i, j in enumerate(tcr_lens)]).view(len(tcr_lens), self.lstm_dim)

        pep_embed = self.pep_embedding(pep)
        # Decode hidden state of last time step for antigen sequence
        pep_lstm_out = self.lstm_pass(self.pep_lstm, pep_embed, pep_lens)
        pep_last_cell = torch.cat([pep_lstm_out[i, j.data - 1] for i, j in enumerate(pep_lens)]).view(len(pep_lens), self.lstm_dim)

        # MLP Classifier
        tcr_pep_concat = torch.cat([tcr_last_cell, pep_last_cell], 1)

        #特征向量层
        hidden_output = self.dropout(self.relu(self.hidden_layer(tcr_pep_concat)))

        output = self.output_layer(hidden_output)
        output = F.sigmoid(output)
        return output


class AttentionBiLSTMClassifier(nn.Module):
    """
    This is the Attention based Bi-LSTM network for TCR-Antigen binding prediction.
    God Bless it.
    """
    def __init__(self, embedding_dim=10, hidden_dim=50, max_tcr_len=18, max_pep_len=9,
                 bidirectional=True, lstm_layer=1, dropout_rate=0.1, pretrained_embeddings=None):
        super(AttentionBiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_tcr_len = max_tcr_len
        self.max_pep_len = max_pep_len
        self.bidirectional = bidirectional
        self.lstm_layer = lstm_layer
        self.lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Embedding metrics - 20 amino acidas + padding
        self.tcr_embedding = nn.Embedding(21, embedding_dim)
        self.pep_embedding = nn.Embedding(21, embedding_dim)

        # Load pretrained embedding
        if pretrained_embeddings is not None:
            pretrained_weight = self.load_embedding_from_file(pretrained_embeddings)
            self.tcr_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.pep_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

        # LSTM layer that extract feature from tcr
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, lstm_layer, bidirectional=bidirectional, batch_first=True)

        # Fully connected layer that extract feature from peptide
        self.pep_fc = nn.Sequential(
            nn.Linear(self.embedding_dim * self.max_pep_len, self.hidden_dim),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )

        # Fully connected layer that extract feature from tcr
        self.tcr_fc = nn.Sequential(
            nn.Linear(self.embedding_dim * self.max_tcr_len, self.hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim + self.lstm_out_dim, self.hidden_dim),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(self.hidden_dim),
            # nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2)
        )

        self.w_omega = nn.Parameter(torch.Tensor(self.lstm_out_dim, self.lstm_out_dim))
        self.u_omega = nn.Parameter(torch.Tensor(self.lstm_out_dim, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def load_embedding_from_file(self, pretrained_embedding_file, aa_list="ACDEFGHIKLMNPQRSTVWY"):
        embedding = KeyedVectors.load_word2vec_format(pretrained_embedding_file)
        embedding_list = []
        embedding_list.append(np.zeros(self.embedding_dim))
        for aa in aa_list:
            embedding_list.append(embedding.wv[aa])
        return np.array(embedding_list)

    def attention_structure1(self, lstm_out):
        # lstm_out: [batch, seq_len, hidden_dim * 2]
        # u: [batch, seq_len, hidden_dim * 2]
        u = torch.tanh(torch.matmul(lstm_out, self.w_omega))
        # att: [batch, seq_len, 1]
        att = torch.matmul(u, self.u_omega)
        att_score = torch.softmax(att, dim=1)
        scored_lstm_out = lstm_out * att_score
        # context: [batch, hidden_dim * 2]
        context = torch.sum(scored_lstm_out, dim=1)
        return context, att_score

    def forward(self, tcr, pep):
        tcr_embed = self.tcr_embedding(tcr)

        # Att-LSTM feature
        lstm_out, (h, c) = self.lstm(tcr_embed)
        att_output, att_score = self.attention_structure1(lstm_out)
        #print(tcr[:5, :], att_score.reshape(-1, 18)[:5, :])

        # MLP feature
        # tcr_embed = tcr_embed.reshape(-1, self.max_tcr_len * self.embedding_dim)
        # att_output = self.tcr_fc(tcr_embed)

        # LSTM hidden layer output feature
        # lstm_out, (h, c) = self.lstm(tcr_embed)
        # att_output = h.reshape(-1, self.hidden_dim)

        # print(att_score)
        pep_embed = self.pep_embedding(pep)
        pep_embed = pep_embed.reshape(-1, self.max_pep_len * self.embedding_dim)
        pep_out = self.pep_fc(pep_embed)

        # lstm_out, (h, c) = self.lstm(pep_embed)
        # pep_out, att_score = self.attention_structure1(lstm_out)

        concat = torch.cat([pep_out, att_output], dim=1)
        #print(concat.shape)

        output = self.fc(concat)
        return output


class AttnBLSTMDataset(Dataset):
    def __init__(self, sequences, labels, max_tcr_len=18, max_antigen_len=9):
        self.max_tcr_len = max_tcr_len
        self.max_antigen_len = max_antigen_len
        self.data = self.__init_data(sequences, labels)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __init_data(self, sequences, labels):
        ret_list = []
        for idx, sequence in enumerate(sequences):
            tcr, pep = sequence
            tcr_padding = self.__sequence_padding_left(tcr, self.max_tcr_len)
            pep_padding = self.__sequence_padding_left(pep, self.max_antigen_len)
            #ret_list.append((tcr_padding, pep_padding, labels[idx]))
            ret_list.append((tcr_padding, pep_padding, labels[idx],tcr,pep))
        return ret_list

    def __sequence_padding_left(self, sequence, max_len):
        aa_list = "-ACDEFGHIKLMNPQRSTVWY"
        sequence = sequence[:max_len] if len(sequence) >= max_len else sequence + '-' * (max_len - len(sequence))
        ret = [aa_list.index(aa) for aa in sequence]
        return torch.tensor(ret)

    def __sequence_padding_middle(self, sequence, max_len):
        aa_list = "-ACDEFGHIKLMNPQRSTVWY"
        sequence = sequence[:max_len] if len(sequence) >= max_len else '-' * math.floor((max_len - len(sequence)) // 2) + sequence + '-' * (max_len - len(sequence) - math.floor((max_len - len(sequence)) // 2))
        ret = [aa_list.index(aa) for aa in sequence]
        return torch.tensor(ret)


class AttentionBiLSTM3merClassifier(nn.Module):
    """
    This is the Attention based Bi-LSTM network for TCR-Antigen binding prediction.
    God Bless it.
    """
    def __init__(self, embedding_dim=50, hidden_dim=50, max_tcr_len=18, max_pep_len=9,
                 bidirectional=True, lstm_layer=1, dropout_rate=0.1, pretrained_embeddings=None, pretrained_3mer_embeddings=None):
        super(AttentionBiLSTM3merClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_tcr_len = max_tcr_len
        self.max_pep_len = max_pep_len
        self.bidirectional = bidirectional
        self.lstm_layer = lstm_layer
        self.lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.kmer_dict = ['None'] + list(KeyedVectors.load_word2vec_format(pretrained_3mer_embeddings).wv.vocab)
        print(len(self.kmer_dict))

        # Embedding metrics - 20 amino acidas + padding
        self.tcr_embedding = nn.Embedding(len(self.kmer_dict), embedding_dim)
        self.pep_embedding = nn.Embedding(21, 10)

        # Load pretrained embedding
        if pretrained_embeddings is not None:
            pretrained_weight = self.load_embedding_from_file(pretrained_embeddings)
            self.pep_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

        if pretrained_3mer_embeddings is not None:
            pretrained_3mer_weight = self.load_3mer_embedding_from_file(pretrained_3mer_embeddings)
            self.tcr_embedding.weight.data.copy_(torch.from_numpy(pretrained_3mer_weight))

        # LSTM layer that extract feature from tcr
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, lstm_layer, dropout=0.2, bidirectional=bidirectional, batch_first=True)

        # Fully connected layer that extract feature from peptide
        self.pep_fc = nn.Sequential(
            nn.Linear(10 * self.max_pep_len, self.hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )

        # Fully connected layer that extract feature from tcr
        self.tcr_fc = nn.Sequential(
            nn.Linear(self.embedding_dim * self.max_tcr_len, self.hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim + self.lstm_out_dim, self.hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2)
        )

        self.w_omega = nn.Parameter(torch.Tensor(self.lstm_out_dim, self.lstm_out_dim))
        self.u_omega = nn.Parameter(torch.Tensor(self.lstm_out_dim, 1))

        nn.init.uniform(self.w_omega, -0.1, 0.1)
        nn.init.uniform(self.u_omega, -0.1, 0.1)

    def load_embedding_from_file(self, pretrained_embedding_file, aa_list="ACDEFGHIKLMNPQRSTVWY"):
        embedding = KeyedVectors.load_word2vec_format(pretrained_embedding_file)
        embedding_list = []
        embedding_list.append(np.zeros(10))
        for aa in aa_list:
            embedding_list.append(embedding.wv[aa])
        return np.array(embedding_list)

    def load_3mer_embedding_from_file(self, pretrained_embedding_file):
        embedding = KeyedVectors.load_word2vec_format(pretrained_embedding_file)
        embedding_list = []
        embedding_list.append(np.zeros(self.embedding_dim))
        for kmer in list(embedding.wv.vocab.keys()):
            embedding_list.append(embedding.wv[kmer])
        return np.array(embedding_list)

    def attention_structure1(self, lstm_out):
        # lstm_out: [batch, seq_len, hidden_dim * 2]
        # u: [batch, seq_len, hidden_dim * 2]
        u = torch.tanh(torch.matmul(lstm_out, self.w_omega))
        # att: [batch, seq_len, 1]
        att = torch.matmul(u, self.u_omega)
        att_score = torch.softmax(att, dim=1)
        scored_lstm_out = lstm_out * att_score
        # context: [batch, hidden_dim * 2]
        context = torch.sum(scored_lstm_out, dim=1)
        return context, att_score

    def forward(self, tcr, pep):
        tcr_embed = self.tcr_embedding(tcr)

        # Att-LSTM feature
        lstm_out, (h, c) = self.lstm(tcr_embed)
        att_output, att_score = self.attention_structure1(lstm_out)
        #print(tcr[:5, :], att_score.reshape(-1, 18)[:5, :])

        # MLP feature
        # tcr_embed = tcr_embed.reshape(-1, self.max_tcr_len * self.embedding_dim)
        # att_output = self.tcr_fc(tcr_embed)

        # LSTM hidden layer output feature
        # lstm_out, (h, c) = self.lstm(tcr_embed)
        # att_output = h.reshape(-1, self.hidden_dim)

        # print(att_score)
        pep_embed = self.pep_embedding(pep)
        pep_embed = pep_embed.reshape(-1, self.max_pep_len * 10)
        pep_out = self.pep_fc(pep_embed)

        # lstm_out, (h, c) = self.lstm(pep_embed)
        # pep_out, att_score = self.attention_structure1(lstm_out)

        concat = torch.cat([pep_out, att_output], dim=1)
        output = self.fc(concat)
        return output


class AttnBLSTM3merDataset(Dataset):
    """
    Dataset for 3mer
    """
    def __init__(self, sequences, labels, max_tcr_len=18, max_antigen_len=9, embedding_file=None):
        self.max_tcr_len = max_tcr_len
        self.max_antigen_len = max_antigen_len
        self.kmer_dict = ['None'] + list(KeyedVectors.load_word2vec_format(embedding_file).vocab)
        self.data = self.__init_data(sequences, labels)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __init_data(self, sequences, labels):
        ret_list = []
        for idx, sequence in enumerate(sequences):
            tcr, pep = sequence
            tcr_padding = self.__sequence_padding_tcr(tcr, self.max_tcr_len)
            pep_padding = self.__sequence_padding_pep(pep, self.max_antigen_len)
            ret_list.append((tcr_padding, pep_padding, labels[idx]))
        return ret_list

    def __sequence_padding_tcr(self, sequence, max_len):
        kmer_list = []
        for i in range(len(sequence) - 2):
            kmer_list.append(sequence[i: i + 3])
        if len(kmer_list) > max_len - 2:
            kmer_list = kmer_list[:max_len - 2]
        if len(kmer_list) < max_len - 2:
            kmer_list = kmer_list + ['None'] * (max_len - 2 - len(kmer_list))
        ret = [self.kmer_dict.index(kmer) if kmer in self.kmer_dict else 0 for kmer in kmer_list]
        return torch.tensor(ret)

    def __sequence_padding_pep(self, sequence, max_len):
        aa_list = "-ACDEFGHIKLMNPQRSTVWY"
        sequence = sequence[:max_len] if len(sequence) >= max_len else sequence + '-' * (max_len - len(sequence))
        ret = [aa_list.index(aa) for aa in sequence]
        return torch.tensor(ret)
