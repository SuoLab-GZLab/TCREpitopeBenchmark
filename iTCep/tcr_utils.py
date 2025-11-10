import pandas as pd
import numpy as np
from Bio.Alphabet import IUPAC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from collections import Counter


class TCRUtils:
    file_path = 'data/'
    # only keep IUPAC protein letters
    aa_list = list(IUPAC.IUPACProtein.letters)  # "ACDEFGHIKLMNPQRSTVWY"
    # aa_list = 'ARNDCQEGHILKMFPSTWYV'
    max_pep_len = 11
    max_cdr3_len = 21

    def __init__(self, max_pep_len=11, max_cdr3_len=21):
        self.max_pep_len = max_pep_len
        self.max_cdr3_len = max_cdr3_len

    @classmethod
    def delete_space(cls, xstr):  # cls.cal_name调用类自己的数据属性
        """
        delete space from a string
        :return: str0
        """
        str0 = ''
        for a in xstr:
            if a != ' ':
                str0 = str0 + a
        return str0

    @classmethod
    def load_file(cls, data_file):
        dataset = pd.read_csv(data_file)
        peptide_seqs = dataset.peptide.tolist()
        cdr3_seqs = dataset.CDR3.tolist()
        return peptide_seqs, cdr3_seqs

    @classmethod
    def data_processing(cls, data_df, pos_data=None):
        """
        Drop data which are missing, duplicate or with wrong format.
        :param dataset: dataset to be processed
        :param pos_data: set the dataset to positive or not
        :return: processed dataset
        """
        dataset = data_df.copy()
        # drop data with missing values in the 'peptide' and 'CDR3' columns
        dataset = dataset.dropna(axis=0, subset=['peptide', 'CDR3'])
        # delete spaces in the sequences
        dataset.loc[:, 'peptide'] = dataset['peptide'].apply(cls.delete_space)
        dataset.loc[:, 'CDR3'] = dataset['CDR3'].apply(cls.delete_space)
        # drop duplicate
        # dataset = dataset.drop_duplicates(subset=['peptide', 'CDR3'])
        # keep peptides between 8 and 11 in length
        dataset = dataset[dataset['peptide'].str.contains(r'^[A-Z]{7,10}[A-Z]$')]
        # drop TCRs longer than 21 and that do not start with C and end with F
        dataset = dataset[dataset['CDR3'].str.contains(r'^[C][A-Z]{6,19}[F]$')]
        # drop seqs with wrong amino acid
        dataset = dataset[~dataset['peptide'].str.contains(r'[B]|[J]|[O]|[U]|[X]|[Z]')]
        dataset = dataset[~dataset['CDR3'].str.contains(r'[B]|[J]|[O]|[U]|[X]|[Z]')]
        if pos_data is not None:
            # add label column
            positive_col_name = dataset.columns.tolist()
            positive_col_name.insert(0, 'label')
            dataset = dataset.reindex(columns=positive_col_name)
            dataset['label'] = 1 if pos_data else 0
        return dataset

    @classmethod
    def peptide_processing(cls, pep_df):
        dataset = pep_df.copy()
        dataset = dataset.dropna()
        dataset.loc[:, 'peptide'] = dataset['peptide'].apply(cls.delete_space)
        dataset = dataset[dataset['peptide'].str.contains(r'^[A-Z]{7,10}[A-Z]$')]
        dataset = dataset[~dataset['peptide'].str.contains(r'[B]|[J]|[O]|[U]|[X]|[Z]')]
        return dataset

    @classmethod
    def get_onehot(cls):
        # numpy快速生成one hot
        aa_onehot = np.eye(20, 20).tolist()
        # aa_onehot.append([0.0] * 21)
        # 将list转为dict
        onehot_dict = dict(zip(cls.aa_list, aa_onehot))
        return onehot_dict

    @classmethod
    def get_phychem(cls, normal=True):
        """
        # hydrophobicity 疏水性
        # Molecular_weight 分子量
        # Bulkiness 膨松性
        # Polarity / Grantham 极性
        # Recognition factors
        # Hphob. OMH / Sweet et al.
        # Hphob. HPLC / Wilson & al
        # Ratio hetero end/side
        # Average flexibility
        # beta-sheet / Chou & Fasman
        # alpha-helix / Deleage & Ro
        # beta-turn / Deleage & Roux
        # Relative mutability 相对稳定性
        # Number of codon(s) coding for each amino acid in universal genetic code
        # Refractivity 折射性
        # Transmembrane tendency 跨膜倾向性
        # accessible residues
        # Average area buried
        # Conformational parameter for coil 线圈构象参数
        # Total beta-strand
        # Parallel beta-strand
        :return: a dict containing amino acid physical and chemical properties
        """
        aa_physicochemical = pd.read_excel(cls.file_path + 'amino_acid_phyche.xlsx', index_col=1)
        aa_physicochemical = aa_physicochemical.iloc[:, 1:]
        phychem_df = aa_physicochemical.copy()
        if normal:
            phychem_df = cls.set_normalization(phychem_df)
            phychem_df = pd.DataFrame(phychem_df, columns=aa_physicochemical.columns, index=aa_physicochemical.index)
        # phychem_df.loc['X'] = [0] * phychem_df.shape[1]
        # convert each feature to dict
        # feat_names = ['hydrophobicity', 'molecular_weight', 'bulkiness', 'polarity', 'recognition',
        #               'OMH', 'HPLC', 'ratio_hetero', 'flexibility', 'beta_sheet', 'alpha_helix',
        #               'beta-turn', 'mutability', 'num_of_codon', 'refractivity', 'transmembrane_tend',
        #               'residues', 'aver_area_buried', 'conformational_param', 'total_beta_strand',
        #               'parallel_beta_strand']
        phychem_dict = {}
        for i in range(len(cls.aa_list)):
            phychem_dict[cls.aa_list[i]] = phychem_df.loc[cls.aa_list[i]]
        return phychem_dict

    @classmethod
    def feat_encode(cls, seqs, max_length, encode_dict):
        """
        encoding sequences using 'encode_dict'
        :param seqs: sequences to encode
        :param max_length: the maximum length of the encoded sequences
        :param encode_dict: encode matrix
        :return seq_feats: encoded sequences
        """
        seq_feats = []
        for seq in seqs:
            f_seq = np.zeros((max_length, len(encode_dict[list(encode_dict.keys())[0]])))
            for i, aa in enumerate(seq):
                f_seq[i] = encode_dict[aa]
            seq_feats.append(f_seq)
        return np.array(seq_feats)


    @classmethod
    def dp_MED(cls, str1, str2):
        L1 = len(str1) + 1
        L2 = len(str2) + 1
        matrix = [[i + j for j in range(L2)] for i in range(L1)]
        for i in range(1, L1):
            for j in range(1, L2):
                d = 0 if (str1[i - 1] == str2[j - 1]) else 1
                matrix[i][j] = min(matrix[i - 1][j] + 1,
                                matrix[i][j - 1] + 1,
                                matrix[i - 1][j - 1] + d)
        return matrix[-1][-1]

    @classmethod
    def similar_peptide(cls, peptide, refer_peps: list):
        res_pep = refer_peps[0]
        min_dis = cls.dp_MED(peptide, res_pep)
        # 计算peptide和阳性肽段的相似性
        for pep in refer_peps:
            temp = cls.dp_MED(peptide, pep)
            if temp < min_dis:
                min_dis = temp
                res_pep = pep
        return res_pep

    @classmethod
    def aapp_encode_TCR(cls, seqs):
        """
        根据每个肽段对应CDR3的位置特征计算氨基酸位置频率，由于CDR3位置1必为C，所以仅计算位置2到位置21的位置频率。
        CDR3 aa distribution feature map. Encoding data with AAPP method.
        在编码时已经normalizaed，无需在此处重复该操作
        :return: Amino Acid Position Preference encode for CDR3 response to a particular peptide
        """
        aapp_encode = []
        aapp_dict_tcr = np.load(cls.file_path + 'aapp_dict_tcr.npy', allow_pickle=True).item()
        for seq in seqs:
            feats_arr = aapp_dict_tcr[seq] if seq in aapp_dict_tcr.keys() else aapp_dict_tcr[
                cls.similar_peptide(seq, list(aapp_dict_tcr.keys()))]
            aapp_encode.append(feats_arr.T)
        return np.array(aapp_encode)

    @classmethod
    def set_normalization(cls, data_arr: np.ndarray):
        scaler = StandardScaler()
        return scaler.fit_transform(data_arr)

    @classmethod
    def labels_encode(cls, labels):
        one_hot_encoder = OneHotEncoder(categories='auto')
        labels = np.array(labels).reshape(-1, 1)
        labels_encode = one_hot_encoder.fit_transform(labels).toarray()
        return labels_encode




