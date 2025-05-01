import os
import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")

AA_list = "ACDEFGHIKLMNPQRSTVWY"

def removeInvalidSequence(df):
    """
    Remove invalid CDR3 sequence.
    Definition of invalid CDR3 sequence: contains invalid amino acid residue, invalid sequence length (between min_seq_len
    and max_seq_len) or not start with cysteine and ends with tyrosine residue.
    """
    del_list = []
    invalid_sequence_list = []
    for idx, row in df.iterrows():
        if not row["CDR3"].startswith("C") or not row["CDR3"].endswith("F"):
            del_list.append(idx)
            invalid_sequence_list.append((row["CDR3"], "Invalid start or end"))

        for ch in row["CDR3"]:
            if ch not in 'ACDEFGHIKLMNPQRSTVWY':
                del_list.append(idx)
                invalid_sequence_list.append((row["CDR3"], "Invalid aa residue"))
                break

    df.drop(index=del_list, inplace=True)
    df.reset_index(drop=True)
    print("[Info] {} invalid sequences removed.".format(len(del_list)))
    return df, invalid_sequence_list

def removeLabelMissedSequence(df):
    df.dropna(subset=['CDR3', 'Epitope'], how='any', axis=0, inplace=True)
    df.reset_index(drop=True)
    return df

def removeInvalidLengthSequence(df, min_seq_len=9, max_seq_len=18):
    """
    Remove invalid length (between min_seq_len and max_seq_len) sequence.
    """
    del_list = []
    invalid_sequence_list = []
    for idx, row in df.iterrows():
        if len(row["CDR3"]) < min_seq_len or len(row["CDR3"]) > max_seq_len:
            del_list.append(idx)
            invalid_sequence_list.append((row["CDR3"], "Invalid sequence length"))

    df.drop(index=del_list, inplace=True)
    df.reset_index()
    print("[Info] {} invalid sequences removed.".format(len(del_list)))
    return df, invalid_sequence_list

def removeDuplicatedSequence(df):
    rows_0 = df.shape[0]
    df.drop_duplicates(subset=["CDR3", "Epitope"], keep="first", inplace=True)
    df.reset_index(drop=True)
    print("{} duplicated sequences removed.".format(rows_0 - df.shape[0]))
    return df


def epitopeStatistics(df, dataset, store_root='../data/statistics/', majority_epitope_threshold=100):
    """
    Majority epitope statistics
    """
    if not os.path.exists(store_root):
        os.mkdir(store_root)
    valid_sequences = 0
    epitope_number = 0
    epitope_list, num_list = [], []
    for epitope, data in df.groupby("Epitope"):
        if data.shape[0] >= majority_epitope_threshold:
            print("{:<25}{}".format(epitope, data.shape[0]))
            epitope_list.append(epitope)
            num_list.append(data.shape[0])
            valid_sequences += data.shape[0]
            epitope_number += 1
    print("Valid sequences {}".format(valid_sequences))
    print("Epitope number {}".format(epitope_number))
    statis_df = pd.DataFrame({'Epitope': epitope_list, 'number': num_list})
    statis_df.to_csv('{}{}.csv'.format(store_root, dataset), index=False)

def binaryClassificationDatasetConstruction(df, use_cols=["CDR3", "Epitope"],majority_epitope_threshold=100):
    """
    Construct binary classification training data set for majority epitopes.
    """
    df = df[use_cols]
    result=pd.DataFrame()

    for epitope, data in df.groupby("Epitope"):
        if data.shape[0] >= majority_epitope_threshold:
            # positive sequence
            positive_df = df[df["Epitope"] == epitope]
            positive_df.dropna(subset=['CDR3'], how='any', axis=0, inplace=True)
            positive_df.reset_index(drop=True)
            result=pd.concat([result,positive_df], ignore_index=True)

    return result

def positive_negative_dataset_builder(df, save_path, neg_samples=1):
    """
        Build train and test dataset.
        """
    assert 'CDR3' in df.columns.tolist() and 'Epitope' in df.columns.tolist(), "Invalid Data"
    def _dataset_builder(df, save_path, negative_samples=1):
        epitope_list = set(df['Epitope'])
        tcr_list, antigen_list, label_list = [], [], []
        for epitope, data in df.groupby('Epitope'):
            negative_epitope_list = epitope_list - set([epitope])

            for seq_cur in data['CDR3']:
                tcr_list.append(seq_cur)
                antigen_list.append(epitope)
                label_list.append(1)

                # sample negative sequences
                for i in range(negative_samples):
                    neg_epitope = np.random.choice(list(negative_epitope_list), size=1)
                    tcr_list.append(seq_cur)
                    antigen_list.append(neg_epitope[0])
                    label_list.append(0)

        df_save = pd.DataFrame({'tcr': tcr_list, 'antigen': antigen_list, 'label': label_list})
        df_save.to_csv(save_path+'clean_data.csv', index=False)
    cdr3, epitope = df['CDR3'], df['Epitope']
    df_save = pd.DataFrame({'CDR3': cdr3, 'Epitope': epitope,'label': label_list})
    _dataset_builder(df_save, save_path, neg_samples)
    return df_save

def create_parser():
    parser = argparse.ArgumentParser(description="Dataset build",formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument("--input_file",dest="input_file",type=str,help="The input file in .csv format.",required=True)
    parser.add_argument("--output_file", dest="output_file", type=str, help="The output file in .csv format.",required=True)
    parser.add_argument("--neg_samples", dest="neg_samples", default=1, type=int, help="The number of negative samples.",required=False)

    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = create_parser()

    df = pd.read_csv(args.input_file)
    df = removeLabelMissedSequence(df)
    df, info = removeInvalidSequence(df)
    df,_ = removeInvalidLengthSequence(df)
    df = removeDuplicatedSequence(df)
    df.reset_index(inplace=True)
    save_df = df[['CDR3', 'Epitope']]
    df=binaryClassificationDatasetConstruction(save_df, majority_epitope_threshold=50)
    positive_negative_dataset_builder(df, save_path=args.output_file, neg_samples=args.neg_samples)
    print("The work is done!")
