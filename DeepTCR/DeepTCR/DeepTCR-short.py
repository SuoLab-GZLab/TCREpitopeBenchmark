import sys
sys.path.append('../')
from DeepTCR.functions.Layers import *
from DeepTCR.functions.utils_u import *
from DeepTCR.functions.utils_s import *
from DeepTCR.functions.act_fun import *
from DeepTCR.functions.plot_func import *
import seaborn as sns
import colorsys
from scipy.cluster.hierarchy import linkage,fcluster,dendrogram, leaves_list
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import pdist, squareform
import umap
from sklearn.cluster import DBSCAN,KMeans
import sklearn
import DeepTCR.phenograph as phenograph
from scipy.spatial import distance
import glob
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import shutil
import warnings
from scipy.stats import spearmanr,gaussian_kde
from distinctipy import distinctipy
from tqdm import tqdm

class DeepTCR_base(object):

    def __init__(self,Name,max_length=40,device=0,tf_verbosity=3):

        #Assign parameters
        self.Name = Name
        self.max_length = max_length
        self.use_beta = False
        self.use_alpha = False
        self.device = '/device:GPU:'+str(device)
        self.use_v_beta = False
        self.use_d_beta = False
        self.use_j_beta = False
        self.use_v_alpha = False
        self.use_j_alpha = False
        self.use_hla = False
        self.use_hla_sup = False
        self.keep_non_supertype_alleles = False
        self.regression = False
        self.use_w = False
        self.ind = None
        self.unknown_str = '__unknown__'

        #Create dataframes for assigning AA to ints
        aa_idx, aa_mat = make_aa_df()
        aa_idx_inv = {v: k for k, v in aa_idx.items()}
        self.aa_idx = aa_idx
        self.aa_idx_inv = aa_idx_inv

        #Create directory for results of analysis
        directory = os.path.join(self.Name,'results')
        self.directory_results = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        #Create directory for any temporary files
        directory = self.Name
        if not os.path.exists(directory):
            os.makedirs(directory)

        tf.compat.v1.disable_eager_execution()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_verbosity)

    def Get_Data(self,directory,Load_Prev_Data=False,classes=None,type_of_data_cut='Fraction_Response',data_cut=1.0,n_jobs=40,
                    aa_column_alpha = None,aa_column_beta = None, count_column = None,sep='\t',aggregate_by_aa=True,
                    v_alpha_column=None,j_alpha_column=None,
                    v_beta_column=None,j_beta_column=None,d_beta_column=None,
                 p=None,hla=None,use_hla_supertype=False,keep_non_supertype_alleles=False):
        """
        # Get Data for DeepTCR
        Parse Data into appropriate inputs for neural network from directories where data is stored.

        This method can be used when your data is stored in directories and you want to load it from directoreis into DeepTCR. This method takes care of all pre-processing of the data including:

         - Combining all CDR3 sequences with the same nucleotide sequence (optional).
         - Removing any sequences with non-IUPAC characters.
         - Removing any sequences that are longer than the max_length set when initializing the training object.
         - Determining how much of the data per file to use (type_of_data_cut)
         - Whether to use HLA/HLA-supertypes during training.

        This method is included in the three main DeepTCR objects:

        - DeepTCR_U (unsupervised)
        - DeepTCR_SS (supervised sequence classifier/regressor)
        - DeepTCR_WF (supervised repertoire classifier/regressor)

        Args:
            directory (str): Path to directory with folders with tsv/csv files are present for analysis. Folders names become           labels for files within them. 
            Load_Prev_Data (bool): Loads Previous Data. This allows the user to run the method once, and then set this parameter to True to reload the data from a local pickle file.

            classes (list): Optional selection of input of which sub-directories to use for analysis.

            type_of_data_cut (str): Method by which one wants to sample from the TCRSeq File.

        Returns:
            variables into training object

            - self.alpha_sequences (ndarray): array with alpha sequences (if provided)
            - self.beta_sequences (ndarray): array with beta sequences (if provided)
            - self.class_id (ndarray): array with sequence class labels
            - self.sample_id (ndarray): array with sequence file labels
            - self.freq (ndarray): array with sequence frequencies from samples
            - self.counts (ndarray): array with sequence counts from samples
            - self.(v/d/j)_(alpha/beta) (ndarray): array with sequence (v/d/j)-(alpha/beta) usage

        """

        if Load_Prev_Data is False:

            if aa_column_alpha is not None:
                self.use_alpha = True

            if aa_column_beta is not None:
                self.use_beta = True

            if v_alpha_column is not None:
                self.use_v_alpha = True

            if j_alpha_column is not None:
                self.use_j_alpha = True

            if v_beta_column is not None:
                self.use_v_beta = True

            if d_beta_column is not None:
                self.use_d_beta = True

            if j_beta_column is not None:
                self.use_j_beta = True


            #Determine classes based on directory names
            data_in_dirs = True
            if classes is None:
                classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory,d))]
                classes = [f for f in classes if not f.startswith('.')]
                if not classes:
                    classes = ['None']
                    data_in_dirs = False


            self.lb = LabelEncoder()
            self.lb.fit(classes)
            self.classes = self.lb.classes_

            if p is None:
                p_ = Pool(n_jobs)
            else:
                p_ = p

            if sep == '\t':
                ext = '*.tsv'
            elif sep == ',':
                ext = '*.csv'
            else:
                print('Not Valid Delimiter')
                return

            #Get data from tcr-seq files
            alpha_sequences = []
            beta_sequences = []
            v_beta = []
            d_beta = []
            j_beta = []
            v_alpha = []
            j_alpha = []
            label_id = []
            file_id = []
            freq = []
            counts=[]
            file_list = []
            seq_index = []
            print('Loading Data...')
            print('self.classes',self.classes)
            for type in self.classes:
                if data_in_dirs:
                    files_read = glob.glob(os.path.join(directory, type, ext))
                else:
                    files_read = glob.glob(os.path.join(directory,ext))
                num_ins = len(files_read)
                args = list(zip(files_read,
                                [type_of_data_cut] * num_ins,
                                [data_cut] * num_ins,
                                [aa_column_alpha] * num_ins,
                                [aa_column_beta] * num_ins,
                                [count_column] * num_ins,
                                [sep] * num_ins,
                                [self.max_length]*num_ins,
                                [aggregate_by_aa]*num_ins,
                                [v_beta_column]*num_ins,
                                [d_beta_column]*num_ins,
                                [j_beta_column]*num_ins,
                                [v_alpha_column]*num_ins,
                                [j_alpha_column]*num_ins))

                DF = p_.starmap(Get_DF_Data, args)

                DF_temp = []
                files_read_temp = []
                for df,file in zip(DF,files_read):
                    if df.empty is False:
                        DF_temp.append(df)
                        files_read_temp.append(file)

                DF = DF_temp
                files_read = files_read_temp

                for df, file in zip(DF, files_read):
                    if aa_column_alpha is not None:
                        alpha_sequences += df['alpha'].tolist()
                    if aa_column_beta is not None:
                        beta_sequences += df['beta'].tolist()

                    if v_alpha_column is not None:
                        v_alpha += df['v_alpha'].tolist()

                    if j_alpha_column is not None:
                        j_alpha += df['j_alpha'].tolist()

                    if v_beta_column is not None:
                        v_beta += df['v_beta'].tolist()

                    if d_beta_column is not None:
                        d_beta += df['d_beta'].tolist()

                    if j_beta_column is not None:
                        j_beta += df['j_beta'].tolist()

                    label_id += [type] * len(df)
                    file_id += [file.split('/')[-1]] * len(df)
                    file_list.append(file.split('/')[-1])
                    freq += df['Frequency'].tolist()
                    counts += df['counts'].tolist()
                    seq_index += df.index.tolist()

            alpha_sequences = np.asarray(alpha_sequences)
            beta_sequences = np.asarray(beta_sequences)
            v_beta = np.asarray(v_beta)
            d_beta = np.asarray(d_beta)
            j_beta = np.asarray(j_beta)
            v_alpha = np.asarray(v_alpha)
            j_alpha = np.asarray(j_alpha)
            label_id = np.asarray(label_id)
            file_id = np.asarray(file_id)
            freq = np.asarray(freq)
            counts = np.asarray(counts)
            seq_index = np.asarray(seq_index)

            Y = self.lb.transform(label_id)
            OH = OneHotEncoder(sparse=False,categories='auto')
            Y = OH.fit_transform(Y.reshape(-1,1))

            print('Embedding Sequences...')
            #transform sequences into numerical space
            if aa_column_alpha is not None:
                args = list(zip(alpha_sequences, [self.aa_idx] * len(alpha_sequences), [self.max_length] * len(alpha_sequences)))
                result = p_.starmap(Embed_Seq_Num, args)
                sequences_num = np.vstack(result)
                X_Seq_alpha = np.expand_dims(sequences_num, 1)

            if aa_column_beta is not None:
                args = list(zip(beta_sequences, [self.aa_idx] * len(beta_sequences),  [self.max_length] * len(beta_sequences)))
                result = p_.starmap(Embed_Seq_Num, args)
                sequences_num = np.vstack(result)
                X_Seq_beta = np.expand_dims(sequences_num, 1)


            if p is None:
                p_.close()
                p_.join()

            if self.use_alpha is False:
                X_Seq_alpha = np.zeros(shape=[len(label_id)])
                alpha_sequences = np.asarray([None]*len(label_id))

            if self.use_beta is False:
                X_Seq_beta = np.zeros(shape=[len(label_id)])
                beta_sequences = np.asarray([None]*len(label_id))

            #transform v/d/j genes into categorical space
            num_seq = X_Seq_alpha.shape[0]
            if self.use_v_beta is True:
                self.lb_v_beta = LabelEncoder()
                self.lb_v_beta.classes_ = np.insert(np.unique(v_beta), 0, self.unknown_str)
                v_beta_num = self.lb_v_beta.transform(v_beta)
            else:
                self.lb_v_beta = LabelEncoder()
                v_beta_num = np.zeros(shape=[num_seq])
                v_beta = np.asarray([None]*len(label_id))

            if self.use_d_beta is True:
                self.lb_d_beta = LabelEncoder()
                self.lb_d_beta.classes_ = np.insert(np.unique(d_beta), 0, self.unknown_str)
                d_beta_num = self.lb_d_beta.transform(d_beta)
            else:
                self.lb_d_beta = LabelEncoder()
                d_beta_num = np.zeros(shape=[num_seq])
                d_beta = np.asarray([None]*len(label_id))

            if self.use_j_beta is True:
                self.lb_j_beta = LabelEncoder()
                self.lb_j_beta.classes_ = np.insert(np.unique(j_beta), 0, self.unknown_str)
                j_beta_num = self.lb_j_beta.transform(j_beta)
            else:
                self.lb_j_beta = LabelEncoder()
                j_beta_num = np.zeros(shape=[num_seq])
                j_beta = np.asarray([None]*len(label_id))

            if self.use_v_alpha is True:
                self.lb_v_alpha = LabelEncoder()
                self.lb_v_alpha.classes_ = np.insert(np.unique(v_alpha), 0, self.unknown_str)
                v_alpha_num = self.lb_v_alpha.transform(v_alpha)
            else:
                self.lb_v_alpha = LabelEncoder()
                v_alpha_num = np.zeros(shape=[num_seq])
                v_alpha = np.asarray([None]*len(label_id))

            if self.use_j_alpha is True:
                self.lb_j_alpha = LabelEncoder()
                self.lb_j_alpha.classes_ = np.insert(np.unique(j_alpha), 0, self.unknown_str)
                j_alpha_num = self.lb_j_alpha.transform(j_alpha)
            else:
                self.lb_j_alpha = LabelEncoder()
                j_alpha_num = np.zeros(shape=[num_seq])
                j_alpha = np.asarray([None]*len(label_id))

            if hla is not None:
                self.use_hla = True
                hla_df = pd.read_csv(hla)
                if use_hla_supertype:
                    hla_df = supertype_conv(hla_df,keep_non_supertype_alleles)
                    self.use_hla_sup = True
                    self.keep_non_supertype_alleles = keep_non_supertype_alleles
                hla_df = hla_df.set_index(hla_df.columns[0])
                hla_id = []
                hla_data = []
                for i in hla_df.iterrows():
                    hla_id.append(i[0])
                    temp = np.asarray(i[1].dropna().tolist())
                    hla_data.append(temp)

                hla_id = np.asarray(hla_id)
                hla_data = np.asarray(hla_data)

                keep,idx_1,idx_2 = np.intersect1d(file_list,hla_id,return_indices=True)
                file_list = keep
                hla_data = hla_data[idx_2]

                self.lb_hla = MultiLabelBinarizer()
                hla_data_num = self.lb_hla.fit_transform(hla_data)

                hla_data_seq_num = np.zeros(shape=[file_id.shape[0],hla_data_num.shape[1]])
                for file,h in zip(file_list,hla_data_num):
                    hla_data_seq_num[file_id==file] = h
                hla_data_seq_num = hla_data_seq_num.astype(int)
                hla_data_seq = np.asarray(self.lb_hla.inverse_transform(hla_data_seq_num))

                #remove sequences with no hla information
                idx_keep = np.sum(hla_data_seq_num,-1)>0
                X_Seq_alpha = X_Seq_alpha[idx_keep]
                X_Seq_beta = X_Seq_beta[idx_keep]
                Y = Y[idx_keep]
                alpha_sequences = alpha_sequences[idx_keep]
                beta_sequences = beta_sequences[idx_keep]
                label_id = label_id[idx_keep]
                file_id = file_id[idx_keep]
                freq = freq[idx_keep]
                counts = counts[idx_keep]
                seq_index = seq_index[idx_keep]
                v_beta = v_beta[idx_keep]
                d_beta = d_beta[idx_keep]
                j_beta = j_beta[idx_keep]
                v_alpha = v_alpha[idx_keep]
                j_alpha = j_alpha[idx_keep]
                v_beta_num = v_beta_num[idx_keep]
                d_beta_num = d_beta_num[idx_keep]
                j_beta_num = j_beta_num[idx_keep]
                v_alpha_num = v_alpha_num[idx_keep]
                j_alpha_num = j_alpha_num[idx_keep]
                hla_data_seq = hla_data_seq[idx_keep]
                hla_data_seq_num = hla_data_seq_num[idx_keep]

            else:
                self.lb_hla = MultiLabelBinarizer()
                file_list = np.asarray(file_list)
                hla_data = np.asarray(['None']*len(file_list))
                hla_data_num = np.asarray(['None']*len(file_list))
                hla_data_seq = np.asarray(['None']*len(file_id))
                hla_data_seq_num = np.asarray(['None']*len(file_id))

            with open(os.path.join(self.Name,'Data.pkl'), 'wb') as f:
                pickle.dump([X_Seq_alpha,X_Seq_beta,Y, alpha_sequences,beta_sequences, label_id, file_id, freq,counts,seq_index,
                             self.lb,file_list,self.use_alpha,self.use_beta,
                             self.lb_v_beta, self.lb_d_beta, self.lb_j_beta,self.lb_v_alpha,self.lb_j_alpha,
                             v_beta, d_beta,j_beta,v_alpha,j_alpha,
                             v_beta_num, d_beta_num, j_beta_num,v_alpha_num,j_alpha_num,
                             self.use_v_beta,self.use_d_beta,self.use_j_beta,self.use_v_alpha,self.use_j_alpha,
                             self.lb_hla, hla_data, hla_data_num,hla_data_seq,hla_data_seq_num,
                             self.use_hla,self.use_hla_sup,self.keep_non_supertype_alleles],f,protocol=4)

        else:
            with open(os.path.join(self.Name,'Data.pkl'), 'rb') as f:
                X_Seq_alpha,X_Seq_beta,Y, alpha_sequences,beta_sequences, label_id, file_id, freq,counts,seq_index,\
                self.lb,file_list,self.use_alpha,self.use_beta,\
                    self.lb_v_beta, self.lb_d_beta, self.lb_j_beta,self.lb_v_alpha,self.lb_j_alpha,\
                    v_beta, d_beta,j_beta,v_alpha,j_alpha,\
                    v_beta_num, d_beta_num, j_beta_num,v_alpha_num,j_alpha_num,\
                    self.use_v_beta,self.use_d_beta,self.use_j_beta,self.use_v_alpha,self.use_j_alpha,\
                    self.lb_hla, hla_data,hla_data_num,hla_data_seq,hla_data_seq_num,\
                self.use_hla,self.use_hla_sup,self.keep_non_supertype_alleles = pickle.load(f)

        self.X_Seq_alpha = X_Seq_alpha
        self.X_Seq_beta = X_Seq_beta
        self.Y = Y
        self.alpha_sequences = alpha_sequences
        self.beta_sequences = beta_sequences
        self.class_id = label_id
        self.sample_id = file_id
        self.freq = freq
        self.counts = counts
        self.sample_list = file_list
        self.v_beta = v_beta
        self.v_beta_num = v_beta_num
        self.d_beta = d_beta
        self.d_beta_num = d_beta_num
        self.j_beta = j_beta
        self.j_beta_num = j_beta_num
        self.v_alpha = v_alpha
        self.v_alpha_num = v_alpha_num
        self.j_alpha = j_alpha
        self.j_alpha_num = j_alpha_num
        self.seq_index = np.asarray(list(range(len(self.Y))))
        self.predicted = np.zeros((len(self.Y),len(self.lb.classes_)))
        self.hla_data_seq = hla_data_seq
        self.hla_data_seq_num = hla_data_seq_num
        self.w = np.ones(len(self.seq_index))
        #self.seq_index_j = seq_index
        print('Data Loaded')


    def Sequence_Inference(self, alpha_sequences=None, beta_sequences=None, v_beta=None, d_beta=None, j_beta=None,
                  v_alpha=None, j_alpha=None, p=None,hla=None, batch_size=10000,models=None,return_dist=False):
        """
        # Predicting outputs of sequence models on new data

        This method allows a user to take a pre-trained autoencoder/sequence classifier and generate outputs from the model on new data. For the autoencoder, this returns the features from the latent space. For the sequence classifier, it is the probability of belonging to each class.

        In the case that multiple models have been trained via MC or K-fold Cross-Validation strategy for the sequence classifier, this method can use some or all trained models in an ensemble fashion to provide the average prediction per sequence as well as the distribution of predictions from all trained models.

        """
        model_type,get = load_model_data(self)
        out, out_dist = inference_method_ss(get,alpha_sequences,beta_sequences,
                               v_beta,d_beta,j_beta,v_alpha,j_alpha,hla,
                                p,batch_size,self,models)

        if return_dist:
            return out, out_dist
        else:
            return out


   


class DeepTCR_S_base(DeepTCR_base,feature_analytics_class,vis_class):
    def AUC_Curve(self,by=None,filename='AUC.tif',title=None,title_font=None,plot=True,diag_line=True,
                  xtick_size = None, ytick_size=None, xlabel_size = None, ylabel_size=None,
                  legend_font_size=None,frameon=True,legend_loc = 'lower right',
                  figsize=None,set='test',color_dict=None):
        """
        # AUC Curve for both Sequence and Repertoire/Sample Classifiers

        Args:

            by (str): To show AUC curve for only one class, set this parameter to the name of the class label one wants to plot.

            filename (str): Filename to save tif file of AUC curve.

            title (str): Optional Title to put on ROC Curve.

            title_font (int): Optional font size for title

            plot (bool): To suppress plotting and just save the data/figure, set to False.

            diag_line (bool): To plot the line/diagonal of y=x defining no predictive power, set to True. To remove from plot, set to False.

            xtick_size (float): Size of xticks

            ytick_size (float): Size of yticks

            xlabel_size (float): Size of xlabel

            ylabel_size (float): Size of ylabel

            legend_font_size (float): Size of legend

            frameon (bool): Whether to show frame around legend.

            figsize (tuple): To change the default size of the figure, set this to size of figure (i.e. - (10,10) )

            set (str): Which partition of the data to look at performance of model. Options are train/valid/test.

            color_dict (dict): An optional dictionary that maps classes to colors in the case user wants to define colors of lines on plot.

        Returns:
            AUC Data

            - self.AUC_DF (Pandas Dataframe):
            AUC scores are returned for each class.

            In addition to plotting the ROC Curve, the AUC's are saved to a csv file in the results directory called 'AUC.csv'

        """
        try:
            y_test = self.test_pred.__dict__[set].y_test
            y_pred = self.test_pred.__dict__[set].y_pred
        except:
            y_test = self.y_test
            y_pred = self.y_pred

        auc_scores = []
        classes = []
        if plot is False:
            plt.ioff()
        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            plt.figure()

        if diag_line:
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')


        if color_dict is None:
            RGB_tuples = distinctipy.get_colors(len(self.lb.classes_),rng=0)
            color_dict = dict(zip(self.lb.classes_, RGB_tuples))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        if by is None:
            for ii, class_name in enumerate(self.lb.classes_, 0):
                try:
                    roc_score = roc_auc_score(y_test[:, ii], y_pred[:,ii])
                    classes.append(class_name)
                    auc_scores.append(roc_score)
                    fpr, tpr, _ = roc_curve(y_test[:, ii], y_pred[:,ii])
                    plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (class_name, roc_score),c=color_dict[class_name])
                except:
                    continue
        else:
            class_name = by
            ii = self.lb.transform([by])[0]
            roc_score = roc_auc_score(y_test[:, ii], y_pred[:, ii])
            auc_scores.append(roc_score)
            classes.append(class_name)
            fpr, tpr, _ = roc_curve(y_test[:, ii], y_pred[:, ii])
            plt.plot(fpr, tpr, lw=2, label='%s (area = %0.4f)' % (class_name, roc_score),c=color_dict[class_name])

        plt.legend(loc=legend_loc,frameon=frameon)
        if legend_font_size is not None:
            plt.legend(prop={'size': legend_font_size},loc=legend_loc,frameon=frameon)

        if title is not None:
            if title_font is not None:
                plt.title(title,fontsize=title_font)
            else:
                plt.title(title)

        ax = plt.gca()

        if xlabel_size is not None:
            ax.xaxis.label.set_size(xlabel_size)

        if ylabel_size is not None:
            ax.yaxis.label.set_size(ylabel_size)

        if xtick_size is not None:
            plt.xticks(fontsize=xtick_size)

        if ytick_size is not None:
            plt.yticks(fontsize=ytick_size)

        plt.tight_layout()
        plt.savefig(os.path.join(self.directory_results,filename))
        if plot is True:
            plt.show(block=False)
        else:
            plt.close()

        df_out = pd.DataFrame()
        df_out['Class'] = classes
        df_out['AUC'] = auc_scores
        df_out.to_csv(os.path.join(self.directory_results,'AUC.csv'),index=False)
        self.AUC_DF = df_out

    
    def _residue(self, alpha_sequence, beta_sequence, v_beta, d_beta, j_beta, v_alpha, j_alpha, hla,
                 p, batch_size, models, chain):

        if self.model_type == 'SS':
            inf_func = self.Sequence_Inference
        elif self.model_type == 'WF':
            inf_func = self.Sample_Inference

        df_alpha = pd.DataFrame()
        df_beta = pd.DataFrame()
        if chain == 'alpha':
            if alpha_sequence is not None:
                alpha_list, pos, ref, alt = make_seq_list(alpha_sequence, ref=list(self.aa_idx.keys()))
                len_list = len(alpha_list)

                if beta_sequence is None:
                    beta_sequences = None
                else:
                    beta_sequences = np.array([beta_sequence] * len_list)

                if v_beta is None:
                    v_beta = None
                else:
                    v_beta = np.array([v_beta] * len_list)

                if d_beta is None:
                    d_beta = None
                else:
                    d_beta = np.array([d_beta] * len_list)

                if j_beta is None:
                    j_beta = None
                else:
                    j_beta = np.array([j_beta] * len_list)

                if v_alpha is None:
                    v_alpha = None
                else:
                    v_alpha = np.array([v_alpha] * len_list)

                if j_alpha is None:
                    j_alpha = None
                else:
                    j_alpha = np.array([j_alpha] * len_list)

                if hla is None:
                    hla = None
                else:
                    hla = np.array([hla] * len_list)

                out = inf_func(beta_sequences=beta_sequences,
                               alpha_sequences=np.array(alpha_list),
                               v_beta=v_beta,
                               d_beta=d_beta,
                               j_beta=j_beta,
                               v_alpha=v_alpha,
                               j_alpha=j_alpha,
                               p=p,
                               hla=hla,
                               batch_size=batch_size,
                               models=models)

                df_alpha['alpha'] = alpha_list
                df_alpha['pos'] = pos
                df_alpha['ref'] = ref
                df_alpha['alt'] = alt
                if self.regression:
                    df_alpha['high'] = out[:, 0]
                else:
                    for ii in range(out.shape[1]):
                        df_alpha[self.lb.inverse_transform([ii])[0]] = out[:, ii]

        if chain == 'beta':
            if beta_sequence is not None:
                beta_list, pos, ref, alt = make_seq_list(beta_sequence, ref=list(self.aa_idx.keys()))
                len_list = len(beta_list)
                if alpha_sequence is None:
                    alpha_sequences = None
                else:
                    alpha_sequences = np.array([alpha_sequence] * len_list)

                if v_beta is None:
                    v_beta = None
                else:
                    v_beta = np.array([v_beta] * len_list)

                if d_beta is None:
                    d_beta = None
                else:
                    d_beta = np.array([d_beta] * len_list)

                if j_beta is None:
                    j_beta = None
                else:
                    j_beta = np.array([j_beta] * len_list)

                if v_alpha is None:
                    v_alpha = None
                else:
                    v_alpha = np.array([v_alpha] * len_list)

                if j_alpha is None:
                    j_alpha = None
                else:
                    j_alpha = np.array([j_alpha] * len_list)

                if hla is None:
                    hla = None
                else:
                    hla = np.array([hla] * len_list)

                out = inf_func(beta_sequences=np.array(beta_list),
                               alpha_sequences=alpha_sequences,
                               v_beta=v_beta,
                               d_beta=d_beta,
                               j_beta=j_beta,
                               v_alpha=v_alpha,
                               j_alpha=j_alpha,
                               p=p,
                               hla=hla,
                               batch_size=batch_size,
                               models=models)

                df_beta['beta'] = beta_list
                df_beta['pos'] = pos
                df_beta['ref'] = ref
                df_beta['alt'] = alt
                if self.regression:
                    df_beta['high'] = out[:, 0]
                else:
                    for ii in range(out.shape[1]):
                        df_beta[self.lb.inverse_transform([ii])[0]] = out[:, ii]

        if chain == 'alpha':
            return df_alpha
        elif chain == 'beta':
            return df_beta

    
class DeepTCR_SS(DeepTCR_S_base):
    def Get_Train_Valid_Test(self,test_size=0.25,LOO=None,split_by_sample=False,combine_train_valid=False):
        """
        # Train/Valid/Test Splits.

        Divide data for train, valid, test set. Training is used to train model parameters, validation is used to set early stopping, and test acts as blackbox independent test set.

        Args:

            test_size (float): Fraction of sample to be used for valid and test set.

            LOO (int): Number of sequences to leave-out in Leave-One-Out Cross-Validation. For example, when set to 20, 20 sequences will be left out for the validation set and 20 samples will be left out for the test set.

            split_by_sample (int): In the case one wants to train the single sequence classifer but not to mix the train/test sets with sequences from different samples, one can set this parameter to True to do the train/test splits by sample.

            combine_train_valid (bool): To combine the training and validation partitions into one which will be used for training and updating the model parameters, set this to True. This will also set the validation partition to the test partition. In other words, new train set becomes (original train + original valid) and then new valid = original test partition, new test = original test partition. Therefore, if setting this parameter to True, change one of the training parameters to set the stop training criterion (i.e. train_loss_min) to stop training based on the train set. If one does not chanage the stop training criterion, the decision of when to stop training will be based on the test data (which is considered a form of over-fitting).

        """
        Vars = [self.X_Seq_alpha,self.X_Seq_beta,self.alpha_sequences,self.beta_sequences,self.sample_id,self.class_id,self.seq_index,
                self.v_beta_num,self.d_beta_num,self.j_beta_num,self.v_alpha_num,self.j_alpha_num,
                self.v_beta,self.d_beta,self.j_beta,self.v_alpha,self.j_alpha,self.hla_data_seq_num]

        var_names = ['X_Seq_alpha','X_Seq_beta','alpha_sequences','beta_sequences','sample_id','class_id','seq_index',
                     'v_beta_num','d_beta_num','j_beta_num','v_alpha_num','j_alpha_num','v_beta','d_beta','j_beta',
                     'v_alpha','j_alpha','hla_data_seq_num']

        self.var_dict = dict(zip(var_names,list(range(len(var_names)))))

        if split_by_sample is False:
            self.train,self.valid,self.test = Get_Train_Valid_Test(Vars=Vars,Y=self.Y,test_size=test_size,regression=self.regression,LOO=LOO)

        # else:
        #     sample = np.unique(self.sample_id)
        #     Y = np.asarray([self.Y[np.where(self.sample_id == x)[0][0]] for x in sample])
        #     train, valid, test = Get_Train_Valid_Test([sample], Y, test_size=test_size,regression=self.regression,LOO=LOO)

        #     self.train_idx = np.where(np.isin(self.sample_id, train[0]))[0]
        #     self.valid_idx = np.where(np.isin(self.sample_id, valid[0]))[0]
        #     self.test_idx = np.where(np.isin(self.sample_id, test[0]))[0]

        #     Vars.append(self.Y)

        #     self.train = [x[self.train_idx] for x in Vars]
        #     self.valid = [x[self.valid_idx] for x in Vars]
        #     self.test = [x[self.test_idx] for x in Vars]

        if combine_train_valid:
            for i in range(len(self.train)):
                self.train[i] = np.concatenate((self.train[i],self.valid[i]),axis=0)
                self.valid[i] = self.test[i]

        if (self.valid[0].size == 0) or (self.test[0].size == 0):
            raise Exception('Choose different train/valid/test parameters!')


    def _reset_models(self):
        self.models_dir = os.path.join(self.Name,'models')
        if os.path.exists(self.models_dir):
            shutil.rmtree(self.models_dir)
        os.makedirs(self.models_dir)

    def _build(self,kernel = 5,trainable_embedding = True,embedding_dim_aa = 64, embedding_dim_genes = 48, embedding_dim_hla = 12,
               num_fc_layers = 0, units_fc = 12,weight_by_class = False, class_weights = None,
               use_only_seq = False, use_only_gene = False, use_only_hla = False, size_of_net = 'medium',graph_seed = None,
               drop_out_rate=0.0,multisample_dropout = False, multisample_dropout_rate = 0.50, multisample_dropout_num_masks = 64,
               batch_size = 1000, epochs_min = 10, stop_criterion = 0.001, stop_criterion_window = 10,
               accuracy_min = None, train_loss_min = None, hinge_loss_t = 0.0, convergence = 'validation', learning_rate = 0.001, suppress_output = False):


        graph_model = tf.Graph()
        GO = graph_object()
        GO.on_graph_clustering=False
        GO.size_of_net = size_of_net
        GO.embedding_dim_genes = embedding_dim_genes
        GO.embedding_dim_aa = embedding_dim_aa
        GO.embedding_dim_hla = embedding_dim_hla
        GO.l2_reg = 0.0
        train_params = graph_object()
        train_params.batch_size = batch_size
        train_params.epochs_min = epochs_min
        train_params.stop_criterion = stop_criterion
        train_params.stop_criterion_window  = stop_criterion_window
        train_params.accuracy_min = accuracy_min
        train_params.train_loss_min = train_loss_min
        train_params.convergence = convergence
        train_params.suppress_output = suppress_output
        train_params.drop_out_rate = drop_out_rate
        train_params.multisample_dropout_rate = multisample_dropout_rate

        with graph_model.device(self.device):
            with graph_model.as_default():
                if graph_seed is not None:
                    tf.compat.v1.set_random_seed(graph_seed)

                GO.net = 'sup'
                GO.Features = Conv_Model(GO,self,trainable_embedding,kernel,use_only_seq,use_only_gene,use_only_hla,
                                         num_fc_layers,units_fc)
                if self.regression is False:
                    GO.Y = tf.compat.v1.placeholder(tf.float64, shape=[None, self.Y.shape[1]])
                else:
                    GO.Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

                if self.regression is False:
                    if multisample_dropout:
                        GO.logits = MultiSample_Dropout(GO.Features,
                                                        num_masks=multisample_dropout_num_masks,
                                                        units=self.Y.shape[1],
                                                        activation=None,
                                                        rate=GO.prob_multisample)
                    else:
                        GO.logits = tf.compat.v1.layers.dense(GO.Features, self.Y.shape[1])

                    per_sample_loss = tf.nn.softmax_cross_entropy_with_logits(labels=GO.Y, logits=GO.logits)
                    per_sample_loss = per_sample_loss - hinge_loss_t
                    per_sample_loss = tf.cast((per_sample_loss > 0), tf.float32) * per_sample_loss
                    if weight_by_class is True:
                        class_weights = tf.constant([(1 / (np.sum(self.Y, 0) / np.sum(self.Y))).tolist()])
                        weights = tf.squeeze(tf.matmul(tf.cast(GO.Y, dtype='float32'), class_weights, transpose_b=True),axis=1)
                        GO.loss = tf.reduce_mean(input_tensor=weights * per_sample_loss)
                    elif class_weights is not None:
                        weights = np.zeros([1, len(self.lb.classes_)]).astype(np.float32)
                        for key in class_weights:
                            weights[:, self.lb.transform([key])[0]] = class_weights[key]
                        class_weights = tf.constant(weights)
                        weights = tf.squeeze(tf.matmul(tf.cast(GO.Y, dtype='float32'), class_weights, transpose_b=True),axis=1)
                        GO.loss = tf.reduce_mean(input_tensor=weights * per_sample_loss)
                    else:
                        GO.loss = tf.reduce_mean(input_tensor=per_sample_loss)

                else:
                    if multisample_dropout:
                        GO.logits = MultiSample_Dropout(GO.Features,
                                                        num_masks=multisample_dropout_num_masks,
                                                        units=1,
                                                        activation=None,
                                                        rate=GO.prob_multisample)
                    else:
                        GO.logits = tf.compat.v1.layers.dense(GO.Features, 1)

                    GO.loss = tf.reduce_mean(input_tensor=tf.square(GO.Y-GO.logits))

                GO.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(GO.loss)

                if self.regression is False:
                    with tf.compat.v1.name_scope('Accuracy_Measurements'):
                        GO.predicted = tf.nn.softmax(GO.logits, name='predicted')
                        correct_pred = tf.equal(tf.argmax(input=GO.predicted, axis=1), tf.argmax(input=GO.Y, axis=1))
                        GO.accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_pred, tf.float32), name='accuracy')
                else:
                    GO.predicted = GO.logits
                    GO.accuracy = GO.loss

                GO.saver = tf.compat.v1.train.Saver(max_to_keep=None)

                self.GO = GO
                self.train_params = train_params
                self.graph_model = graph_model
                self.kernel = kernel

    def _train(self,batch_seed=None,iteration=0):

        GO = self.GO
        graph_model = self.graph_model
        train_params = self.train_params

        batch_size = train_params.batch_size
        epochs_min = train_params.epochs_min
        stop_criterion = train_params.stop_criterion
        stop_criterion_window = train_params.stop_criterion_window
        accuracy_min = train_params.accuracy_min
        train_loss_min = train_params.train_loss_min
        convergence = train_params.convergence
        suppress_output = train_params.suppress_output
        drop_out_rate = train_params.drop_out_rate
        multisample_dropout_rate = train_params.multisample_dropout_rate


        #Initialize Training
        tf.compat.v1.reset_default_graph()
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(graph=graph_model,config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            val_loss_total = []
            train_accuracy_total = []
            train_loss_total = []
            stop_check_list = []
            e = 0

            print('')
            while True:
                if batch_seed is not None:
                    np.random.seed(batch_seed)
                train_loss, train_accuracy, train_predicted,train_auc = \
                            Run_Graph_SS(self.train,sess,self,GO,batch_size,random=True,train=True,drop_out_rate=drop_out_rate,multisample_dropout_rate=multisample_dropout_rate)

                train_accuracy_total.append(train_accuracy)
                train_loss_total.append(train_loss)

                valid_loss, valid_accuracy, valid_predicted,valid_auc = \
                    Run_Graph_SS(self.valid,sess,self,GO,batch_size,random=False,train=False)

                val_loss_total.append(valid_loss)

                test_loss, test_accuracy, test_predicted,test_auc = \
                    Run_Graph_SS(self.test,sess,self,GO,batch_size,random=False,train=False)
                self.y_pred = test_predicted
                self.y_test = self.test[-1]


                if suppress_output is False:
                    print("Training_Statistics: \n",
                          "Epoch: {}".format(e + 1),
                          "Training loss: {:.5f}".format(train_loss),
                          "Validation loss: {:.5f}".format(valid_loss),
                          "Testing loss: {:.5f}".format(test_loss),
                          "Training Accuracy: {:.5}".format(train_accuracy),
                          "Validation Accuracy: {:.5}".format(valid_accuracy),
                          "Testing AUC: {:.5}".format(test_auc))

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    if e > epochs_min:
                        if accuracy_min is not None:
                            if np.mean(train_accuracy_total[-3:]) >= accuracy_min:
                                break
                        elif train_loss_min is not None:
                            if np.mean(train_loss_total[-3:]) < train_loss_min:
                                break
                        elif convergence == 'validation':
                            if val_loss_total:
                                stop_check_list.append(stop_check(val_loss_total, stop_criterion, stop_criterion_window))
                                if np.sum(stop_check_list[-3:]) >= 3:
                                    break

                        elif convergence == 'training':
                            if train_loss_total:
                                stop_check_list.append(stop_check(train_loss_total, stop_criterion, stop_criterion_window))
                                if np.sum(stop_check_list[-3:]) >= 3:
                                    break

                e += 1

            train_loss, train_accuracy, train_predicted, train_auc = \
                Run_Graph_SS(self.train, sess, self, GO, batch_size, random=False, train=False)

            self.test_pred.train.y_test.append(self.train[-1])
            self.test_pred.train.y_pred.append(train_predicted)

            valid_loss, valid_accuracy, valid_predicted, valid_auc = \
                Run_Graph_SS(self.valid, sess, self, GO, batch_size, random=False, train=False)

            self.test_pred.valid.y_test.append(self.valid[-1])
            self.test_pred.valid.y_pred.append(valid_predicted)

            test_loss, test_accuracy, test_predicted, test_auc = \
                Run_Graph_SS(self.test, sess, self, GO, batch_size, random=False, train=False)

            self.test_pred.test.y_test.append(self.test[-1])
            self.test_pred.test.y_pred.append(test_predicted)

            Get_Seq_Features_Indices(self,batch_size,GO,sess)
            self.features = Get_Latent_Features(self,batch_size,GO,sess)

            idx_base = np.asarray(range(len(self.sample_id)))
            self.train_idx = np.isin(idx_base,self.train[self.var_dict['seq_index']])
            self.valid_idx = np.isin(idx_base,self.valid[self.var_dict['seq_index']])
            self.test_idx = np.isin(idx_base,self.test[self.var_dict['seq_index']])

            if hasattr(self,'predicted'):
                self.predicted[self.test[self.var_dict['seq_index']]] += self.y_pred

            #
            if self.use_alpha is True:
                var_save = [self.alpha_features,self.alpha_indices,self.alpha_sequences]
                with open(os.path.join(self.Name, 'alpha_features.pkl'), 'wb') as f:
                    pickle.dump(var_save, f)

            if self.use_beta is True:
                var_save = [self.beta_features,self.beta_indices,self.beta_sequences]
                with open(os.path.join(self.Name, 'beta_features.pkl'), 'wb') as f:
                    pickle.dump(var_save, f)

            with open(os.path.join(self.Name, 'kernel.pkl'), 'wb') as f:
                pickle.dump(self.kernel, f)

            print('Done Training')
            # save model data and information for inference engine
            save_model_data(self, GO.saver, sess, name='SS', get=GO.predicted, iteration=iteration)

    def Train(self,kernel = 5,trainable_embedding = True,embedding_dim_aa = 64, embedding_dim_genes = 48, embedding_dim_hla = 12,
               num_fc_layers = 0, units_fc = 12,weight_by_class = False, class_weights = None,
               use_only_seq = False, use_only_gene = False, use_only_hla = False, size_of_net = 'medium',graph_seed = None,
               drop_out_rate=0.0,multisample_dropout = False, multisample_dropout_rate = 0.50, multisample_dropout_num_masks = 64,
               batch_size = 1000, epochs_min = 10, stop_criterion = 0.001, stop_criterion_window = 10,
               accuracy_min = None, train_loss_min = None, hinge_loss_t = 0.0, convergence = 'validation', learning_rate = 0.001, suppress_output = False,
                batch_seed = None):
        """
        # Train Single-Sequence Classifier

        This method trains the network and saves features values at the end of training for downstream analysis.

        The method also saves the per sequence predictions at the end of training in the variable self.predicted

        The multiesample parameters are used to implement Multi-Sample Dropout at the final layer of the model as described in "Multi-Sample Dropout for Accelerated Training and Better Generalization" https://arxiv.org/abs/1905.09788. This method has been shown to improve generalization of deep neural networks as well as improve convergence.

        Args:

            kernel (int): Size of convolutional kernel for first layer of convolutions.

            trainable_embedding (bool): Toggle to control whether a trainable embedding layer is used or native one-hot representation for convolutional layers.

            embedding_dim_aa (int): Learned latent dimensionality of amino-acids.

            embedding_dim_genes (int): Learned latent dimensionality of VDJ genes

            embedding_dim_hla (int): Learned latent dimensionality of HLA

        
            size_of_net (list or str): The convolutional layers of this network have 3 layers for which the use can modify the number of neurons per layer. The user can either specify the size of the network with the following options:

                - small == [12,32,64] neurons for the 3 respective layers
                - medium == [32,64,128] neurons for the 3 respective layers
                - large == [64,128,256] neurons for the 3 respective layers
                - custom, where the user supplies a list with the number of nuerons for the respective layers
                    i.e. [3,3,3] would have 3 neurons for all 3 layers.
                    One can also adjust the number of layers for the convolutional stack by changing the length of
                    this list. [3,3,3] = 3 layers, [3,3,3,3] = 4 layers.

        

        """
        self._reset_models()
        self.test_pred = make_test_pred_object()
        self._build(kernel,trainable_embedding,embedding_dim_aa, embedding_dim_genes, embedding_dim_hla,
               num_fc_layers, units_fc,weight_by_class, class_weights,
               use_only_seq, use_only_gene, use_only_hla, size_of_net,graph_seed,
               drop_out_rate,multisample_dropout, multisample_dropout_rate, multisample_dropout_num_masks,
               batch_size, epochs_min, stop_criterion, stop_criterion_window,
               accuracy_min, train_loss_min, hinge_loss_t, convergence, learning_rate, suppress_output)
        self._train(batch_seed=batch_seed,iteration=0)

        # 保存训练好的模型
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        model_file = os.path.join(save_model_path, "trained_model.h5")
        self.model.save(model_file)
        print(f"Model saved at: {model_file}")
    
        for set in ['train', 'valid', 'test']:
            self.test_pred.__dict__[set].y_test = np.vstack(self.test_pred.__dict__[set].y_test)
            self.test_pred.__dict__[set].y_pred = np.vstack(self.test_pred.__dict__[set].y_pred)

 
