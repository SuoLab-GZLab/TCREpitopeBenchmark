import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
from matplotlib import ticker
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve as pccurve
from sklearn.metrics import average_precision_score as AUPRC
from scipy.stats import pearsonr as pr

def kl_divergence(dis1,dis2,avoid_zeros=True):
    assert np.abs(np.sum(dis1)-1) <=0.01, 'you need to normalize' 
    if avoid_zeros:
        dis2 = [d + 1e-20 for d in dis2] #to avoid zero division
        dis1 = [d + 1e-20 for d in dis1]
    return sum([dis1[i]*np.log(dis1[i]/dis2[i]) for i in range(len(dis1))])

def js_divergence(p11,p12,p21,p22):
    if np.abs(np.sum(p11) - 1)>0.01:
        p11,p12,p21,p22 = p11/np.sum(p11),p12/np.sum(p12),p21/np.sum(p21),p22/np.sum(p22)
    assert len(p11) == len(p21)
    assert len(p12) == len(p22)
    #p11,p12,p21,p22 = np.array(p11),np.array(p12),np.array(p21),np.array(p22)
    return 0.5*kl_divergence(p11,0.5*(p11+p21)) + 0.5*kl_divergence(p22,0.5*(p22+p12))

class plotting:
    def __init__(self):
        self.aas = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa2idx = {self.aas[i]:i for i in range(len(self.aas))}
        self.idx2aa = {v: k for k, v in self.aa2idx.items()}
        self.vs = ['TRBV10-1','TRBV10-2','TRBV10-3','TRBV11-1','TRBV11-2','TRBV11-3','TRBV12-5', 'TRBV13', 'TRBV14', 'TRBV15', 
                'TRBV16', 'TRBV18','TRBV19','TRBV2', 'TRBV20-1', 'TRBV25-1', 'TRBV27', 'TRBV28', 'TRBV29-1', 'TRBV3-1', 'TRBV30',
        'TRBV4-1', 'TRBV4-2','TRBV4-3',
        'TRBV5-1', 'TRBV5-4', 'TRBV5-5', 'TRBV5-6', 'TRBV5-8','TRBV6-1', 'TRBV6-4', 'TRBV6-5', 'TRBV6-6','TRBV6-8', 'TRBV6-9', 'TRBV7-2', 'TRBV7-3',
        'TRBV7-4', 'TRBV7-6', 'TRBV7-7', 'TRBV7-8', 'TRBV7-9', 'TRBV9']
        self.js = ['TRBJ1-1', 'TRBJ1-2', 'TRBJ1-3','TRBJ1-4', 'TRBJ1-5', 'TRBJ1-6','TRBJ2-1', 'TRBJ2-2', 'TRBJ2-3', 'TRBJ2-4', 'TRBJ2-6', 'TRBJ2-7']


    def valid_seq(self,seq):
        if type(seq) is not str:
            return False
        if len(seq) <= 2:
            return False
        if seq[-1] != 'F' and seq[-2:] != 'YV':

            return False
        if seq[0] != 'C':

            return False
        if 's' in seq or 'e' in seq:
            return False
        return True

    def length_dis(self,data,tcrpeg,sonnia,tcrvae,lengths,fig_name=None):
        '''
        each data input should be in a format (mean_freq,err)
        mean_freq is mapped to lengths
        '''
        plt.figure(figsize=(6,4))
        plt.plot(lengths,data,'b-',linewidth=2,label='Data')
        plt.plot(lengths,tcrpeg[0],'r--',linewidth=1,label='TCRpeg',alpha=0.8)
        plt.fill_between(lengths, tcrpeg[0]-tcrpeg[1], tcrpeg[0]+tcrpeg[1],alpha=0.4)
        plt.plot(lengths,sonnia[0],'g--',linewidth=1,label='soNNia',alpha=0.8)
        #plt.errorbar(lengths, sonnia[0], sonnia[1], fmt='o-', linewidth=1, capsize=5,label='soNNia',markersize=5)
        plt.fill_between(lengths, sonnia[0]-sonnia[1], sonnia[0]+sonnia[1],alpha=0.4)
        plt.plot(lengths,tcrvae[0],'y--',linewidth=1,label='TCRvae',alpha=1)
        #plt.errorbar(lengths, tcrvae[0], tcrvae[1], fmt='o-', linewidth=1, capsize=5,label='TCRvae',markersize=5)
        plt.fill_between(lengths, tcrvae[0]-tcrvae[1], tcrvae[0]+tcrvae[1],alpha=0.4)

        # plt.plot(np.arange(l_length),self.sonia_model.model_marginals[:l_length],label='POST marginals',alpha=0.9)
        # plt.plot(np.arange(l_length),self.sonia_model.data_marginals[:l_length],label='DATA marginals',alpha=0.9)
        # plt.plot(np.arange(l_length),self.sonia_model.gen_marginals[:l_length],label='GEN marginals',alpha=0.9)
        #kl1,kl2,kl3 = kl_divergence(data,tcrpeg[0]),kl_divergence(data,sonnia[0]),kl_divergence(data,tcrvae[0])
        # plt.xticks(rotation='vertical')
        plt.grid()
        plt.legend()
        #plt.title('Length Distribution')
        plt.xlabel('CDR3 length',size=12)
        plt.ylabel('Observed frequency',size=12)
        plt.tight_layout()
        # plt.title('CDR3 LENGTH DISTRIBUTION',fontsize=20)
        if fig_name is not None:
            plt.savefig('results/pictures/'+fig_name + '.jpg',dpi=200)
        else: plt.show()
        #return kl1,kl2,kl3
    def aas_dis(self,data,tcrpeg,sonnia,tcrvae,fig_name=None):
        '''
        inputs should be a dict with len(keys)=20
        '''
        fig,axs = plt.subplots(4,5,figsize=(30,20))
        x_axis = list(range(1,31))
        for row in range(4):
            for col in range(5):
                index = row*5 + col
                d,d1,d2,d3,aa = data[index],tcrpeg[index],sonnia[index],tcrvae[index],self.idx2aa[index]
                y,y1,y2,y3 = [d[k] for k in x_axis],[d1[k][0] for k in x_axis],[d2[k][0] for k in x_axis],[d3[k][0] for k in x_axis]
                e,e1,e2,e3 = [d[k] for k in x_axis],[d1[k][1] for k in x_axis],[d2[k][1] for k in x_axis],[d3[k][1] for k in x_axis]
                # sum1,sum2 = sum(y1),sum(y2)
                # if sum1 == 0 or sum2 == 0:
                #     print(sum1)
                #     print(sum2)
                #     print(aa)
                # y1,y2 = [k/sum1 for k in y1],[k/sum2 for k in y2]
                # kl = kl_divergence(y1,y2)
                #axs[row,col].plot(x_axis,y1,x_axis,y2)
                y1,y2,y3 = np.array(y1),np.array(y2),np.array(y3)
                e1,e2,e3 = np.array(e1),np.array(e2),np.array(e3)
                axs[row,col].plot(x_axis,y,'b-',linewidth=3,label='Data',alpha=1)
                axs[row,col].plot(x_axis, y1,'r--' ,  linewidth=2,label='TCRpeg',alpha=0.8)
                axs[row,col].plot(x_axis, y2,'g--', linewidth=2,label='soNNia',alpha=0.8)
                axs[row,col].plot(x_axis, y3, 'y--', linewidth=2,label='TCRvae',alpha=0.8)
                axs[row,col].fill_between(x_axis, y1-e1, y1+e1, color = 'r',alpha=0.4)
                axs[row,col].fill_between(x_axis, y2-e2, y2+e2,color='g' ,alpha=0.4)
                axs[row,col].fill_between(x_axis, y3-e2, y3+e2, color = 'y', alpha=0.4)
                #axs[row,col].errorbar(x_axis, y3, e3, fmt='o-', linewidth=1, capsize=2,label='TCRvae',markersize=2)
                #axs[row,col].set_title('Amino acid: ' +str(aa))
                axs[row,col].tick_params(axis="x", labelsize=15) 
                axs[row,col].tick_params(axis="y", labelsize=15) 
                axs[row,col].legend(fontsize=15)
                axs[row,col].text(0.02, 0.95, aa, ha='center', va='center',transform = axs[row,col].transAxes,size=20,color='red')
                # axs[row,col].set_xlabel('Position',size=8)
                # axs[row,col].set_ylabel('Frequency',size=8)
                #use a common label for aa dis
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # fig.text(0.5, -0.01, 'common X', ha='center',size=20)
        # fig.text(0.02, 0.5, 'common Y', va='center', size=20,rotation='vertical')    
        fig.tight_layout()
        if fig_name is not None:
            plt.savefig('results/pictures/'+fig_name + '.jpg',dpi=200)
        else: plt.show()
        

    def pos_dis(self,seqs1:list,seqs2:list,fig_name:str):
        #now is for pos2-10
        pos2aa_dic1 = [defaultdict(int) for _ in range(9)]
        pos2aa_dic2 = [defaultdict(int) for _ in range(9)]

        for i in range(len(seqs1)):
            seq1 = seqs1[i]
            if len(seq1) < 10:
                print(seq1)
                continue
            for j in range(9): #j: 0-8
                pos2aa_dic1[j][seq1[j+1]] += 1
        for i in range(len(seqs2)):
            seq2 = seqs2[i]
            if len(seq2) < 10:
                print(seq2)
                continue
            for j in range(9): #j: 0-8
                pos2aa_dic2[j][seq2[j+1]] += 1
        fig,axs = plt.subplots(3,3,figsize=(16,16))
        x_axis = list('ACDEFGHIKLMNPQRSTVWY')
        for row in range(3):
            for col in range(3):
                index = row*3 + col
                d1,d2,pos = pos2aa_dic1[index],pos2aa_dic2[index],index+2
                y1,y2 = [d1[k] for k in x_axis],[d2[k] for k in x_axis]
                sum1,sum2 = sum(y1),sum(y2)
                y1,y2 = [k/sum1 for k in y1],[k/sum2 for k in y2]
                kl = kl_divergence(y1,y2)
                axs[row,col].plot(x_axis,y1,x_axis,y2)
                axs[row,col].set_title('Position: '+str(pos) +' kl: ' +str(kl))
                axs[row,col].legend(['seqs1','seqs2'])
                plt.savefig('results/pictures/'+fig_name + '.png')

    def aa_num_dis(self,seqs1:list,seqs2:list,fig_name:str):
        aas_1 = defaultdict(int)
        aas_2 = defaultdict(int)
        for i in range(len(seqs1)):
            seq1 = seqs1[i]
            if len(seq1) <= 2 or seq1[0] != 'C' or (seq1[-1] != 'F' and seq1[:-2] != 'YV'):
                continue
            for j,aa in enumerate(seq1):
                #if aa == 'e' or 
                aas_1[aa] += 1
        for i in range(len(seqs2)):
            seq2 = seqs2[i]
            for j,aa in enumerate(seq2):
                aas_2[aa] += 1
        x_axis = list(self.aas)
        y1,y2 = [aas_1[aa] for aa in x_axis],[aas_2[aa] for aa in x_axis]
        sum1,sum2 = sum(y1),sum(y2)
        
        y1,y2 = [k/sum1 for k in y1],[k/sum2 for k in y2]
        kl = kl_divergence(y1,y2)
        plt.figure()
        plt.plot(x_axis,y1,x_axis,y2)
        plt.legend(['fre1','fre2'])
        plt.title(str(kl))
        plt.savefig('results/pictures/'+fig_name + '.png')

    def v_dis(self,data,tcrpeg,sonnia,tcrvae,fig_name=None):
        '''
        each input should be in the format of (mean_freqs,errs)
        also should be one-to-one with the gene list
        '''   
        plt.figure(figsize=(12,4))
        order=np.argsort(data[0])[::-1]
        x_axis = np.array(list(range(len(self.vs))))
        plt.errorbar(x_axis-0.2, np.array(data[0])[order] , fmt='b+', linewidth=4, capsize=5,label='Data',markersize=8)
        plt.errorbar(x_axis-0.1, np.array(tcrpeg[0])[order],np.array(tcrpeg[1])[order] , fmt='ro', linewidth=1, capsize=3,label='TCRpeg',markersize=3)
        plt.errorbar(x_axis+0.1, np.array(sonnia[0])[order] ,np.array(sonnia[1])[order], fmt='go', linewidth=1, capsize=3,label='soNNia',markersize=3)
        plt.errorbar(x_axis+0.2, np.array(tcrvae[0])[order] ,np.array(tcrvae[1])[order], fmt='yo', linewidth=1, capsize=3,label='TCRvae',markersize=3)
        plt.xticks(x_axis,np.array(self.vs)[order])
        plt.xticks(rotation=45)
        plt.ylabel('Frequency',size=12)
        plt.xlabel('V Gene',size=12)
        plt.grid()
        plt.legend()
        #plt.title('J USAGE DISTRIBUTION',fontsize=20)
        plt.tight_layout()
        if fig_name is not None:
            plt.savefig('results/pictures/'+fig_name + '.jpg',dpi=200)
        else: plt.show()
        


    def j_dis(self,data,tcrpeg,sonnia,tcrvae,fig_name=None):
        plt.figure(figsize=(6,4))
        order=np.argsort(data[0])[::-1]
        x_axis = np.array(list(range(len(self.js))))
        plt.errorbar(x_axis-0.2, np.array(data[0])[order] , fmt='b+', linewidth=4, capsize=3,label='Data',markersize=10)
        plt.errorbar(x_axis-0.1, np.array(tcrpeg[0])[order],np.array(tcrpeg[1])[order] , fmt='ro', linewidth=1, capsize=3,label='TCRpeg',markersize=3)
        plt.errorbar(x_axis+0.1, np.array(sonnia[0])[order] ,np.array(sonnia[1])[order], fmt='go', linewidth=1, capsize=3,label='soNNia',markersize=3)
        plt.errorbar(x_axis+0.2, np.array(tcrvae[0])[order] ,np.array(tcrvae[1])[order], fmt='yo', linewidth=1, capsize=3,label='TCRvae',markersize=3)
        plt.xticks(x_axis,np.array(self.js)[order])
        plt.xticks(rotation=45)
        plt.ylabel('Frequency',size=12)
        plt.xlabel('J Gene',size=12)
        plt.grid()
        plt.legend()
        #plt.title('J USAGE DISTRIBUTION',fontsize=20)
        plt.tight_layout()
        if fig_name is not None:
            plt.savefig('results/pictures/'+fig_name + '.jpg',dpi=200)
        else: plt.show()

    def J_Dis(self,data,tcrpeg,fig_name=None):
        #for single input
        #process data
        
        plt.figure(figsize=(6,4))        
        data_, tcrpeg_ = [0]*len(self.js),[0] * len(self.js)
        gene2i = {self.js[i]:i for i in range(len(self.js))}
        for i in range(len(data)):
            data_[gene2i[data[i]]] += 1
        data = [x/len(data) for x in data_]
        for i in range(len(tcrpeg)):
            tcrpeg_[gene2i[tcrpeg[i]]] += 1
        tcrpeg = [x/len(tcrpeg) for x in tcrpeg_] #fres
        order=np.argsort(data)[::-1]
        x_axis = np.array(list(range(len(self.js))))          
        plt.errorbar(x_axis-0.2, np.array(data)[order] , fmt='b+', linewidth=4, capsize=3,label='Data',markersize=10)
        plt.errorbar(x_axis+0.2, np.array(tcrpeg)[order] , fmt='ro', linewidth=1, capsize=3,label='TCRpeg',markersize=3)        
        plt.xticks(x_axis,np.array(self.js)[order])
        plt.xticks(rotation=45)
        plt.ylabel('Frequency',size=12)
        plt.xlabel('J Gene',size=12)
        plt.grid()
        plt.legend()
        #plt.title('J USAGE DISTRIBUTION',fontsize=20)
        plt.tight_layout()
        if fig_name is not None:
            plt.savefig('results/pictures/'+fig_name + '.jpg',dpi=200)
        else: plt.show()

    def V_Dis(self,data,tcrpeg,fig_name=None):
        '''
        each input should be in the format of (mean_freqs,errs)
        also should be one-to-one with the gene list
        '''   
        plt.figure(figsize=(12,4))
        data_, tcrpeg_ = [0]*len(self.vs),[0] * len(self.vs)
        gene2i = {self.vs[i]:i for i in range(len(self.vs))}
        for i in range(len(data)):
            data_[gene2i[data[i]]] += 1
        data = [x/len(data) for x in data_]
        for i in range(len(tcrpeg)):
            tcrpeg_[gene2i[tcrpeg[i]]] += 1
        tcrpeg = [x/len(tcrpeg) for x in tcrpeg_]
        order=np.argsort(data)[::-1]
        x_axis = np.array(list(range(len(self.vs))))
        plt.errorbar(x_axis-0.2, np.array(data)[order] , fmt='b+', linewidth=4, capsize=5,label='Data',markersize=8)
        plt.errorbar(x_axis+0.2, np.array(tcrpeg)[order] , fmt='ro', linewidth=1, capsize=3,label='TCRpeg',markersize=3)
                
        plt.xticks(x_axis,np.array(self.vs)[order])
        plt.xticks(rotation=45)
        plt.ylabel('Frequency',size=12)
        plt.xlabel('V Gene',size=12)
        plt.grid()
        plt.legend()
        #plt.title('J USAGE DISTRIBUTION',fontsize=20)
        plt.tight_layout()
        if fig_name is not None:
            plt.savefig('results/pictures/'+fig_name + '.jpg',dpi=200)
        else: plt.show()

    def density_scatter(self, x , y, ax = None, method='tcrpeg',sort = True, bins = 20, fig_name=None )   :
        """
        Scatter plot colored by 2d histogram
        """
        if ax is None :
            fig , ax = plt.subplots()
        bins = [100, 500] # number of bins
        # histogram the data
        hh, locx, locy = np.histogram2d(x, y, bins=bins)
        hh=hh#/hh.max()
        z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
        idx = z.argsort()
        x2, y2, z2 = x[idx], y[idx], z[idx]
        map_reversed = matplotlib.cm.get_cmap('viridis_r')
        s=ax.scatter(x2, y2, c=z2, cmap=map_reversed,s=10,alpha=1.,rasterized=True)
        tick_locator = ticker.MaxNLocator(nbins=4)
        cb=plt.colorbar(s,ax=ax,ticks=tick_locator)
        cb.ax.tick_params(labelsize=5)
        ax.text(0.1, 0.9, method, ha='center', va='center',transform = ax.transAxes,size=10,color='k')
        #plt.tight_layout()
        if fig_name is not None:
            plt.savefig('results/pictures/'+fig_name + '.jpg',dpi=200)
        #else: plt.show()


    def plot_model_learning(self,train,val,label, save_name = None):
        
        """Plots L1 convergence curve
        Parameters
        ----------
        save_name : str or None
            File name to save output figure. If None (default) does not save.
        """

        fig = plt.figure(figsize =(4, 4))
        
        # fig.add_subplot(122)
        # plt.title('Likelihood', fontsize = 15)
        plt.plot(train,label='train',c='k')
        plt.plot(val,label='validation',c='r')
        plt.legend(fontsize = 10)
        plt.xlabel('Iteration', fontsize = 13)
        plt.ylabel(label, fontsize = 13)

        fig.tight_layout()

        if save_name is not None:
            fig.savefig(save_name)
        else: plt.show()

    def plot_violin(self,a1,a2,epitopes,save_name=None,method1='TCRpeg-c',method2 = 'TCRGP' ):
        '''
        a1, a2 should a list of list, each list should contain the metric value for each experiment
        Note that each sub_list should correspond to the epitope in epitopes (the order should be match)
        '''
        a1,a2,epitopes = np.array(a1),np.array(a2),np.array(epitopes) #a1 is 5 x 22
        es,vs,ms = [],[],[]
        means = [-np.mean(x) for x in a1]
        index = np.argsort(means)
        a1,a2,epitopes = a1[index],a2[index],epitopes[index]
        for i in range(len(a1)):
            vs += list(a1[i])
            vs += list(a2[i])
            es += ([epitopes[i]]  * (len(a1[i]) * 2))
            ms += [method1] * len(a1[i])
            ms += [method2]* len(a2[i])
    #         print(len(vs))
    #         print(len(es))
        df = pd.DataFrame({'Epitope':es,'AUPRC':vs,'Method':ms})
        plt.figure(figsize = (20,7))
        sns.violinplot(data = df, y = "AUPRC", x = "Epitope", hue = "Method")
        plt.grid(axis='x')
        plt.xlabel('Epitope',size=12)
        plt.ylabel('AUPRC',size=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(save_name+'.jpg',dpi=200)

    def plot_sup(self,a1,a2,save_name=None ,xlabel='Mean AUPRC of TCRGP',ylabel='Mean AUPRC of TCRpeg-c'):
        '''
        a1, a2 should a list of list, each list should contain the metric value for each experiment 1 is TCRpeg
        Note that each sub_list should correspond to the epitope in epitopes (the order should be match)
        '''
        a1,a2 = np.array(a1),np.array(a2)
        es,vs,ms = [],[],[]
        mean_1,mean_2 = np.mean(a1,1),np.mean(a2,1)
        infer = (mean_1 < mean_2)
        colors = np.array(['#578EB9'] * 22)
        colors[infer] = '#FFA500'
        
        plt.figure(figsize = (6,4.5))
        plt.scatter(mean_2,mean_1,c=colors)
        xpoints = ypoints = plt.xlim()
        plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=2, scalex=False, scaley=False)
        plt.xlabel(xlabel,size=12)
        plt.ylabel(ylabel,size=12)
        if save_name is not None:
            plt.savefig(save_name+'.jpg',dpi=200)

    def plot_auc3(self,a1,a2,a3,y_true1,y_true2,y_true3,name1='TCRpeg-c',name2='DeepCAT',name3='jyp',save_name = None):
        '''
        list of list, each list is the prediction scores
        for example, 5 x len(y_true)
        '''
        fig, ax = plt.subplots()
        tprs_1,aucs_1 = [] ,[]
        tprs_2,aucs_2 = [] ,[]
        tprs_3,aucs_3 = [] ,[]
        mean_fpr = np.linspace(0, 1, 1000)
        #scores = classifier.predict_proba(X[test])
        for i in range(len(a1)):
            viz_1 = roc_curve(
                    y_true1[i],
                    a1[i]
                )
            viz_2 = roc_curve(
                    y_true2[i],
                    a2[i]
                )
            viz_3 = roc_curve(
                    y_true3[i],
                    a3[i]
                )
            interp_tpr = np.interp(mean_fpr, viz_1[0], viz_1[1])
            interp_tpr[0] = 0.0
            tprs_1.append(interp_tpr)
            aucs_1.append(roc_auc_score(y_true1[i],a1[i]))
            interp_tpr = np.interp(mean_fpr, viz_2[0], viz_2[1])
            interp_tpr[0] = 0.0
            tprs_2.append(interp_tpr)
            aucs_2.append(roc_auc_score(y_true2[i],a2[i]))
            interp_tpr = np.interp(mean_fpr, viz_3[0], viz_3[1])
            interp_tpr[0] = 0.0
            tprs_3.append(interp_tpr)
            aucs_3.append(roc_auc_score(y_true3[i],a3[i]))
        # exit()
        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="k", alpha=0.8)
        mean_tpr1 = np.mean(tprs_1, axis=0)
        mean_tpr2 = np.mean(tprs_2, axis=0)
        mean_tpr3 = np.mean(tprs_3, axis=0)
        mean_tpr1[-1] = 1.0
        mean_tpr2[-1] = 1.0
        mean_tpr3[-1] = 1.0
        mean_auc1 = auc(mean_fpr, mean_tpr1)
        mean_auc2 = auc(mean_fpr, mean_tpr2)
        mean_auc3 = auc(mean_fpr, mean_tpr3)    
        std_auc1 = np.std(aucs_1)  
        std_auc2 = np.std(aucs_2) 
        std_auc3 = np.std(aucs_3) 


        ax.plot(
            mean_fpr,
            mean_tpr1,
            color="b",
            label=name1+ r" (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc1, std_auc1),
            lw=1.5,
            alpha=0.8,
        )
        ax.plot(
            mean_fpr,
            mean_tpr2,
            color="r",
            label=name2+ r" (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc2, std_auc2),
            lw=1.5,
            alpha=0.8,
        )
        ax.plot(
            mean_fpr,
            mean_tpr3,
            color="g",
            label=name3+ r" (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc3, std_auc3),
            lw=1.5,
            alpha=0.8,
        )
        std_tpr1 = np.std(tprs_1, axis=0)
        tprs_upper = np.minimum(mean_tpr1 + std_tpr1, 1)
        tprs_lower = np.maximum(mean_tpr1 - std_tpr1, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="b",
            alpha=0.2
        )
        std_tpr2 = np.std(tprs_2, axis=0)
        tprs_upper = np.minimum(mean_tpr2 + std_tpr2, 1)
        tprs_lower = np.maximum(mean_tpr2 - std_tpr2, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="r",
            alpha=0.2
        )
        std_tpr3 = np.std(tprs_3, axis=0)
        tprs_upper = np.minimum(mean_tpr3 + std_tpr3, 1)
        tprs_lower = np.maximum(mean_tpr3 - std_tpr3, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="r",
            alpha=0.2
        )
        plt.legend(loc="lower right")
        plt.xlabel('False Positive Rate',size=12)
        plt.ylabel('True Positive Rate',size=12)
        if save_name is not None:
            plt.savefig(save_name + '.jpg',dpi=200)
        else :
            plt.show()

    def plot_aucs(self,pres,trues,names,colors,save_name = None):
        '''
        unified version of plot_auc 
        '''
        fig, ax = plt.subplots()
        tprs,aucs = [],[]
        for i in range(len(pres)):
            tprs.append([])
            aucs.append([])
        mean_fpr = np.linspace(0, 1, 1000)

        for i in range(len(pres[0])): #repeat times
            for k in range(len(pres)): #of plots
                viz = roc_curve(
                    trues[k][i],
                    pres[k][i]
                )
                interp_tpr = np.interp(mean_fpr, viz[0], viz[1])
                interp_tpr[0] = 0.0
                tprs[k].append(interp_tpr)
                aucs[k].append(roc_auc_score(trues[k][i],pres[k][i]))
        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="k", alpha=0.8)
        mean_tprs, mean_aucs, std_aucs = [],[],[]
        for i in range(len(pres)):
            temp = np.mean(tprs[i],axis=0)
            temp[-1] = 1.0
            mean_tprs.append(temp)
            mean_aucs.append(auc(mean_fpr, temp))
            std_aucs.append(np.std(aucs[i]))
            ax.plot(
            mean_fpr,
            mean_tprs[i],
            color=colors[i],
            label=names[i]+ r" (AUC = %0.3f $\pm$ %0.3f)" % (mean_aucs[i], std_aucs[i]),
            lw=1.5,
            alpha=0.8,
            )
            std_tpr = np.std(tprs[i], axis=0)
            tprs_upper = np.minimum(mean_tprs[i] + std_tpr, 1)
            tprs_lower = np.maximum(mean_tprs[i] - std_tpr, 0)
            ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=colors[i],
            alpha=0.2
            )
        plt.legend(loc="lower right")
        plt.xlabel('False Positive Rate',size=12)
        plt.ylabel('True Positive Rate',size=12)
        if save_name is not None:
            plt.savefig(save_name + '.jpg',dpi=200)
        else :
            plt.show()

    def plot_auc(self,a1,a2,y_true1,y_true2,name1='TCRpeg-c',name2='DeepCAT',save_name = None):
        '''
        list of list, each list is the prediction scores
        for example, 5 x len(y_true)
        '''
        fig, ax = plt.subplots()
        tprs_1,aucs_1 = [] ,[]
        tprs_2,aucs_2 = [] ,[]
        mean_fpr = np.linspace(0, 1, 1000)
        #scores = classifier.predict_proba(X[test])
        for i in range(len(a1)):
            viz_1 = roc_curve(
                    y_true1[i],
                    a1[i]
                )
            viz_2 = roc_curve(
                    y_true2[i],
                    a2[i]
                )
            interp_tpr = np.interp(mean_fpr, viz_1[0], viz_1[1])
            interp_tpr[0] = 0.0
            tprs_1.append(interp_tpr)
            aucs_1.append(roc_auc_score(y_true1[i],a1[i]))
            interp_tpr = np.interp(mean_fpr, viz_2[0], viz_2[1])
            interp_tpr[0] = 0.0
            tprs_2.append(interp_tpr)
            aucs_2.append(roc_auc_score(y_true2[i],a2[i]))
        print(aucs_1)   
        print(aucs_2)
        # exit()
        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="k", alpha=0.8)
        mean_tpr1 = np.mean(tprs_1, axis=0)
        mean_tpr2 = np.mean(tprs_2, axis=0)
        mean_tpr1[-1] = 1.0
        mean_tpr2[-1] = 1.0
        mean_auc1 = auc(mean_fpr, mean_tpr1)
        mean_auc2 = auc(mean_fpr, mean_tpr2)
        std_auc1 = np.std(aucs_1)  
        std_auc2 = np.std(aucs_2) 
        
        
        ax.plot(
            mean_fpr,
            mean_tpr1,
            color="b",
            label=name1+ r" (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc1, std_auc1),
            lw=1.5,
            alpha=0.5,
        )
        ax.plot(
            mean_fpr,
            mean_tpr2,
            color="r",
            label=name2+ r" (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc2, std_auc2),
            lw=1.5,
            alpha=0.8,
        )
        std_tpr1 = np.std(tprs_1, axis=0)
        tprs_upper = np.minimum(mean_tpr1 + std_tpr1, 1)
        tprs_lower = np.maximum(mean_tpr1 - std_tpr1, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="b",
            alpha=0.2
        )
        std_tpr2 = np.std(tprs_2, axis=0)
        tprs_upper = np.minimum(mean_tpr2 + std_tpr2, 1)
        tprs_lower = np.maximum(mean_tpr2 - std_tpr2, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="r",
            alpha=0.2
        )
        plt.legend(loc="lower right")
        plt.xlabel('False Positive Rate',size=12)
        plt.ylabel('True Positive Rate',size=12)
        if save_name is not None:
            plt.savefig(save_name + '.jpg',dpi=200)
        else :
            plt.show()

    def plot_prc(self,a1,a2,y_true1,y_true2,name1='TCRpeg-c',name2='DeepCAT',save_name = None):
        '''
        list of list, each list is the prediction scores
        for example, 5 x len(y_true)
        '''
        fig, ax = plt.subplots()
        tprs_1,aucs_1 = [] ,[]
        tprs_2,aucs_2 = [] ,[]
        mean_fpr = np.linspace(0, 1, 1500)
        #scores = classifier.predict_proba(X[test])
        for i in range(len(a1)):
            p1,r1,_ = pccurve(
                    y_true1[i],
                    a1[i]
                )
            p2,r2,_ = pccurve(
                    y_true2[i],
                    a2[i]
                )
            r1,p1,r2,p2 = r1[::-1],p1[::-1],r2[::-1],p2[::-1]
            interp_tpr = np.interp(mean_fpr, r1, p1)
            print(interp_tpr)
            interp_tpr[0] = 1.0
            tprs_1.append(interp_tpr)
            aucs_1.append(AUPRC(y_true1[i],a1[i]))
            interp_tpr = np.interp(mean_fpr, r2, p2)
            interp_tpr[0] = 1.0
            tprs_2.append(interp_tpr)
            aucs_2.append(AUPRC(y_true2[i],a2[i]))
        #ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="k", alpha=0.8)
        mean_tpr1 = np.mean(tprs_1, axis=0)
        mean_tpr2 = np.mean(tprs_2, axis=0)
        # mean_tpr1[-1] = 0.0
        # mean_tpr2[-1] = 0.0
        mean_auc1 = auc(mean_fpr, mean_tpr1)
        mean_auc2 = auc(mean_fpr, mean_tpr2)
        std_auc1 = np.std(aucs_1)  
        std_auc2 = np.std(aucs_2) 
        print(aucs_1)
        print(aucs_2)
        
        ax.plot(
            mean_fpr,
            mean_tpr1,
            color="b",
            label=name1+ r" (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc1, std_auc1),
            lw=1.5,
            alpha=0.5,
        )
        ax.plot(
            mean_fpr,
            mean_tpr2,
            color="r",
            label=name2+ r" (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc2, std_auc2),
            lw=1.5,
            alpha=0.8,
        )
        std_tpr1 = np.std(tprs_1, axis=0)
        tprs_upper = np.minimum(mean_tpr1 + std_tpr1, 1)
        tprs_lower = np.maximum(mean_tpr1 - std_tpr1, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="b",
            alpha=0.2
        )
        std_tpr2 = np.std(tprs_2, axis=0)
        tprs_upper = np.minimum(mean_tpr2 + std_tpr2, 1)
        tprs_lower = np.maximum(mean_tpr2 - std_tpr2, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="r",
            alpha=0.2
        )
        plt.legend(loc="lower left")
        plt.xlabel('Recall',size=12)
        plt.ylabel('Precisiion',size=12)
        if save_name is not None:
            plt.savefig(save_name + '.jpg',dpi=200)
        else :
            plt.show()

    def plot_prcs(self,pres,trues,names,colors,save_name = None):
        '''
        list of list, each list is the prediction scores
        for example, 5 x len(y_true)
        '''
        fig, ax = plt.subplots()
        tprs,aucs = [],[]
        for i in range(len(pres)):
            tprs.append([])
            aucs.append([])
        mean_fpr = np.linspace(0, 1, 1000)

        for i in range(len(pres[0])): #repeat times
            for k in range(len(pres)): #of plots
                p,r,_ = pccurve(
                    trues[k][i],
                    pres[k][i]
                )
                r,p = r[::-1],p[::-1]
                interp_tpr = np.interp(mean_fpr, r,p)
                interp_tpr[0] = 1.0
                tprs[k].append(interp_tpr)
                aucs[k].append(AUPRC(trues[k][i],pres[k][i]))
    #     ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="k", alpha=0.8)
        
        mean_tprs, mean_aucs, std_aucs = [],[],[]
        for i in range(len(pres)):
            temp = np.mean(tprs[i],axis=0)
            mean_tprs.append(temp)
            mean_aucs.append(auc(mean_fpr, temp))
            std_aucs.append(np.std(aucs[i]))
            ax.plot(
            mean_fpr,
            mean_tprs[i],
            color=colors[i],
            label=names[i]+ r" (AUC = %0.3f $\pm$ %0.3f)" % (mean_aucs[i], std_aucs[i]),
            lw=1.5,
            alpha=0.8,
            )
            std_tpr = np.std(tprs[i], axis=0)
            tprs_upper = np.minimum(mean_tprs[i] + std_tpr, 1)
            tprs_lower = np.maximum(mean_tprs[i] - std_tpr, 0)
            ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=colors[i],
            alpha=0.2
            )
        plt.legend(loc="upper right")
        plt.xlabel('Recall',size=12)
        plt.ylabel('Precision',size=12)
        if save_name is not None:
            plt.savefig(save_name + '.jpg',dpi=300)
        else :
            plt.show()

    def plot_auc_aug(self, inputs,y_true, nums ,save_name=None):
        fig, ax = plt.subplots()
        #originals = [np.mean([auc(y_true,x) for x in input]) for input in inputs]
        mean_fpr = np.linspace(0, 1, 1000)
        for k in range(len(nums)):
            tprs_1,aucs_1 = [] ,[]
            for i in range(len(inputs[k])):
                viz_1 = roc_curve(
                        y_true[k][i],
                        inputs[k][i]
                    )
                interp_tpr = np.interp(mean_fpr, viz_1[0], viz_1[1])
                interp_tpr[0] = 0.0
                tprs_1.append(interp_tpr)
                aucs_1.append(roc_auc_score(y_true[k][i],inputs[k][i]))
                
            ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="k", alpha=0.8)
            mean_tpr1 = np.mean(tprs_1, axis=0)
            mean_tpr1[-1] = 1.0
            mean_auc1 = auc(mean_fpr, mean_tpr1)
            std_auc1 = np.std(aucs_1)  
            
            if k == len(nums) - 1:
                ax.plot(
                mean_fpr,
                mean_tpr1,
                label=r"No augmentation (AUC = %0.3f $\pm$ %0.3f)" % (0.897, 0.002),
                lw=2,
                alpha=1,
                color='k'
                )
                std_tpr1 = np.std(tprs_1, axis=0)
                tprs_upper = np.minimum(mean_tpr1 + std_tpr1, 1)
                tprs_lower = np.maximum(mean_tpr1 - std_tpr1, 0)
                ax.fill_between(
                    mean_fpr,
                    tprs_lower,
                    tprs_upper,
                    alpha=0.1,
                    color='k'
                )
            else :
                ax.plot(
                    mean_fpr,
                    mean_tpr1,
                    label=r"%d (AUC = %0.3f $\pm$ %0.3f)" % (nums[k],np.mean(aucs_1), std_auc1),
                    lw=1,
                    alpha=0.5,
                )
                std_tpr1 = np.std(tprs_1, axis=0)
                tprs_upper = np.minimum(mean_tpr1 + std_tpr1, 1)
                tprs_lower = np.maximum(mean_tpr1 - std_tpr1, 0)
                ax.fill_between(
                    mean_fpr,
                    tprs_lower,
                    tprs_upper,
                    alpha=0.2
                )
        plt.legend(loc="lower right")
        plt.xlabel('False Positive Rate',size=12)
        plt.ylabel('True Positive Rate',size=12)
        if save_name is not None:
            plt.savefig(save_name + '.jpg',dpi=200)
        else :
            plt.show()

    def augmentation(self,nums,vs,original=None,method_name='Augmentation by TCRpeg',ax=None,save_name=None):
        '''
        original is just a list
        nums is the number of generated seqs; vs is the performances under that num, should be in shape len(nums)x n
        #now is only for TCRpeg
        '''
        if ax is None:
            fig, ax = plt.subplots()
        mean_v = np.mean(vs,axis=1)
        print(mean_v)
        std_v = np.std(vs,axis=1) / np.sqrt(5)
        print(std_v)
        ax.plot(nums,
            mean_v,
            label=method_name,
            lw=2,
            alpha=0.8)
        lower,upper = mean_v + std_v, mean_v - std_v
        ax.fill_between(
            nums,
            lower,
            upper,
            alpha=0.2
        )
        if original is not None:
            mean = np.mean(original)
            mean = np.array([mean]*len(nums))
            std = np.std(original)/5
            lower,upper = mean + std, mean - std
            ax.plot(nums, mean,label='No augmentation',lw=2,alpha=0.8)
            ax.fill_between(nums,lower,upper,alpha=0.2)
            
        plt.xlabel('Number of generated TCR sequences',size=12)
        plt.ylabel('AUROC',size=12)
        plt.legend(loc='upper right')
        if save_name is not None:
            plt.savefig(save_name + '.jpg',dpi=200)
        else :
            plt.show()

    def Length_Dis(self,seqs1:list,seqs2 :list,save_path=None):
        'plot the length distribution for a single model in one experiment'

        lens1 = [seq for seq in seqs1 if self.valid_seq(seq)]
        lens1,lens2 = [len(seq) for seq in seqs1],[len(seq) for seq in seqs2]
        len2count1,len2count2 = defaultdict(int),defaultdict(int)
        for i in range(len(lens1)):        
            len2count1[lens1[i]] += 1
        for i in range(len(lens2)):
            len2count2[lens2[i]] += 1
        k1,k2 = list(len2count1.keys()),list(len2count2.keys())
        x_axis = list(range(1,31))
        fre1,fre2 = [len2count1[k]/len(lens1) for k in x_axis],[len2count2[k]/len(lens2) for k in x_axis]
        plt.figure()
        plt.plot(x_axis,fre1,x_axis,fre2)
        plt.legend(['Data','Generated'])
        plt.xlabel('TCR Length')
        plt.ylabel('Frequency')
        if save_path is not None:
            plt.savefig(save_path+ '.png')
        else :
            plt.show()

    def AAs_Dis(self,seqs1:list,seqs2:list,save_path = None):
        'plot the position distribution for AA, for single model and single experiment'        
        aa2pos_dic1 = [defaultdict(int) for _ in range(20)]
        aa2pos_dic2 = [defaultdict(int) for _ in range(20)]
        for i in range(len(seqs1)):
            seq1 = seqs1[i]
            if not self.valid_seq(seq1):
                continue
            for j,aa in enumerate(seq1):
                #if aa == 'e' or 
                aa2pos_dic1[self.aa2idx[aa]][j+1] += 1
        for i in range(len(seqs2)):
            seq2 = seqs2[i]
            if not self.valid_seq(seq2):
                continue
            for j,aa in enumerate(seq2):
                aa2pos_dic2[self.aa2idx[aa]][j+1] += 1
        
        fig,axs = plt.subplots(4,5,figsize=(20,15))
        x_axis = list(range(1,31))
        for row in range(4):
            for col in range(5):
                index = row*5 + col
                d1,d2,aa = aa2pos_dic1[index],aa2pos_dic2[index],self.idx2aa[index]
                y1,y2 = [d1[k] for k in x_axis],[d2[k] for k in x_axis]
                sum1,sum2 = sum(y1),sum(y2)            
                y1,y2 = [k/sum1 for k in y1],[k/sum2 for k in y2]
                axs[row,col].plot(x_axis,y1,x_axis,y2)

                axs[row,col].legend(['Data','Generated'])
                axs[row,col].set_title('AA: ' +aa)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path +  '.png')
        else :
            plt.show()

    def plot_prob(self,p_data,p_infer,save_path=None):
        'plot the scatter plot for Fig.1 in the paper'
        fig=plt.figure(figsize=(4,2),dpi=200)
        ax1=plt.subplot(111)
        ax1.set_ylim([-21,0])
        ax1.set_xlim([-5.5,-2.5])
        ax1.locator_params(nbins=4)
        ax1.set_xlabel(r'$log_{10}P_{data}$')
        ax1.set_ylabel(r'$log_{10}P_{infer}$')
        ax1.plot([-5.5, -2.5], [-5.5, -2.5], color='k', linestyle='-', linewidth=2)
        #Plot.density_scatter(np.log10(p_data),np.log10(p_tcrpeg),bins = [10,50],ax=ax1,fig_name='prob_tcrpeg',method='TCRpeg')
        self.density_scatter(np.log10(p_data),np.log10(p_infer),bins = [10,50],ax=ax1,method='TCRpeg')
        r = pr(p_data,p_infer)[0]    
        ax1.text(0.65, 0.32, r'r = %0.3f' % (r) , ha='center', va='center',transform = ax1.transAxes,size=10,color='k')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path+'.jpg',dpi=200)
        else :
            plt.show()