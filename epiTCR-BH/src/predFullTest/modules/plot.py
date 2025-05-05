
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report,roc_curve,auc, f1_score, plot_roc_curve 


def evaluationGroupedBarChart(pdata, phead=5):
    fig = go.Figure()
    pdata = pdata.head(phead)
    x_labels = [f'{x}' for x in (pdata['model'])]

    fig.add_trace(go.Bar(x=x_labels, y=pdata['train_acc']*100, name='Train Acc'))
    fig.add_trace(go.Bar(x=x_labels, y=pdata['test_acc']*100, name='Test Acc'))
    fig.add_trace(go.Bar(x=x_labels, y=pdata['train_roc_auc']*100, name='Train ROC-AUC'))
    fig.add_trace(go.Bar(x=x_labels, y=pdata['test_roc_auc']*100, name='Test ROC-AUC'))

    fig.show()

def roc_auc(model, X_test, y_test):
    ax = plt.subplot()
    n = X_test.shape[0]
    mean_fpr = np.linspace(0, 1, n)
    tprs = []
    aucs = []
    viz = plot_roc_curve(model, X_test, y_test,
                            name='Random Forest',
                            alpha=.3, lw=2, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    label='Chance', alpha=.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
    title="ROC curve comparison")
    ax.legend(loc="lower right")
    plt.show()