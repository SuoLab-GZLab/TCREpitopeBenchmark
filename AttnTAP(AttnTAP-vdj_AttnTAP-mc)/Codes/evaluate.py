import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, matthews_corrcoef, precision_score,auc
def Evaluate_all_indicators(test,result):
    test = pd.read_csv(test)
    result = pd.read_csv(result)
    result['class'] = test['antigen']

    # Calculate metrics for each class
    metrics = []
    fpr_all = []
    tpr_all = []
    roc_auc_all = []

    for cls in result['class'].unique():
        class_df = result[result['class'] == cls]
        y_true = class_df['y_true']
        y_pred = class_df['y_pred']
        y_prob = class_df['y_prob']

        acc = accuracy_score(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_prob)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)  # Added precision calculation
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        metrics.append({
            'Class': cls,
            'ACC': acc,
            'AUC': auc_score,
            'Recall': recall,
            'Precision': precision,  # Added Precision metric
            'F1': f1,
            'MCC': mcc
        })

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fpr_all.append(fpr)
        tpr_all.append(tpr)
        roc_auc_all.append(roc_auc)
    metrics_class = pd.DataFrame(metrics)

    # Calculate overall metrics
    y_true_all = result['y_true']  # Modified from df to result
    y_pred_all = result['y_pred']  # Modified from df to result
    y_prob_all = result['y_prob']  # Modified from df to result

    acc_all = accuracy_score(y_true_all, y_pred_all)
    auc_all = roc_auc_score(y_true_all, y_prob_all)
    recall_all = recall_score(y_true_all, y_pred_all)
    precision_all = precision_score(y_true_all, y_pred_all)  # Added Precision calculation
    f1_all = f1_score(y_true_all, y_pred_all)
    mcc_all = matthews_corrcoef(y_true_all, y_pred_all)


    metrics_all = pd.DataFrame({
        'Metric': ['ACC', 'AUC', 'Recall', 'Precision', 'F1', 'MCC'],
        'Value': [acc_all, auc_all, recall_all, precision_all, f1_all, mcc_all]})

    import pandas as pd

    all_value_row = pd.DataFrame({
        'Class': ['all_Value'],
        'ACC': [metrics_all.loc[0, 'Value']],
        'AUC': [metrics_all.loc[1, 'Value']],
        'Recall': [metrics_all.loc[2, 'Value']],
        'Precision': [metrics_all.loc[3, 'Value']],
        'F1': [metrics_all.loc[4, 'Value']],
        'MCC': [metrics_all.loc[5, 'Value']]
    })

    # Append the new row to metrics_class dataframe
    metrics_class = pd.concat([metrics_class, all_value_row], ignore_index=True)
    metrics_class.to_csv('../evaluate/all_result.csv')
    metrics_class
    
    
    
#使用方法 
#test=path
#result=path
#Evaluate_all_indicators（test,result）
