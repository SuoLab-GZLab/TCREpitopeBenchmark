<h1 align="center">
 TCREpitopeBenchmark
</h1>

## Overview
> In this study, we conducted a comprehensive evaluation of TCR-epitope binding prediction models, focusing on their performance in both seen and unseen epitope scenarios and identifying key factors that influence model performance.

<img width="1140" src="/result_path/20250501-201723.jpg">
	
## Code Structure
There are three modules for each model: (1)original model-based prediction, (2)model retraining, and (3)retrained model-based prediction on both seen and unseen data. You can select either to retrain the model or to generate predictions depending on your requirements. The relevant code for each model, contained in a Jupyter notebook, is saved in a separate folder with a file name that matches the model's name. Additionally, we provide the scripts for preprocessing all original datasets ```(Original_model_data_statistics.ipynb)```, prepping the datasets used in model retraining ```(Raw_database_data_filtering.ipynb)```, matching negative datasets ```(Negative_dataset_matching.ipynb)```,  and plotting all the figures shown in the article ```(fig)```。



## Requirements
You need to create a separate environment for each model as these models are based on different Python framework and packages. During model evaluation, we configured a separate environment for each model according to the requirements file provided in its GitHub repository. All environments were stored in the ``` environment ``` folder for easy access. The time required to configure the environment depends on factors such as the number of model dependencies, network speed, and hardware performance.

## Input File Format

The model input should be a CSV file with the following format:
```
Epitope,CDR3B,Affinity
```

Where the epitope and CDR3B(TCR) are protein sequences and binding affinity is either 0 or 1.

```
# Example
Epitope,CDR3B,Affinity
GLCTLVAML,CASSEGQVSPGELF,1
GLCTLVAML,CSATGTSGRVETQYF,0
```

If your data is unlabeled and you are only interested in the function of prediction, simply fill the ‘Affinity’ column with either 0 or 1. The performance statistics can be ignored in this case and the predicted binding affinity scores can be collected from the output files.

## Make Predictions Using Original Models
The code for making predictions using original models has been encapsulated into a function named ```Original_model_prediction```.  You can directly call this function in the first module of the Jupyter notebook within the respective model folder. As an illustration, the original ATM-TCR model can be used with the following code snippet:

```
testfile_path="../data/test.csv"
modelfile_path="../Original_model/ATM_TCR.ckpt"
result_path="../result_path/Original_model_prediction"
Original_model_prediction(testfile_path,modelfile_path,result_path)
```

The original models other than ATM_TCR have been deposited on the figshare website. You can download them by visiting this link (https://doi.org/10.6084/m9.figshare.27020455) and navigating to the "Original_model" folder.

## Model Retraining 
We have refactored the training and testing code for each model into a function named ```Model_retraining```. You can directly call this function in the second module of the Jupyter notebook within the respective model folder. As an illustration, the ATM-TCR model can be retrained using the following code snippet:

```
trainfile_path ="../data/train.csv"
testfile_path="../data/test.csv"
save_model_path="../Retraining_model/ATM_TCR.ckpt"
result_path="../result_path/Retraining_model_prediction"
Model_retraining(trainfile_path,testfile_path,save_model_path,result_path) 
```

## Make Predictions Using Retrained Models

If you want to use your own trained model for prediction, you can call the ```Retraining_model_prediction ``` function in the third module of the Jupyter notebook. As an illustration, the retrained ATM-TCR model can be used with the following code snippet :
```
testfile_path="../data/Validation.csv"
modelfile_path="../Retraining_model/ATM_TCR.ckpt"
result_path="../result_path/Retraining_model_prediction"
Retraining_model_prediction(testfile_path,modelfile_path,result_path)
```

We have uploaded the retrained models to the figshare website. You can download them by visiting this link (https://doi.org/10.6084/m9.figshare.27020455) and accessing the "Retrained_model" folder.

## Model Output
The prediction results of each model are stored in the result_path directory, comprising the columns ``` Epitope, CDR3B,y_true, y_pred, and y_prob ```. Here, y_prob represents the predicted probability of TCR binding to the epitope, and y_pred indicates the binding status (“1”-binding, “0”-not binding) based on the probability. If y_prob is greater than or equal to 0.5, y_pred is set to 1; otherwise, y_pred is set to 0.

``` 
# Example
Epitope,CDR3B, y_true, y_pred, y_prob
GLCTLVAML,CASSEGQVSPGELF,1,1,0.89
GLCTLVAML,CSATGTSGRVETQYF,0,1,0.68
```

If you already know the actual TCR and epitope binding labels, you can calculate the model prediction accuracy using the Jupyter notebook file named ```Evaluation_metrics_calculation ``` . You can directly call the  ``` calculate ```  function from that file using the following code snippet:
``` 
data_path="../result_path/predition.csv"
result_path="../result_path/predition"
column='epitope'
calculate(data_path, result_path, column)
```

Note: Due to code permission issues, the models DLpTCR-FULL, DLpTCR-CNN, and DLpTCR-RESNET can only be accessed and used via the GitHub repository (https://github.com/JiangBioLab/DLpTCR/).
