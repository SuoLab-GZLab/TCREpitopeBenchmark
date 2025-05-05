import pandas as pd
import modules.processor as Processor
import modules.model as Model
import pickle

full_test = pd.read_csv("../../data/split-data/without-mhc/test/fulltest.csv")
full_test = full_test[:5]
pX_full_test = Processor.dataRepresentationBlosum62WithoutMHCb(full_test)
py_full_test = full_test[["binder"]]


model_rf = pickle.load(open('./models/rdforestWithoutMHCModel.pickle', 'rb'))
auc_data_full, acc_data_full, sens_data_full, spec_data_full = Model.predicMLModel(model_rf, full_test, pX_full_test, py_full_test, '../../data/predict-data/full-testset/predict_withoutmhc.csv')