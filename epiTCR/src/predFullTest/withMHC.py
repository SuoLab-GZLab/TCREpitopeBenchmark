import pandas as pd
import modules.processor as Processor
import modules.model as Model
import pickle


full_test = pd.read_csv("../../data/split-data/with-mhc/test/fulltest.csv")
pX_full_test = Processor.dataRepresentationBlosum62WithMHCb(full_test)
py_full_test = full_test[["binder"]]


model_rf = pickle.load(open('./models/rdforestWithMHCModel.pickle', 'rb'))
auc_data_full, acc_data_full, sens_data_full, spec_data_full = Model.predicMLModel(model_rf, full_test, pX_full_test, py_full_test, '../../data/predict-data/full-testset/predict_withmhc.csv')
