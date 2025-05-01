#收集原始数据，并设置负样本，对原始数据进行处理
python ./Codes/AttnTAP_data_clean.py --input_file=./data/raw_file.csv --output_file=./results/example_data_clean.csv --neg_samples=1


python ./Codes/AttnTAP_data_clean.py --input_file=./data/train.csv --output_file=./results/train_data_clean.csv --neg_samples=1
python ./Codes/AttnTAP_data_clean.py --input_file=./data/test.csv --output_file=./results/test_data_clean.csv --neg_samples=1



#使用数据集对模型进行训练
python  ./Codes/AttnTAP_train.py --input_file=./results/train_data_clean.csv  --save_model_file=./Models/mc_vad_train.pth --valid_set=False --epoch=15 --learning_rate=0.005 --dropout_rate=0.1 --embedding_dim=10 --hidden_dim=50 --plot_train_curve=False --plot_roc_curve=False



#使用测试集对模型进行测试
python ./Codes/AttnTAP_test.py --input_file=./results/test_data_clean.csv  --output_file=./results/mc_vdj.csv --load_model_file=./Models/mc_vad_train.pth






