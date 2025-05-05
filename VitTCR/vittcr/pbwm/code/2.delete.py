import pandas as pd 
import os 


def delete(repeat, fold):
    # Determine the selected epoch.
    metricpath = '../metrics_repeat{}'.format(repeat)  
    modelpath = '../model_repeat{}/fold{}'.format(repeat, fold)        
    temp = [i for i in os.listdir(metricpath) if i.startswith("Test_model") and i.endswith("fold{}.tsv".format(fold))][0]
    validname = "{}/{}".format(metricpath, temp)
    validdata = pd.read_csv(validname, sep='\t')            
    ind = validdata['loss'].idxmin()
    epoch = validdata.iloc[ind]['epoch']
    except_file = "{}_epochs.pt".format(str(int(epoch)))
    
    # Delete unselected epochs.
    for root, dirs, files in os.walk(modelpath):
        for item in files:
            if item != except_file:
                del_file=os.path.join(root, item)
                os.remove(del_file)
                print('remove file[%s] successfully' % del_file)


if __name__=="__main__":
    for repeat in range(5):
        for fold in range(5):
            delete(repeat=repeat, fold=fold)
