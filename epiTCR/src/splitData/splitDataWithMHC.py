import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import zipfile
from sklearn.model_selection import train_test_split


###---------------

df = pd.read_csv("../../data/finalData/finalWithHLA.csv")

df = df[["CDR3b", "epitope", "binder", "HLA"]]
X = df.iloc[:, lambda df: ["CDR3b", "epitope", "HLA"]]
y = df.iloc[:, lambda df: ["binder"]]

X_train,  X_test,   y_train,  y_test   = train_test_split(X, y, test_size=0.9, random_state=42)
X_test1,  X_test2,  y_test1,  y_test2  = train_test_split(X_test, y_test, test_size=0.111, random_state=42)
X_test3,  X_test4,  y_test3,  y_test4  = train_test_split(X_test1, y_test1, test_size=0.125, random_state=42)
X_test5,  X_test6,  y_test5,  y_test6  = train_test_split(X_test3, y_test3, test_size=0.145, random_state=42)
X_test7,  X_test8,  y_test7,  y_test8  = train_test_split(X_test5, y_test5, test_size=0.17, random_state=42)
X_test9,  X_test10, y_test9,  y_test10 = train_test_split(X_test7, y_test7, test_size=0.205, random_state=42)
X_test11, X_test12, y_test11, y_test12 = train_test_split(X_test9, y_test9, test_size=0.25, random_state=42)
X_test13, X_test14, y_test13, y_test14 = train_test_split(X_test11, y_test11, test_size=0.33, random_state=42)
X_test15, X_test16, y_test15, y_test16 = train_test_split(X_test13, y_test13, test_size=0.5, random_state=42)

###---------------

test = pd.concat([X_test, y_test], axis=1)
test = test.reset_index(drop=True)
test.to_csv("../../data/splitData/withMHC/test/fulltest.csv", index=False)

train = pd.concat([X_train, y_train], axis=1)
train = train.reset_index(drop=True)
train.to_csv("../../data/splitData/withoutMHC/train/train.csv", index=False)

###---------------

test01 = pd.concat([X_test2, y_test2], axis=1)
test01 = test01.reset_index(drop=True)

test01_1 = test01[test01["binder"] == 1]
test01_0 = test01[test01["binder"] == 0]
test01_0 = test01_0.sample(n = 6682 * 10, random_state = 1)
test01 = pd.concat([test01_0, test01_1], axis = 0)
test01 = test01.reset_index(drop=True)
test01.to_csv("../../data/splitData/withoutMHC/test/test01.csv", index=False)

###---------------

test02 = pd.concat([X_test4, y_test4], axis=1)
test02 = test02.reset_index(drop=True)

test02_1 = test02[test02["binder"] == 1]
test02_0 = test02[test02["binder"] == 0]
test02_0 = test02_0.sample(n = 6587 * 10, random_state = 1)
test02 = pd.concat([test02_0, test02_1], axis = 0)
test02 = test02.reset_index(drop=True)
test02.to_csv("../../data/splitData/withoutMHC/test/test02.csv", index=False)

###---------------

test03 = pd.concat([X_test6, y_test6], axis=1)
test03 = test03.reset_index(drop=True)

test03_1 = test03[test03["binder"] == 1]
test03_0 = test03[test03["binder"] == 0]
test03_0 = test03_0.sample(n = 6721 * 10, random_state = 1)
test03 = pd.concat([test03_0, test03_1], axis = 0)
test03 = test03.reset_index(drop=True)

test03.to_csv("../../data/splitData/withoutMHC/test/test03.csv", index=False)

###---------------

test04 = pd.concat([X_test8, y_test8], axis=1)
test04 = test04.reset_index(drop=True)

test04_1 = test04[test04["binder"] == 1]
test04_0 = test04[test04["binder"] == 0]
test04_0 = test04_0.sample(n = 6728 * 10, random_state = 1)
test04 = pd.concat([test04_0, test04_1], axis = 0)
test04 = test04.reset_index(drop=True)

test04.to_csv("../../data/splitData/withoutMHC/test/test04.csv", index=False)    

###---------------

test05 = pd.concat([X_test10, y_test10], axis=1)
test05 = test05.reset_index(drop=True)

test05_1 = test05[test05["binder"] == 1]
test05_0 = test05[test05["binder"] == 0]
test05_0 = test05_0.sample(n = 6843 * 10, random_state = 1)
test05 = pd.concat([test05_0, test05_1], axis = 0)
test05 = test05.reset_index(drop=True)

test05.to_csv("../../data/splitData/withoutMHC/test/test05.csv", index=False)

###---------------


test06 = pd.concat([X_test12, y_test12], axis=1)
test06 = test06.reset_index(drop=True)

test06_1 = test06[test06["binder"] == 1]
test06_0 = test06[test06["binder"] == 0]
test06_0 = test06_0.sample(n = 6685 * 10, random_state = 1)
test06 = pd.concat([test06_0, test06_1], axis = 0)
test06 = test06.reset_index(drop=True)

test06.to_csv("../../data/splitData/withoutMHC/test/test06.csv", index=False)

###---------------

test07 = pd.concat([X_test14, y_test14], axis=1)
test07 = test07.reset_index(drop=True)

test07_1 = test07[test07["binder"] == 1]
test07_0 = test07[test07["binder"] == 0]
test07_0 = test07_0.sample(n = 6387 * 10, random_state = 1)
test07 = pd.concat([test07_0, test07_1], axis = 0)
test07 = test07.reset_index(drop=True)

test07.to_csv("../../data/splitData/withoutMHC/test/test07.csv", index=False)

###---------------

test08 = pd.concat([X_test16, y_test16], axis=1)
test08 = test08.reset_index(drop=True)

test08_1 = test08[test08["binder"] == 1]
test08_0 = test08[test08["binder"] == 0]
test08_0 = test08_0.sample(n = 6584 * 10, random_state = 1)
test08 = pd.concat([test08_0, test08_1], axis = 0)
test08 = test08.reset_index(drop=True)

test08.to_csv("../../data/splitData/withoutMHC/test/test08.csv", index=False)

###---------------

test09 = pd.concat([X_test15, y_test15], axis=1)
test09 = test09.reset_index(drop=True)

test09_1 = test09[test09["binder"] == 1]
test09_0 = test09[test09["binder"] == 0]
test09_0 = test09_0.sample(n = 6580 * 10, random_state = 1)
test09 = pd.concat([test09_0, test09_1], axis = 0)
test09 = test09.reset_index(drop=True)

test09.to_csv("../../data/splitData/withoutMHC/test/test09.csv", index=False)