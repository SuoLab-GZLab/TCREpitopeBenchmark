import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import zipfile
from sklearn.model_selection import train_test_split

df = pd.read_csv("../../data/finalData/finalWithoutHLA.csv")

df = df.drop_duplicates()
df = df[["CDR3b", "epitope", "binder"]]
df = df.drop_duplicates()
X = df[['CDR3b', 'epitope']]
y = df[['binder']]

X_train,  X_test,   y_train,  y_test   = train_test_split(X, y, test_size=0.88, random_state=42)
X_test1,  X_test2,  y_test1,  y_test2  = train_test_split(X_test, y_test, test_size=0.14, random_state=42)
X_test3,  X_test4,  y_test3,  y_test4  = train_test_split(X_test1, y_test1, test_size=0.165, random_state=42)
X_test5,  X_test6,  y_test5,  y_test6  = train_test_split(X_test3, y_test3, test_size=0.2, random_state=42)
X_test7,  X_test8,  y_test7,  y_test8  = train_test_split(X_test5, y_test5, test_size=0.2, random_state=42)
X_test9,  X_test10, y_test9,  y_test10 = train_test_split(X_test7, y_test7, test_size=0.2, random_state=42)
X_test11, X_test12, y_test11, y_test12 = train_test_split(X_test9, y_test9, test_size=0.2, random_state=42)

## split data
test = pd.concat([X_test, y_test], axis = 1)
test = test.reset_index(drop=True)


train = pd.concat([X_train, y_train], axis=1)
train = train.reset_index(drop=True)

train_1 = train[train["binder"] == 1]
train_1_1 = train_1.iloc[:6684, :]
train_1_2 = train_1.iloc[6684:, :]

train_0 = train[train["binder"] == 0]
train_0_2 = train_0.iloc[0:62950, :]
train_0_1 = train_0.iloc[62950:, :]

train =  pd.concat([train_1_1, train_0_1], axis = 0)
trainp = pd.concat([train_1_2, train_0_2], axis = 0)
train =  train.reset_index(drop=True)
trainp = trainp.reset_index(drop=True)

train.to_csv("../../data/splitData/withoutMHC/train/train.csv", index=False)
trainp.to_csv("../../data/splitData/withoutMHC/test/test15.csv", index=False)

full_test = pd.concat([test, trainp], axis = 0)
full_test.to_csv("../../data/splitData/withoutMHC/test/full_test.csv", index=False)

###-------------------------

test01 = pd.concat([X_test2, y_test2], axis=1)
test01 = test01.reset_index(drop=True)

test01_1 = test01[test01["binder"] == 1]
test01_1_1 = test01_1.iloc[:6628, :]
test01_1_2 = test01_1.iloc[6628:, :]
test01_0 = test01[test01["binder"] == 0]
test01_0_1 = test01_0.sample(n = 6628 * 10, random_state = 1)
test01_0_2 = test01_0.sample(n = 6629 * 10, random_state = 2)
test01 =  pd.concat([test01_1_1, test01_0_1], axis = 0)
test01p = pd.concat([test01_1_2, test01_0_2], axis = 0)
test01 =  test01.reset_index(drop=True)
test01p = test01p.reset_index(drop=True)

test01.to_csv("../../data/splitData/withoutMHC/test/test01.csv", index=False)
test01p.to_csv("../../data/splitData/withoutMHC/test/test02.csv", index=False)

###-------------------------

test02 = pd.concat([X_test4, y_test4], axis=1)
test02 = test02.reset_index(drop=True)

test02_1 = test02[test02["binder"] == 1]
test02_1_1 = test02_1.iloc[:6695, :]
test02_1_2 = test02_1.iloc[6695:, :]
test02_0 = test02[test02["binder"] == 0]
test02_0_1 = test02_0.sample(n = 6695 * 10, random_state = 1)
test02_0_2 = test02_0.sample(n = 6695 * 10, random_state = 2)
test02 =  pd.concat([test02_1_1, test02_0_1], axis = 0)
test02p = pd.concat([test02_1_2, test02_0_2], axis = 0)
test02 =  test02.reset_index(drop=True)
test02p = test02p.reset_index(drop=True)

test02.to_csv("../../data/splitData/withoutMHC/test/test03.csv", index=False)
test02p.to_csv("../../data/splitData/withoutMHC/test/test04.csv", index=False)

###-------------------------

test03 = pd.concat([X_test6, y_test6], axis=1)
test03 = test03.reset_index(drop=True)

test03_1 = test03[test03["binder"] == 1]
test03_1_1 = test03_1.iloc[:6728, :]
test03_1_2 = test03_1.iloc[6728:, :]
test03_0 = test03[test03["binder"] == 0]
test03_0_1 = test03_0.sample(n = 6728 * 10, random_state = 1)
test03_0_2 = test03_0.sample(n = 6729 * 10, random_state = 2)
test03 =  pd.concat([test03_1_1, test03_0_1], axis = 0)
test03p = pd.concat([test03_1_2, test03_0_2], axis = 0)
test03 =  test03.reset_index(drop=True)
test03p = test03p.reset_index(drop=True)

test03.to_csv("../../data/splitData/withoutMHC/test/test05.csv", index=False)
test03p.to_csv("../../data/splitData/withoutMHC/test/test06.csv", index=False)

###-------------------------

test04 = pd.concat([X_test8, y_test8], axis=1)
test04 = test04.reset_index(drop=True)

test04_1 = test04[test04["binder"] == 1]
test04_1_1 = test04_1.iloc[:6667, :]
test04_1_2 = test04_1.iloc[6667:, :]
test04_0 = test04[test04["binder"] == 0]
test04_0_1 = test04_0.sample(n = 6667 * 10, random_state = 1)
test04_0_2 = test04_0.sample(n = 6668 * 10, random_state = 2)
test04 =  pd.concat([test04_1_1, test04_0_1], axis = 0)
test04p = pd.concat([test04_1_2, test04_0_2], axis = 0)
test04 =  test04.reset_index(drop=True)
test04p = test04p.reset_index(drop=True)

test04.to_csv("../data/splitData/withoutMHC/test/test07.csv", index=False)
test04p.to_csv("../data/splitData/withoutMHC/test/test08.csv", index=False)


###-------------------------

test05 = pd.concat([X_test10, y_test10], axis=1)
test05 = test05.reset_index(drop=True)

test05_1 = test05[test05["binder"] == 1]
test05_1_1 = test05_1.iloc[:6652, :]
test05_1_2 = test05_1.iloc[6652:, :]
test05_0 = test05[test05["binder"] == 0]
test05_0_1 = test05_0.sample(n = 6652 * 10, random_state = 1)
test05_0_2 = test05_0.sample(n = 6653 * 10, random_state = 2)
test05 =  pd.concat([test05_1_1, test05_0_1], axis = 0)
test05p = pd.concat([test05_1_2, test05_0_2], axis = 0)
test05 =  test05.reset_index(drop=True)
test05p = test05p.reset_index(drop=True)

test05.to_csv("../../data/splitData/withoutMHC/test/test09.csv", index=False)
test05p.to_csv("../../data/splitData/withoutMHC/test/test10.csv", index=False)

###-------------------------

test06 = pd.concat([X_test12, y_test12], axis=1)
test06 = test06.reset_index(drop=True)

test06_1 = test06[test06["binder"] == 1]
test06_1_1 = test06_1.iloc[:6688, :]
test06_1_2 = test06_1.iloc[6688:, :]
test06_0 = test06[test06["binder"] == 0]
test06_0_1 = test06_0.sample(n = 6688 * 10, random_state = 1)
test06_0_2 = test06_0.sample(n = 6688 * 10, random_state = 2)
test06 =  pd.concat([test06_1_1, test06_0_1], axis = 0)
test06p = pd.concat([test06_1_2, test06_0_2], axis = 0)
test06 =  test06.reset_index(drop=True)
test06p = test06p.reset_index(drop=True)

test06.to_csv("../../data/splitData/withoutMHC/test/test11.csv", index=False)
test06p.to_csv("../../data/splitData/withoutMHC/test/test12.csv", index=False)

###-------------------------

test07 = pd.concat([X_test11, y_test11], axis=1)
test07 = test07.reset_index(drop=True)

test07_1 = test07[test07["binder"] == 1]
test07_1_1 = test07_1.iloc[:6738, :]
test07_1_2 = test07_1.iloc[6738:, :]
test07_0 = test07[test07["binder"] == 0]
test07_0_1 = test07_0.sample(n = 6738 * 10, random_state = 1)
test07_0_2 = test07_0.sample(n = 6739 * 10, random_state = 2)
test07 =  pd.concat([test07_1_1, test07_0_1], axis = 0)
test07p = pd.concat([test07_1_2, test07_0_2], axis = 0)
test07 =  test07.reset_index(drop=True)
test07p = test07p.reset_index(drop=True)

test07.to_csv("../../data/splitData/withoutMHC/test/test13.csv", index=False)
test07p.to_csv("../../data/splitData/withoutMHC/test/test14.csv", index=False)