# -*- coding: utf-8 -*-
"""blood.ipynb

Automatically generated by Colaboratory.
"""

import pandas as pd
from sklearn.svm import SVC
from tpot import TPOTClassifier

X = []
Y = []
list_clf = []

df = pd.read_csv('drive2/Colab Notebooks/bloodtrain2.csv', na_values = {'?'})
df = df.values


X = df[:, :(df.shape[1]-1)]
Y = df[:, df.shape[1]-1]

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
tpot.fit(X, Y)

clf_svm = SVC()

list_clf = [clf_svm]

kf = KFold(n_splits=5)
kf.get_n_splits(X)
a = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    clfs.fit(X_test, Y_test)  
    print(clfs.score(X_train, Y_train)) 
    a += clfs.score(X_train, Y_train)
a = a/5
print("Average=",a,"\n")
print(clfs,"\n")