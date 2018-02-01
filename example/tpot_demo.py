from sklearn import model_selection
from tpot import TPOTClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
Y = np.array(df['class'])
 
X_trian,X_test,Y_train,Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

'''
tpot = TPOTClassifier(generations=6, verbosity=2)
tpot.fit(X_trian, Y_train)
tpot.score(X_test, Y_test)
# 导出
tpot.export('pipeline.py')
'''
clf = RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.4, min_samples_leaf=3, min_samples_split=3, n_estimators=100)
clf.fit(X_trian, Y_train)
accuracy = clf.score(X_test, Y_test)
print(accuracy)