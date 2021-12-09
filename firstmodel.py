import pandas as pd
import numpy as np
import re
from sklearn import datasets
from sklearn.model_selection import train_test_split
from vowpalwabbit.sklearn_vw import VWClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve # grade the results
from sklearn.preprocessing import StandardScaler # standardize data
import matplotlib.pyplot as plt
from vowpalwabbit import pyvw
from gensim.models import word2vec


def to_vw_format(document, label=None):
    return str(label or '') + ' |text ' + ' '.join(re.findall('\w{3,}', document.lower())) + '\n'

train = pd.read_csv('detecting-insults-in-social-commentary/train.csv')
y = train[['Insult']].to_numpy()
X = train[['Comment']].to_numpy()
y = np.transpose(y)
y = y[0]
X_train_pre, X_test_pre, y_train, y_test = train_test_split(X,y, test_size=.3,random_state=0)
X_train = []
for i in np.arange(0,len(X_train_pre)):
    X_train.append( to_vw_format(np.array2string(X_train_pre[i]), 1 if y_train[i] == 1 else -1))
X_test = []
for i in np.arange(0,len(X_test_pre)):
    X_test.append( to_vw_format(np.array2string(X_test_pre[i])))
print('Example of training input:')
print(X_train[0])
model = pyvw.vw(quiet=True)
for tra in X_train:
    model.learn(tra)
test_prediction = []
test_predictionround = []
for tes in X_test:
    test_prediction.append((model.predict(tes)+1)*.5)
    test_predictionround.append(round((model.predict(tes)+1)*.5))
print('Example of testing input:')
print(X_test[0])
print('Example of predictied output:'+str(test_prediction[0]))
print('Expected output:' + str(y_test[0]))

#print('{:.2%}'.format(accuracy_score(y_test, test_predictionround)))


auc = roc_auc_score(y_test, test_prediction)
print('AUC:'+str(auc))
roc_curve = roc_curve(y_test, test_prediction)

# Interactive section
n = 0
while n == 0:
    print('Which model to try?')
    print('1: Vowpalwabbit')
    print('2: Exit')
    cho = input()
    if cho == '2':
        n = 1
    else:
        teststr = input('What sentance do you want to test? \n')
        if cho == '1':
            testout = (model.predict(to_vw_format(teststr))+1)*.5
            print('Output:'+str(testout))
    
    