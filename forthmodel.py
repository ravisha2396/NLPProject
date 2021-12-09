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
import PySimpleGUI as sg          # get the higher-level GUI package

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

################################################################################
# Function to compute final balance given specified time and payments          #
# Input:                                                                       #
#    window  - the top-level widget                                            #
#    entries - the dictionary of entry fields                                  #
# Output:                                                                      #
#    only output is to screen (for debug) and the GUI                          #
################################################################################

def final_balance(window,entries):
   # # period (monthly) rate:
   # r = (float(entries[FN_RATE]) / 100) / 12
   # #print("r", r)
   teststr = str(entries[FN_SENTENCE])
   # # get the remaining values
   # loan = float(entries[FN_PRINC])
   # n =  float(entries[FN_NUMPAY]) 
   # monthly = float(entries[FN_MONTHPAY])

   # # compute the compounding and the remaining balance
   # q = (1 + r)** n
   # remaining = q * loan  - ( (q - 1) / r) * monthly
   # remaining = ("%8.2f" % remaining).strip()
   
   testout = (model.predict(to_vw_format(teststr))+1)*.5
   
   # put the values into the GUI and print to the screen
   window[FN_INSULT].Update(testout)
   #print("Remaining Loan: %f" % float(remaining))
   

# define the field names and their indices
# FN_RATE = 'Sentence:'
# FN_NUMPAY = 'Number of Payments'
# FN_PRINC = 'Loan Principle'
# FN_MONTHPAY = 'Monthly Payment'
# FN_REMAINS = 'Remaining Loan'
FN_SENTENCE = 'Sentence'
FN_INSULT = 'Scale of Insult (0 to 1)'
FIELD_NAMES = [ FN_SENTENCE, FN_INSULT ]
# F_RATE = 0 # index for annual rate
# F_NUMPAY = 1 # index for number of payments
# F_PRINC = 2 # index for loan principle
# F_MONTHPAY = 3 # index for monthly payment
# F_REMAINS = 4 # index for remaining loan
NUM_FIELDS = 2 # how many fields there are
B_BALANCE = 'Test Sentence' # need things in more than one place…
#B_PAYMENT = 'Monthly Payment'
B_QUIT = 'Quit'
#BK_PAYMNT = 'payment' # needed to differentiate button from field

sg.set_options(font=('Helvetica',20))
layout = [] # start with the empty list
for index in range(NUM_FIELDS): # for each of the fields to create
    layout.append([sg.Text(FIELD_NAMES[index]+': ',size=(20,1)), \
                   sg.InputText(key=FIELD_NAMES[index],size=(50,1))])
layout.append([sg.Button(B_BALANCE), \
                 sg.Button(B_QUIT)])

# start the window manager
window = sg.Window('Insult Detector',layout)

# Run the event loop
while True:
    event, values = window.read()
    print(event,values)
    if event == sg.WIN_CLOSED or event == B_QUIT:
        break
    if event == B_BALANCE:
        final_balance(window,values)
    # elif event == BK_PAYMNT:
    #     monthly_payment(window,values)

window.close()
# n = 0
# while n == 0:
#     print('Which model to try?')
#     print('1: Vowpalwabbit')
#     print('2: Exit')
#     cho = input()
#     if cho == '2':
#         n = 1
#     else:
#         teststr = input('What sentance do you want to test? \n')
#         if cho == '1':
#             testout = (model.predict(to_vw_format(teststr))+1)*.5
#             print('Output:'+str(testout))
    
    