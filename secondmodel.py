import pandas as pd
import numpy as np
import re
from sklearn import datasets
from sklearn.model_selection import train_test_split
from vowpalwabbit.sklearn_vw import VWClassifier
from sklearn import metrics # accuracy_score, recall_score, roc_auc_score, roc_curve # grade the results
from sklearn.preprocessing import StandardScaler # standardize data
import matplotlib.pyplot as plt
from vowpalwabbit import pyvw
from gensim.models import word2vec
import gensim
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from tensorflow.keras import models, layers, preprocessing as kprocessing
#from sklearn.tree import DecisionTreClassifer
from sklearn.decomposition import PCA
#from tensorflow.keras import backend as K

'''
Preprocess a string.
:parameter
    :param text: string - name of column containing text
    :param lst_stopwords: list - list of stopwords to remove
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''
def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text

df = pd.read_csv('ydata-ynacc-v1_0/ydata-ynacc-v1_0_expert_annotations.tsv',sep='\t')
dft = pd.read_csv('ydata-ynacc-v1_0/ydata-ynacc-v1_0_train-ids.txt', header=None)
#print([item for sublist in df['sd_type'].values for item in sublist])
dftrain = df[ df['sdid'].isin([item for sublist in dft.values for item in sublist])][['text','sd_type']]
#x = [sublist for sublist in dftrain['sd_type'].values]
print(dftrain)
#print([True for sublist in x if 'Flamewar (insulting)' in str(sublist)])
#print(len(dftrain['sd_type'].values))
#print(len(x))
boo = []
for sublist in dftrain['sd_type'].values:
    if 'Flamewar (insulting)' in str(sublist):
        boo.append('insulting')
    else:
        boo.append('not insulting')
#print([int(inner) for inner in boo])
#print(dftrain["sd_type"].loc[boo])
dftrain['y'] = boo
print(dftrain.sample(20))
lst_stopwords = stopwords.words("english")
dftrain["text_clean"] = dftrain["text"].apply(lambda x: 
          utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, 
          lst_stopwords=lst_stopwords))
print(dftrain)
dftrain_train, dftrain_cv = train_test_split(dftrain,test_size=.3)
print(dftrain_train)
print(dftrain_cv)
y_train = dftrain_train["y"].values
y_cv = dftrain_cv["y"].values
#print(len(y_cv))
corpus = dftrain_train["text_clean"]

## create list of lists of unigrams
lst_corpus = []
for string in corpus:
   lst_words = string.split()
   lst_grams = [" ".join(lst_words[i:i+1]) 
               for i in range(0, len(lst_words), 1)]
   lst_corpus.append(lst_grams)
#print(lst_corpus)

nlp = word2vec.Word2Vec(lst_corpus, vector_size=300,   
            window=8, min_count=1, sg=1)
word = "homeless"
fig = plt.figure()
## word embedding
tot_words = [word] + [tupla[0] for tupla in 
                 nlp.wv.most_similar(word, topn=20)]
X = nlp.wv[tot_words]
## pca to reduce dimensionality from 300 to 3
pca = PCA( n_components=3)
X = pca.fit_transform(X)
## create dtf
dtf_ = pd.DataFrame(X, index=tot_words, columns=["x","y","z"])
dtf_["input"] = 0
dtf_["input"].iloc[0:1] = 1
## plot 3d
#from mpl_toolkits.mplot3d import Axes3D
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dtf_[dtf_["input"]==0]['x'], 
           dtf_[dtf_["input"]==0]['y'], 
           dtf_[dtf_["input"]==0]['z'], c="black")
ax.scatter(dtf_[dtf_["input"]==1]['x'], 
           dtf_[dtf_["input"]==1]['y'], 
           dtf_[dtf_["input"]==1]['z'], c="red")
ax.set(xlabel=None, ylabel=None, zlabel=None, xticklabels=[], 
       yticklabels=[], zticklabels=[])
for label, row in dtf_[["x","y","z"]].iterrows():
    x, y, z = row
    ax.text(x, y, z, s=label)
plt.show()
#print(nlp)


# ## tokenize text
# tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', 
#                      oov_token="NaN", 
#                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
# tokenizer.fit_on_texts(lst_corpus)
# dic_vocabulary = tokenizer.word_index
# ## create sequence
# lst_text2seq= tokenizer.texts_to_sequences(lst_corpus)
# ## padding sequence
# X_train = kprocessing.sequence.pad_sequences(lst_text2seq, 
#                     maxlen=15, padding="post", truncating="post")
# print(X_train)
# corpus = dftrain_cv["text_clean"]

# ## create list of n-grams
# lst_corpus = []
# for string in corpus:
#     lst_words = string.split()
#     lst_grams = [" ".join(lst_words[i:i+1]) for i in range(0, 
#                  len(lst_words), 1)]
#     lst_corpus.append(lst_grams)

# X_cv = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=15,
#              padding="post", truncating="post")
# print(len(X_cv))
# ## start the matrix (length of vocabulary x vector size) with all 0s
# embeddings = np.zeros((len(dic_vocabulary)+1, 300))
# for word,idx in dic_vocabulary.items():
#     ## update the row with vector
#     try:
#         embeddings[idx] =  nlp[word]
#     ## if word not in model then skip and the row stays all 0s
#     except:
#         pass

# ## code attention layer
# def attention_layer(inputs, neurons):
#     x = layers.Permute((2,1))(inputs)
#     x = layers.Dense(neurons, activation="softmax")(x)
#     x = layers.Permute((2,1), name="attention")(x)
#     x = layers.multiply([inputs, x])
#     return x

# ## input
# x_in = layers.Input(shape=(15,))
# ## embedding
# x = layers.Embedding(input_dim=embeddings.shape[0],  
#                      output_dim=embeddings.shape[1], 
#                      weights=[embeddings],
#                      input_length=15, trainable=False)(x_in)
# ## apply attention
# x = attention_layer(x, neurons=15)
# ## 2 layers of bidirectional lstm
# x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2, 
#                          return_sequences=True))(x)
# x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2))(x)
# ## final dense layers
# x = layers.Dense(64, activation='relu')(x)
# y_out = layers.Dense(3, activation='softmax')(x)
# ## compile
# model = models.Model(x_in, y_out)
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='adam', metrics=['accuracy'])

# model.summary()

# ## encode y
# dic_y_mapping = {n:label for n,label in 
#                  enumerate(np.unique(y_train))}
# inverse_dic = {v:k for k,v in dic_y_mapping.items()}
# y_train = np.array([inverse_dic[y] for y in y_train])
# ## train
# training = model.fit(x=X_train, y=y_train, batch_size=256, 
#                      epochs=10, shuffle=True, verbose=0, 
#                      validation_split=0.3)
# ## plot loss and accuracy
# # metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
# # fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
# # ax[0].set(title="Training")
# # ax11 = ax[0].twinx()
# # ax[0].plot(training.history['loss'], color='black')
# # ax[0].set_xlabel('Epochs')
# # ax[0].set_ylabel('Loss', color='black')
# # for metric in metrics:
# #     ax11.plot(training.history[metric], label=metric)
# # ax11.set_ylabel("Score", color='steelblue')
# # ax11.legend()
# # ax[1].set(title="Validation")
# # ax22 = ax[1].twinx()
# # ax[1].plot(training.history['val_loss'], color='black')
# # ax[1].set_xlabel('Epochs')
# # ax[1].set_ylabel('Loss', color='black')
# # for metric in metrics:
# #      ax22.plot(training.history['val_'+metric], label=metric)
# # ax22.set_ylabel("Score", color="steelblue")
# # plt.show()

# ## test
# predicted_prob = model.predict(X_cv)
# print(predicted_prob, len(predicted_prob))
# predicted = [dic_y_mapping[np.argmax(pred)] for pred in 
#              predicted_prob]

# classes = np.unique(y_cv)
# y_cv_array = pd.get_dummies(y_cv, drop_first=False).values

# print(predicted, len(predicted))
    
# ## Accuracy, Precision, Recall
# accuracy = metrics.accuracy_score(y_cv, predicted)
# auc = metrics.roc_auc_score(y_cv, predicted_prob, 
#                             multi_class="ovr")
# print("Accuracy:",  round(accuracy,2))
# print("Auc:", round(auc,2))
# print("Detail:")
# print(metrics.classification_report(y_cv, predicted))
    
# ## Plot confusion matrix
# cm = metrics.confusion_matrix(y_cv, predicted)
# fig, ax = plt.subplots()
# sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, 
#             cbar=False)
# ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, 
#        yticklabels=classes, title="Confusion matrix")
# plt.yticks(rotation=0)

# fig, ax = plt.subplots(nrows=1, ncols=2)
# ## Plot roc
# for i in range(len(classes)):
#     fpr, tpr, thresholds = metrics.roc_curve(y_cv_array[:,i],  
#                            predicted_prob[:,i])
#     ax[0].plot(fpr, tpr, lw=3, 
#               label='{0} (area={1:0.2f})'.format(classes[i], 
#                               metrics.auc(fpr, tpr))
#                )
# ax[0].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
# ax[0].set(xlim=[-0.05,1.0], ylim=[0.0,1.05], 
#           xlabel='False Positive Rate', 
#           ylabel="True Positive Rate (Recall)", 
#           title="Receiver operating characteristic")
# ax[0].legend(loc="lower right")
# ax[0].grid(True)
    
# ## Plot precision-recall curve
# for i in range(len(classes)):
#     precision, recall, thresholds = metrics.precision_recall_curve(
#                  y_cv_array[:,i], predicted_prob[:,i])
#     ax[1].plot(recall, precision, lw=3, 
#                label='{0} (area={1:0.2f})'.format(classes[i], 
#                                   metrics.auc(recall, precision))
#               )
# ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall', 
#           ylabel="Precision", title="Precision-Recall curve")
# ax[1].legend(loc="best")
# ax[1].grid(True)
# plt.show()