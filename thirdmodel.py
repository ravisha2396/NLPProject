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
from sklearn.tree import DecisionTreeClassifier

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

boo = []
for sublist in dftrain['sd_type'].values:
    if 'Flamewar (insulting)' in str(sublist):
        boo.append(1)
    else:
        boo.append(-1)
#print([int(inner) for inner in boo])
#print(dftrain["sd_type"].loc[boo])
dftrain['y'] = boo
lst_stopwords = stopwords.words("english")
dftrain["text_clean"] = dftrain["text"].apply(lambda x: 
          utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, 
          lst_stopwords=lst_stopwords))
emp = [not bool(tc) for tc in dftrain['text_clean']]
dfblank = dftrain.loc[emp]
ind = dfblank.index
print(ind)
dftrain.to_csv('dftrain_before.csv')
dftrain = dftrain.drop(ind)
dftrain.to_csv('dftrain_after.csv')
corpus = dftrain['text_clean']
lst_corpus = []
for string in corpus:
   lst_words = string.split()
   lst_grams = [" ".join(lst_words[i:i+1]) 
               for i in range(0, len(lst_words), 1)]
   lst_corpus.append(lst_grams)
dftrain['text_clean'] = lst_corpus
print(dftrain)
print(sum(dftrain['y']))
X_train, X_test, Y_train, Y_test = train_test_split(dftrain['text_clean'],dftrain['y'],test_size=.3)
X_train = X_train.reset_index()
X_test = X_test.reset_index()
Y_train = Y_train.to_frame()
Y_train = Y_train.reset_index()
Y_test = Y_test.to_frame()
Y_test = Y_test.reset_index()

#X_train.to_csv('X_train.csv')

size = 1000
window = 3
min_count = 1
workers = 3
sg = 1

word2vec_model_file =  'word2vec_300.model'
#stemmed_tokens = pd.Series(dftrain['text_clean']).values
#print(lst_corpus)
# Train the Word2Vec Model
w2v_model = word2vec.Word2Vec(lst_corpus, min_count = min_count, vector_size = size, workers = workers, window = window, sg = sg)
#print("Time taken to train word2vec model: " + str(time.time() - start_time))
w2v_model.save(word2vec_model_file)

# Load the model from the model file
sg_w2v_model = word2vec.Word2Vec.load(word2vec_model_file)

# word = 'every'
# print(sg_w2v_model.wv[word])
# # Unique ID of the word
# # print("Index of the word 'every':")
# # print(sg_w2v_model.wv.vocab["every"].index)
# # Total number of the words 
# print(len(sg_w2v_model.wv.vocab))
# # Print the size of the word2vec vector for one word
# print("Length of the vector generated for a word")
# print(len(sg_w2v_model['action']))
# # Get the mean for the vectors for an example review
# print("Print the length after taking average of all word vectors in a sentence:")
# print(np.mean([sg_w2v_model[token] for token in dftrain['text_clean'][0]], axis=0))

word2vec_filename = 'train_review_word2vec.csv'
with open(word2vec_filename, 'w+') as word2vec_file:
    for index, row in X_train.iterrows():
        model_vector = (np.mean([sg_w2v_model.wv[token] for token in row['text_clean']], axis=0)).tolist()
        if index == 0:
            header = ",".join(str(ele) for ele in range(1000))
            word2vec_file.write(header)
            word2vec_file.write("\n")
        # Check if the line exists else it is vector of zeros
        if type(model_vector) is list:  
            line1 = ",".join( [str(vector_element) for vector_element in model_vector] )
        else:
            line1 = ",".join([str(0) for i in range(1000)])
        word2vec_file.write(line1)
        word2vec_file.write('\n')
word2vec_df = pd.read_csv(word2vec_filename)
#Initialize the model
clf_decision_word2vec = DecisionTreeClassifier()

# Fit the model
clf_decision_word2vec.fit(word2vec_df, Y_train['y'])

test_features_word2vec = []
for index, row in X_test.iterrows():
    model_vector = np.mean([sg_w2v_model.wv[token] for token in row['text_clean']], axis=0)
    if type(model_vector) is list:
        test_features_word2vec.append(model_vector)
    else:
        test_features_word2vec.append(np.array([0 for i in range(1000)]))
test_predictions_word2vec = clf_decision_word2vec.predict(test_features_word2vec)
print(metrics.classification_report(Y_test['y'],test_predictions_word2vec))