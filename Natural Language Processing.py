
# Normalizing- text means converting it to a more convenient, standard form.
#Tokenization- seperating words by whitespace
#lemmatization- the task of determining that two words have the same root,
#Stemming refers to a simpler version of lemmatization in which we mainly 
    #   just strip suffixes from the end of the word.
#Text normalization also includes sentence segmentation: breaking up a text into individual 
    #   sentences, using cues like sentence segmentation periods or exclamation points.
#Edit distance that measures how similar two strings are based on the number
    #of edits (insertions, deletions, substitutions) it takes to change one string 
    # into the other.

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import matplotlib
from matplotlib import pyplot as plt
import random

# import seaborn as sns
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

f=open("SMSSpamCollection.txt", "r")
contents =f.readlines()
dd=[]
for lines in contents:
    dd.append(lines)

data=pd.DataFrame(dd,columns=['Comments'])

data['target']=[random.randint(1,2) for i in range(5574) ]

col='Comments'
target='target'



def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    import re
    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    text = re.sub(r'<.*?>', '', text)
    
    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)    
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)    
    
    # convert text to lowercase
    text = text.strip().lower()
    
    # replace punctuation characters with spaces
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)
    # return the text stripped of punctuation marks
    return text

data[col] = data[col].apply(remove_punctuation)

def stop_words(text):
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    sw = stopwords.words('english')
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)

data[col] = data[col].apply(stop_words)


#We will not use word counts as feature for NLP since tf-idf is a better metric
def count_vector(data,col,plot=True):
    # create a count vectorizer object
    count_vectorizer = CountVectorizer()
    # fit the count vectorizer using the text data
    count_vectorizer.fit(data[col])
    # collect the vocabulary items used in the vectorizer
    dictionary = count_vectorizer.vocabulary_.items()  
    # lists to store the vocab and counts
    vocab = []
    count = []
    # iterate through each vocab and count append the value to designated lists
    for key, value in dictionary:
        vocab.append(key)
        count.append(value)
    # store the count in panadas dataframe with vocab as index
    vocab_bef_stem = pd.Series(count, index=vocab)
    # sort the dataframe
    vocab_bef_stem = vocab_bef_stem.sort_values(ascending=False)
    if plot:
        top_vacab = vocab_bef_stem.head(20)
        top_vacab.plot(kind = 'barh', figsize=(5,5), xlim= (10, 9000))
    return(vocab_bef_stem)

a=count_vector(data,col)


#Stemming operations
#Stemming operation bundles together words of same root. E.g. 
# stem operation bundles "response" and "respond" into a common "respon"

# create an object of stemming function


def stemming(text):    
    stemmer = SnowballStemmer("english")
    '''a function which stems each word in the given text'''
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text) 

data[col] = data[col].apply(stemming)

#Top words after stemming operation
def tf_id_vec(data,col,plot=True):
    # create the object of tfid vectorizer
    tfid_vectorizer = TfidfVectorizer("english")
    # fit the vectorizer using the text data
    tfid_vectorizer.fit(data[col])
    # collect the vocabulary items used in the vectorizer
    dictionary = tfid_vectorizer.vocabulary_.items()  
    # lists to store the vocab and counts
    vocab = []
    count = []
    # iterate through each vocab and count append the value to designated lists
    for key, value in dictionary:
        vocab.append(key)
        count.append(value)
    # store the count in panadas dataframe with vocab as index
    vocab_after_stem = pd.Series(count, index=vocab)
    # sort the dataframe
    vocab_after_stem = vocab_after_stem.sort_values(ascending=False)
    # plot of the top vocab
    if plot:
        top_vacab = vocab_after_stem.head(20)
        top_vacab.plot(kind = 'barh', figsize=(5,5), xlim= (10, 9000))
    return(vocab_after_stem)

a=tf_id_vec(data,col)

def length(text):    
    '''a function which returns the length of text'''
    return len(text)

data['length'] = data[col].apply(length)


#TF-IDF Extraction

#tf-idf weight is product of two terms: the first term is the normalized 
# Term Frequency (TF), aka. the number of times a word appears in a document, 
# divided by the total number of words in that document; the second term is the 
# Inverse Document Frequency (IDF), computed as the logarithm of the number of 
# the documents in the corpus divided by the number of documents where the 
# specific term appears.

#TF(t) = (Number of times term t appears in a document) / (Total number of 
#                                       terms in the document).

#IDF(t) = log_e(Total number of documents / Number of documents with term t 
#                               in it).

# extract the tfid representation matrix of the text data
tfid_vectorizer = TfidfVectorizer("english")
tfid_vectorizer.fit(data[col])
tfid_matrix = tfid_vectorizer.transform(data[col])
# collect the tfid matrix in numpy array
array = tfid_matrix.todense()

# store the tf-idf array into pandas dataframe
df = pd.DataFrame(array)

df['output'] = data[target]

#data[target].value_counts()

features = df.columns.tolist()
output = 'output'
# removing the output and the id from features
features.remove(output)

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV

#List of alpha parameter we are going to try
alpha_list1 = np.linspace(0.006, 0.1, 20)
alpha_list1 = np.around(alpha_list1, decimals=4)

# parameter grid
parameter_grid = [{"alpha":alpha_list1}]

#binomial classificaiton

def bi_class(data, features,output):
    # classifier object
    classifier1 = BernoulliNB()
    # gridsearch object using 4 fold cross validation and neg_log_loss as scoring paramter
    model = GridSearchCV(classifier1,parameter_grid, scoring = 'neg_log_loss', cv = 4)
    # fit the gridsearch
    model.fit(df[features], df[output])

    results1 = pd.DataFrame()
    # collect alpha list
    results1['alpha'] = model.cv_results_['param_alpha'].data
    # collect test scores
    results1['neglogloss'] = model.cv_results_['mean_test_score'].data

    matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
    plt.plot(results1['alpha'], -results1['neglogloss'])
    plt.xlabel('alpha')
    plt.ylabel('logloss')
    plt.grid()
    print("Best parameter: ",model.best_params_)
    print("Best score: ",model.best_score_) 

bi_class(data,features,output)
#multinomial classificaiton

def multi_class(data,features,output):
    # classifier object
    classifier2 = MultinomialNB()
    # gridsearch object using 4 fold cross validation and neg_log_loss as scoring paramter
    model = GridSearchCV(classifier2,parameter_grid, scoring = 'neg_log_loss', cv = 4)
    # fit the gridsearch
    model.fit(df[features], df[output])

    results2 = pd.DataFrame()
    # collect alpha list
    results2['alpha'] = model.cv_results_['param_alpha'].data
    # collect test scores
    results2['neglogloss'] = model.cv_results_['mean_test_score'].data

    matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
    plt.plot(results2['alpha'], -results2['neglogloss'])
    plt.xlabel('alpha')
    plt.ylabel('logloss')
    plt.grid()

    print("Best parameter: ",model.best_params_)

    print("Best score: ",model.best_score_)
    return model
