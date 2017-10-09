import pandas as pd
import numpy as np
from bs4 import BeautifulSoup 
import warnings
import re,string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.model_selection  import train_test_split

# -*- coding: utf-8 -*-
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

#global Data
words = set(stopwords.words('english'))

'''
Description of all Methods:
---------------------------
-->get_clean : Purpose : Read files and remove uneccesory files. - Parameter : None - Return : pandas dataframe object   

-->process : Purpose : Parse html code into text using beautiful soup- Parameters : pandas dataframe object- Return : pandas dataframe object

-->remove_punc : Purpose : remove punctuation and all numerical numbers from string leaving string with only alphabates
                 Parameters : pandas dataframe - Returns : pandas dataframe

-->stop_rem : helper method
                 
-->remove_stopwords : Purpose : remoce stop words using NLTK library and stop_rem helper method - Parameters : Pandas
                    dataframe obj - Return : pandas dataframe    

-->feature selection : Purpose : using term frequecy of a word in corpus select some most frequent word as features - 
                      Parameters : Pandas dataframe obj - Return : pandas dataframe    

-->preprocess : Purpose : run all needed mothod for preprocess data in one go. - Parameters : None - Returns : a list of    
                objects [train-data, test-data, features]

'''
       
def get_clean():
    files = ['Youtube01-Psy.csv','Youtube02-KatyPerry.csv','Youtube03-LMFAO.csv','Youtube04-Eminem.csv',
            'Youtube05-Shakira.csv']
    frames = [pd.read_csv(f) for f in files]
    data = pd.concat(frames,axis=0)
    data = data.drop(['COMMENT_ID','AUTHOR','DATE'],axis=1)
    print("Data shape ",end=":")
    print(data.shape)
    return data

def process(data): 
    warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
    data['CONTENT'] = data['CONTENT'].apply(lambda x: BeautifulSoup(x,'html.parser').get_text())
    return data

def stop_rem(s):
    new_s = ''
    word_list = word_tokenize(s)
    word_list = ["_link_feature_" if x.find('http')!=-1 else x for x in word_list]
    word_list = ["_link_feature_" if x.find('watch')!=-1 and len(x)>9 else x for x in word_list]
    for w in word_list :
        if w.lower() not in words and isEnglish(w):
            new_s = new_s + w.lower() + ' '
    return new_s
    
def remove_punc(data):
    data['CONTENT'] = data['CONTENT'].apply(lambda x: re.sub('[^a-zA-Z ]', '', x))
    return data

def remove_stopwords(data):
    data['CONTENT'] = data['CONTENT'].apply(lambda x: stop_rem(x))
    return data
    
    
def feature_selection(data_train):
    token = []
    for idx,x in data_train.iterrows():
        token = token + word_tokenize(x['CONTENT'])
    print(len(token))
    token = ["_link_feature_" if x.find('http')!=-1 else x for x in token]
    token = ["_link_feature_" if x.find('watch')!=-1 and len(x)>9 else x for x in token]
    c= Counter(token)
    features = [x for x,y in c.items() if y>3]
    print("Total Number of features to use : " + str(len(features)))
    return features

def preprocess():
    data = get_clean()
    data = process(data)
    data = remove_punc(data)
    data = remove_stopwords(data)
    data_train,data_test = train_test_split(data,test_size = 0.25,random_state = 42)
    features = feature_selection(data_train)
    return [data_train, data_test, features]
    
if __name__=='__main__':
    dt,dte,feature = np.array(preprocess()).ravel()
    print(dt.shape)
    print(dte.shape)