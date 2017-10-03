import pandas as pd
import numpy as np
from bs4 import BeautifulSoup 
import warnings
import re,string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
    for w in word_list :
        if w.lower() not in words and isEnglish(w):
            new_s = new_s + w.lower() + ' '
    return new_s
    
def remove_punc(data):
    data['CONTENT'] = data['CONTENT'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
    return data

def remove_stopwords(data):
    data['CONTENT'] = data['CONTENT'].apply(lambda x: stop_rem(x))
    print(data.head())
    return data
    
if __name__=='__main__':
    data = get_clean()
    data = process(data)
    data = remove_punc(data)
    data = remove_stopwords(data)