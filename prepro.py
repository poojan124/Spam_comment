import pandas as pd
import numpy as np
from bs4 import BeautifulSoup 
import warnings

def get_clean():
    files = ['Youtube01-Psy.csv','Youtube02-KatyPerry.csv','Youtube03-LMFAO.csv','Youtube04-Eminem.csv','Youtube05-Shakira.csv']
    frames = [pd.read_csv(f) for f in files]
    data = pd.concat(frames,axis=0)
    data = data.drop(['COMMENT_ID','AUTHOR','DATE'],axis=1)
    print("Data shape ",end=":")
    print(data.shape)
    return data

def process(): 
    warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
    data = get_clean()
    data['CONTENT'] = data['CONTENT'].apply(lambda x: BeautifulSoup(x,'html.parser').get_text())

    
if __name__=='__main__':
    process()