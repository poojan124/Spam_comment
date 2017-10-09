from prepro import preprocess
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize

#global data 
data_train ,data_test ,features  = np.array(preprocess()).ravel()

#prior probability of spam and ham
pspam = 0.0
pham = 0.0
#counter array to calculate probability
counter_words = dict((key,[0,0]) for key in features)

#array to store probability of each word in spam and ham
prob_spam = dict((key,0.0) for key in features)
prob_ham = dict((key,0.0) for key in features)

def prior_prob():
    total = data_train.shape[0]
    spamcounter = 0
    hamcounter = 0
    for value in data_train['CLASS']:
        if value:
            spamcounter +=1
        else:
            hamcounter +=1
    pspam = spamcounter/float(total)
    pham = hamcounter/float(total)
    return [pspam,pham]
    
def cal_prob():
    #index 0 of counter_words value is for spam and 1 index value for non spam coutner
    #EX : {'check':[12,5]} means 'check' word appear 12 times in spam mail and 5 times in non spam mail
    for idx,x in data_train.iterrows():
        tokens = word_tokenize(x['CONTENT'])
        tokens = ["_link_feature_" if x.find('http')!=-1 else x for x in tokens]
        tokens = ["_link_feature_" if x.find('watch')!=-1 and len(x)>9 else x for x in tokens]
        tokens = [x for x in tokens if x in features]
        if x['CLASS']==1:
            for word in tokens:
                cntr = counter_words[word]
                cntr[0] +=1
                counter_words[word] = cntr
        else:
            for word in tokens:
                cntr = counter_words[word]
                cntr[1] +=1
                counter_words[word] = cntr
    #calculating probability
    for key,value in counter_words.items():
            prob_spam[key] = value[0]/float(value[0]+value[1])
            prob_ham[key] = value[1]/float(value[0]+value[1])
    return [prob_spam,prob_ham]

pspam,pham = np.array(prior_prob()).ravel()
prob_spam,prob_ham = np.array(cal_prob()).ravel()    

def classify(s):
    prospam = 1.0
    proham = 1.0
    input_ = word_tokenize(s)
    input_ = ["_link_feature_" if x.find('http')!=-1 else x for x in input_]
    input_ = ["_link_feature_" if x.find('watch')!=-1 and len(x)>9 else x for x in input_]
    input_ = [x for x in input_ if x in features]
    for inp in input_:
        prospam *= prob_spam[inp]
        proham *= prob_ham[inp]
    print("spam ->" + str(prospam*pspam))
    print("ham ->" + str(proham*pham))
    if prospam*pspam > proham*pham:
        return 1
    else:
        return 0
        
if __name__ == "__main__":
    pspam,pham = np.array(prior_prob()).ravel()
    prob_spam,prob_ham = np.array(cal_prob()).ravel()
    
    output = [classify(S) for S in data_test['CONTENT']]
    matrix = confusion_matrix(data_test['CLASS'], output)
    acc = accuracy_score(data_test['CLASS'], output)*100
    tn,fp,fn,tp = matrix.ravel()

    print("Confustion Matrix : \n" + str(matrix),end='\n\n')
    print(str(acc) + "%")