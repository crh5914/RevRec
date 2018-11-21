import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import numpy as np
def load_doc(prefix):
    docs = [] 
    with open(prefix+'_train.csv','r') as fp:
        for line in fp:
            vals = line.strip().split(',')
            words = vals[2].split(' ')
            docs.append(words)
    return docs
def word_dict(docs,english_stopwords):
    d = {}
    for doc in docs:
        for word in doc:
            if word in english_stopwords:
                continue
            d[word] = d.get(word,0)+1
    return d
def show(d):
    vals = sorted(list(d.values()),reverse=True)
    print('total words:',len(d))
    threhold = vals[int(0.8*len(d))]
    #vals = (np.array(vals,dtype=np.float32)-np.min(vals))/(np.max(vals) - np.min(vals))
    plt.plot(list(range(len(vals))),np.log(vals))
    plt.show()
    essential_words = {}
    for word in d:
        if d.get(word) > threhold:
            essential_words[word] = d.get(word)
    ewords = sorted(essential_words.items(),key=lambda x: x[1],reverse=True)
    print('essential words:',len(ewords))
    print(ewords[:10])

def main():
    prefix = './data/baby/baby'
    docs = load_doc(prefix)
    english_stopwords = stopwords.words('english')
    d = word_dict(docs,english_stopwords)
    show(d)
if __name__ == '__main__':
    main()
