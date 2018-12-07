from collections import Counter
from nltk.corpus import stopwords
import pickle
import numpy as np
import argparse
import re
def parse_args():
    parser = argparse.ArgumentParser(description='run dataset preprocessing')
    parser.add_argument('--dataset',type=str,default='baby',help='specify the dataset')
    return parser.parse_args()
class Cropus:
    def __init__(self,prefix):
        self.prefix = prefix
        self.stopwords = stopwords.words('english')
        self.build_up()
    def clean_str(self,string):
        string = re.sub(r"[^A-Za-z]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.lower()
    def build_up(self):
        self.num_users = 0
        self.num_items = 0
        self.train_data = []
        self.test_data = []
        utext = ''
        docs = []
        with open(self.prefix+'_train.csv','r') as fp:
            for line in fp:
                vals = line.strip().split(',')
                u,i,text,r = int(vals[0]),int(vals[1]),vals[2],float(vals[3])
                self.num_users = max(u,self.num_users)
                self.num_items = max(i,self.num_items)
                text = self.clean_str(text.strip())
                utext += text
                docs.append(text)
                self.train_data.append([u,i,r])
        with open(self.prefix+'_test.csv','r') as fp:
            for line in fp:
                vals = line.strip().split(',')
                u,i,r = int(vals[0]),int(vals[1]),float(vals[3])
                self.num_items = max(i,self.num_items)
                self.num_users = max(u,self.num_users)
                self.test_data.append([u,i,r])
        self.vocab = self.get_vocab(utext)
        self.docs = []
        ulens = []
        for doc in docs:
            doc = [self.vocab[w] for w in doc.split() if w in self.uvocab]
            ulens.append(len(doc))
            self.docs.append(doc)
        ulens = sorted(ulens)
        self.length = max(ulens)
        self.padding_text()
    def padding_text(self,padding='<PAD/>'):
        self.masks = []
        for i in range(len(self.docs)):
            doc = self.docs[i]
            if len(doc) > self.length:
                doc = doc[:self.length]
                self.masks.append(self.length)
            else:
                self.masks.append(len(doc))
                doc = doc + [self.vocab[padding]]*(self.length - len(doc))
            self.docs[i] = doc
    def get_vocab(self,text):
        word_count = Counter(text.split())
        words = word_count.most_common()
        threshold = words[int(0.8*len(words))][1]
        words = [w for w,f in words if f>threshold and w not in self.stopwords]
        if '<PAD/>' not in words:
            words.append('<PAD/>')
        np.random.shuffle(words)
        vocab = {w:i for i,w in enumerate(words)}
        return vocab
    def save(self):
        data = []
        for idx in range(len(self.train_data)):
            u,i,r = self.train_data[idx]
            data.append([u,i,r,self.docs[idx],self.masks[idx]])
        with open(self.prefix+'_single_train.pkl','wb') as fw:
            pickle.dump(data,fw)
        with open(self.prefix+'_single_test.pkl','wb') as fw:
            pickle.dump(self.test_data,fw)
        param = {'doc_length':self.length,'vocab_size':len(self.vocab),'num_users':self.num_users,'num_items':self.num_items,'train_length':len(self.train_data),'test_length':len(self.test_data)}
        print(param)
        with open(self.prefix+'_single_param.pkl','wb') as fw:
            pickle.dump(param,fw)
def main():
    args = parse_args()
    prefix = './data/{}/{}'.format(args.dataset,args.dataset)
    ds = Cropus(prefix)
    ds.save()
if __name__ == '__main__':
    main()
