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
        words  = [word for word in string.strip().lower().split() if word not in self.stopwords]
        return string.strip().lower()
    def build_up(self):
        self.user_text = {}
        self.item_text = {}
        self.num_users = 0
        self.num_items = 0
        self.train_data = []
        self.test_data = []
        with open(self.prefix+'_train.csv','r') as fp:
            for line in fp:
                vals = line.strip().split(',')
                u,i,text,r = int(vals[0]),int(vals[1]),vals[2],float(vals[3])
                self.num_users = max(u,self.num_users)
                self.num_items = max(i,self.num_items)
                if u not in self.user_text:
                    self.user_text[u] = '<PAD/>'
                else:
                    self.user_text[u] = self.user_text[u] + ' ' + text.strip()
                if i not in self.item_text:
                    self.item_text[i] = '<PAD/>'
                else:
                    self.item_text[i] = self.item_text[i] + ' ' + text.strip()
                self.train_data.append([u,i,r])
        with open(self.prefix+'_test.csv','r') as fp:
            for line in fp:
                vals = line.strip().split(',')
                u,i,r = int(vals[0]),int(vals[1]),float(vals[3])
                self.num_items = max(i,self.num_items)
                self.num_users = max(u,self.num_users)
                self.test_data.append([u,i,r])
        self.ulen = 0
        self.itlen = 0
        utext = []
        ittext = []
        for u in self.user_text:
            self.user_text[u] = self.clean_str(self.user_text[u])
            utext += self.user_text[u]
        self.uvocab = self.get_vocab(utext)
        for it in self.item_text:
            self.item_text[it] = self.clean_str(self.item_text[it])
            ittext += self.item_text[it]
        self.itvocab = self.get_vocab(ittext)
        for u in self.user_text:
            self.user_text[u] = [self.uvocab[w] for w in self.user_text[u] if w in self.uvocab]
            self.ulen = max(self.ulen,len(self.user_text[u]))
        for it in self.item_text:
            self.item_text[it] = [self.itvocab[w] for w in self.item_text[it] if w in self.itvocab]
            self.itlen = max(self.itlen,len(self.item_text[it]))
        self.padding_text()
    def padding_text(self,padding='<PAD/>'):
        for u in self.user_text:
            self.user_text[u] = self.user_text[u] + [self.uvocab[padding]]*(self.ulen - len(self.user_text[u]))
        for it in self.item_text:
            self.item_text[it] = self.item_text[it] + [self.itvocab[padding]]*(self.itlen - len(self.item_text[u]))
    def get_vocab(self,text):
        word_count = Counter(text)
        words = word_count.most_common()
        threshold = words[int(0.8*len(words))][1]
        words = [w for w,f in words if f>threshold]
        if '<PAD/>' not in words:
            words.append('<PAD/>')
        np.random.shuffle(words)
        vocab = {w:i for i,w in enumerate(words)}
        return vocab
    def save(self):
        data = []
        for u, i, r in self.train_data:
            data.append([u, i, r, self.user_text[u], self.item_text[i]])
        with open(self.prefix+'_aggregated_train.pkl','wb') as fw:
            pickle.dump(data,fw)
        data = []
        for u, i, r in self.test_data:
            data.append([u, i, r, self.user_text[u], self.item_text[i]])
        with open(self.prefix+'_aggregated_test.pkl','wb') as fw:
            pickle.dump(data,fw)
        param = {'user_length':self.ulen,'item_length':self.itlen,'uvocab_size':len(self.uvocab),'itvocab_size':len(self.itvocab),'num_users':self.num_users,'num_items':self.num_items,'train_length':len(self.train_data),'test_length':len(self.test_data)}
        with open(self.prefix+'_aggregated_param.pkl','wb') as fw:
            pickle.dump(param,fw)

def main():
    args = parse_args()
    prefix = './data/{}/{}'.format(args.dataset,args.dataset)
    ds = Cropus(prefix)
    ds.save()
if __name__ == '__main__':
    main()
