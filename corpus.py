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
        #words  = [word for word in string.strip().lower().split() if word not in self.stopwords]
        return string.lower()
    def build_up(self):
        self.user_text = {}
        self.item_text = {}
        self.num_users = 0
        self.num_items = 0
        self.train_data = []
        self.test_data = []
        utext = ''
        with open(self.prefix+'_train.csv','r') as fp:
            for line in fp:
                vals = line.strip().split(',')
                u,i,text,r = int(vals[0]),int(vals[1]),vals[2],float(vals[3])
                self.num_users = max(u,self.num_users)
                self.num_items = max(i,self.num_items)
                text = self.clean_str(text.strip())
                utext += text
                if u not in self.user_text:
                    self.user_text[u] = '<PAD/>'
                else:
                    self.user_text[u] = self.user_text[u] + ' ' + text
                if i not in self.item_text:
                    self.item_text[i] = '<PAD/>'
                else:
                    self.item_text[i] = self.item_text[i] + ' ' + text
                self.train_data.append([u,i,r])
        with open(self.prefix+'_test.csv','r') as fp:
            for line in fp:
                vals = line.strip().split(',')
                u,i,r = int(vals[0]),int(vals[1]),float(vals[3])
                self.num_items = max(i,self.num_items)
                self.num_users = max(u,self.num_users)
                self.test_data.append([u,i,r])
        ulens = []
        itlens = []
        self.uvocab = self.get_vocab(utext)
        self.itvocab = self.uvocab
        for u in self.user_text:
            self.user_text[u] = [self.uvocab[w] for w in self.user_text[u].split() if w in self.uvocab]
            ulens.append(len(self.user_text[u]))
        for it in self.item_text:
            self.item_text[it] = [self.itvocab[w] for w in self.item_text[it].split() if w in self.itvocab]
            itlens.append(len(self.item_text[it]))
        ulens = sorted(ulens)
        itlens = sorted(itlens)
        self.ulen = ulens[int(0.85*len(ulens))-1]
        self.itlen = itlens[int(0.85*len(itlens))-1]
        self.padding_text()
    def padding_text(self,padding='<PAD/>'):
        for u in self.user_text:
            if len(self.user_text[u]) > self.ulen:
                self.user_text[u] = self.user_text[u][:self.ulen]
            else:
                self.user_text[u] = self.user_text[u] + [self.uvocab[padding]]*(self.ulen - len(self.user_text[u]))
        for it in self.item_text:
            if len(self.item_text[it]) > self.itlen:
                self.item_text[it] = self.item_text[it][:self.itlen]
            else:
                self.item_text[it] = self.item_text[it] + [self.itvocab[padding]]*(self.itlen - len(self.item_text[it]))
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
        with open(self.prefix+'_aggregated_train.pkl','wb') as fw:
            pickle.dump(self.train_data,fw)
        with open(self.prefix+'_aggregated_ureview.pkl','wb') as fw:
            pickle.dump(self.user_text,fw)
        with open(self.prefix+'_aggregated_test.pkl','wb') as fw:
            pickle.dump(self.test_data,fw)
        with open(self.prefix+'_aggregated_itreview.pkl','wb') as fw:
            pickle.dump(self.item_text,fw)
        param = {'user_length':self.ulen,'item_length':self.itlen,'uvocab_size':len(self.uvocab),'itvocab_size':len(self.itvocab),'num_users':self.num_users,'num_items':self.num_items,'train_length':len(self.train_data),'test_length':len(self.test_data)}
        print(param)
        with open(self.prefix+'_aggregated_param.pkl','wb') as fw:
            pickle.dump(param,fw)
def main():
    args = parse_args()
    prefix = './data/{}/{}'.format(args.dataset,args.dataset)
    ds = Cropus(prefix)
    ds.save()
if __name__ == '__main__':
    main()
