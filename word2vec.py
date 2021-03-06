import argparse
from gensim.models import Word2Vec
import pickle
def parse_args():
    parser = argparse.ArgumentParser(description='run extract script')
    parser.add_argument('--dataset',type=str,default='baby',help='specify the dataset')
    parser.add_argument('--window',type=int,default=10,help='window size')
    parser.add_argument('--dim',type=int,default=100,help='embedding dimemsionality')
    parser.add_argument('--alpha',type=float,default=0.005,help='learning rate')
    parser.add_argument('--itr',type=int,default=100,help='iterations')
    return parser.parse_args()
def load_docs(prefix):
    docs = []
    file = prefix+'_train.csv'
    with open(file,'r') as fp:
        for line in fp:
            vals = line.strip().split(',')
            doc = vals[2].strip().split(' ')
            docs.append(doc)
    return docs
def main():
    args = parse_args()
    prefix = './data/{}/{}'.format(args.dataset,args.dataset)
    docs = load_docs(prefix)
    model = Word2Vec(docs,iter=args.itr,size=args.dim,window=args.window,alpha=args.alpha,min_count=1,workers=4)
    file = prefix+'_w2v.emb'
    with open(file,'wb') as fp:
        pickle.dump(model,fp)
    #print(model.wv.vocab)
if __name__ == '__main__':
    main()
