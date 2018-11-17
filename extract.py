import json
import argparse
import re
import os

def parse_args():
    parser = argparse.ArgumentParser(description='run extract script')
    parser.add_argument('--dataset',default='Baby',help='specify the dataset')
    return parser.parse_args()
def clean_text(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
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
    return string.strip().lower()

def extract(file,prefix):
    field = {'user':'reviewerID','item':'asin','review':'reviewText','rating':'overall','timestamp':'unixReviewTime'}
    users = {}
    items = {}
    with open(file,'r') as fp:
        for line in fp:
            line = line.strip()
            vals = json.loads(line)
            users[vals[field['user']]] = users.get(vals[field['user']],0)+1
            items[vals[field['item']]] = items.get(vals[field['item']],0)+1
    user2idx = {u:idx+1 for idx,u in enumerate(users)}
    item2idx = {it:idx+1 for idx,it in enumerate(items)}
    fw = open(prefix+'.csv','w')
    with open(file,'r') as fp:
        for line in fp:
            line = line.strip()
            vals = json.loads(line)
            user,item,review,rating,ts = user2idx[vals[field['user']]],item2idx[vals[field['item']]],vals[field['review']],vals[field['rating']],vals[field['timestamp']]
            review = clean_text(review)
            nline = '{}_|_{}_|_{}_|_{}_|_{}\n'.format(user,item,review,rating,ts)
            fw.write(nline)
    fw.close()
    with open(prefix+'_umap.csv','w') as fp:
        for u in user2idx:
            line = '{},{},{}\n'.format(user2idx[u],u,users[u])
            fp.write(line)
    with open(prefix+'_itmap.csv','w') as fp:
        for it in item2idx:
            line = '{},{},{}\n'.format(item2idx[it],it,items[it])
            fp.write(line)
def main():
    args = parse_args()
    ds = args.dataset.lower()
    if not os.path.exists('./data/{}/'.format(ds)):
	    os.makedirs('./data/{}/'.format(ds))
    prefix = './data/{}/{}'.format(ds,ds)
    file = './reviews_{}_5.json'.format(args.dataset)
    extract(file,prefix)
if __name__ == '__main__':
    main()
