import numpy as np
from sklearn.utils import shuffle
def get_word_emb(vocab,dim,model):
    emb_matrix = np.zeros(shape=(len(vocab),dim))
    for word in vocab:
        if word in model:
            emb_matrix[vocab[word]-1] = model[word] 
    return emb_matrix

def load_train_data(file):
    users,items,revs,rs = [],[],[],[]
    vocab = {}
    maxlen = 0
    with open(file,'r') as fp:
         for line in fp:
             line = line.strip()
             vals = line.split(',')
             users.append(int(vals[0])-1)
             items.append(int(vals[1])-1)
             words = vals[2].split()
             revs.append(words)
             if len(words) > maxlen:
                 maxlen = len(words)
             for word in words:
                 vocab[word] = vocab.get(word,0) + 1
             rs.append(float(vals[3]))
    return users,items,revs,rs,vocab,maxlen
def load_test_data(file):
    users,items,rs = [],[],[]
    with open(file,'r') as fp:
         for line in fp:
             line = line.strip()
             vals = line.split(',')
             users.append(int(vals[0])-1)
             items.append(int(vals[1])-1)
             rs.append(float(vals[3]))
    return users,items,rs

def generate_test_batch(users,items,rs,batch_size=256):
    batch_u,batch_i,batch_r = [],[],[]
    for u,i,r in zip(users,items,rs):
        batch_u.append(u)
        batch_i.append(i)
        batch_r.append(r)
        if len(batch_u) == batch_size:
            yield batch_u,batch_i,batch_r
            batch_u,batch_i,batch_r = [],[],[]
    if len(batch_u) > 0:
        yield batch_u,batch_i,batch_r

def get_user_item_info(ufile,ifile):
    with open(ufile,'r') as fp:
        num_users = len(fp.readlines())
    with open(ifile,'r') as fp:
        num_items = len(fp.readlines())
    return num_users,num_items

def get_vocab(words_count):
    vocab = {word:idx+1 for idx,word in enumerate(words_count)}
    return vocab

def generate_train_batch(users,items,revs,rs,vocab,maxlen,batch_size=256):
    users,items,revs,rs = shuffle(users,items,revs,rs)
    batch_u,batch_i,batch_rev,batch_len,batch_r = [],[],[],[],[]
    for u,i,rev,r in zip(users,items,revs,rs):
        batch_u.append(u)
        batch_i.append(i)
        rev = [ vocab[word] for word in rev]
        batch_len.append([len(rev)])
        rev = rev + [0]*(maxlen-len(rev)) 
        batch_rev.append(rev)
        batch_r.append(r)
        if len(batch_u) == batch_size:
            yield batch_u,batch_i,batch_rev,batch_len,batch_r
            batch_u,batch_i,batch_rev,batch_len,batch_r = [],[],[],[],[]
    if len(batch_u) > 0:
        yield batch_u,batch_i,batch_rev,batch_len,batch_r
        
