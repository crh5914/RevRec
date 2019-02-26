import sys
import pickle
def to_int(ll):
    return list(map(int,ll))
def load_rating(file):
    data = []
    with open(file,'r') as fp:
        for line in fp:
            uid,iid,rating = line.strip().split('\t')[:3]
            uid,iid,rating = int(uid),int(iid),float(rating)
            data.append([uid,iid,rating])
    return data
def load_review(file):
    ureview,itreview = {},{}
    with open(file,'r') as fp:
        for line in fp:
            uid,iid,rating,ur,ir = line.strip().split('\t')[3:]
            uid,iid = int(uid),int(iid)
            ur,ir = to_int(ur),to_int(ir)
            if uid not in ureview:
                 ureview[uid] = []
            if iid not in itreview:
                 itreview[iid] = []
            ureview.extend(ur)
            itreview.extend(ir)
    user_length,item_length = 0,0
    for u in ureview:
        user_length = max(user_length,len(ureview[u]))
    for i in itreview:
        item_length = max(item_length,len(itreview[i]))
    return ureview,itreview,user_length,item_length
if len(sys.argv) >= 2:
    ds = sys.argv[1]
    path = '../data/{}'.format(ds)
    train_file = '{}/train_int_{}.txt'.format(path,ds)
    valid_file = '{}/valid_int_{}.txt'.format(path,da)
    test_file = '{}/test_int_{}.txt'.format(path,ds)
    train_data = load_rating(train_file)
    valid_data = load_rating(valid_file)
    test_data = load_rating(test_file)
    with open('{}/{}_train_rating.pkl'.format(path,ds),'wb') as fp:
        pickle.dump(train_data,fp)
    train_length = len(train_data)
    del train_data
    valid_data = load_rating(valid_file)
    with open('{}/{}_valid_rating.pkl'.format(path,ds),'wb') as fp:
        pickle.dump(valid_data,fp)
    valid_length = len(valid_data)
    del valid_data
    test_data = load_rating(test_file)
    with open('{}/{}_test_rating.pkl'.format(path,ds),'wb') as fp:
        pickle.dump(test_data,fp)
    test_length = len(test_data)
    del test_data
    ureview,itreview,user_length,item_length = load_review(train_file)
    with open('{}/{}_ureview.pkl'.format(path,ds),'wb') as fp:
        pickle.dump(ureview,fp)
    del ureview
    with open('{}/{}_itreview.pkl'.format(path,ds),'wb') as fp:
        pickle.dump(itreview,fp)
    del itreview
    with open('{}/{}_umap.csv','r') as fp:
        num_users = len(fp.readlines())
    with open('{}/{}_itmap.csv','r') as fp:
        num_items = len(fp.readlines())
    with open('{}/{}_emb.pkl','rb') as fp:
        emb = pickle.load(fp)
        uvocab_size = itvocab_size = len(emb)
    param = {}
    param['num_users'] = num_users
    param['num_items'] = num_items
    param['user_length'] = user_length
    param['item_length'] = item_length
    param['uvocab_size'] = uvocab_size
    param['itvocab_size'] = itvocab_size
    param['train_length'] = train_length
    param['valid_length'] = valid_length
    param['test_length'] = test_length
    with open('{}/{}_param.pkl','wb') as fp:
        pickle.dump(param,fp)
else:
    sys.exit(-1)
