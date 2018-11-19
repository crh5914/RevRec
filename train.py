import tensorflow as tf
from gensim.models import Word2Vec
from util import *
from simple_model import SimpleModel
from tqdm import tqdm
import pickle
def main():
    prefix = './data/{}/{}'.format('baby','baby')
    dim = 100
    epochs = 50
    layer_size = [64,32,16,1]
    lr = 0.0001
    alpha = 0.5
    num_users,num_items = get_user_item_info(prefix+'_umap.csv',prefix+'_itmap.csv')
    train_users,train_items,train_revs,train_rs,word_dict,maxlen = load_train_data(prefix+'_train.csv')
    test_users,test_items,test_rs = load_test_data(prefix+'_test.csv')
    vocab = get_vocab(word_dict)
    #w2v = Word2Vec.load(prefix+'_w2v.emb')
    with open(prefix+'_w2v.emb','rb') as fp:
        w2v = pickle.load(fp)
    word_emb = get_word_emb(vocab,dim,w2v)
    sess = tf.Session()
    model = SimpleModel(sess,num_users,num_items,maxlen,len(vocab),dim,layer_size,lr,alpha)
    sess.run(model.word_emb.assign(tf.constant(word_emb,dtype=tf.float32)))
    best_mse,best_epoch = 10,0
    for epoch in range(epochs):
        loss,mse = 0,0
        for batch_u,batch_i,batch_rev,batch_len,batch_r in tqdm(generate_train_batch(train_users,train_items,train_revs,train_rs,vocab,maxlen)):
            batch_mse,batch_loss = model.train(batch_u,batch_i,batch_rev,batch_len,batch_r)
            loss += batch_loss*len(batch_u)
            mse += batch_mse*len(batch_u)
        mse = mse / len(train_users)
        loss = loss / len(train_users)
        print('train epoch:{},mse:{},loss:{}'.format(epoch+1,mse,loss))
        mse = 0
        for batch_u,batch_i,batch_r in tqdm(generate_test_batch(test_users,test_items,test_rs)):
            y,batch_mse = model.test(batch_u,batch_i,batch_r)
            mse += len(batch_u) * batch_mse
        mse = mse / len(test_users)
        if mse < best_mse:
            best_mse = mse
            best_epoch = epoch + 1
        print('test epoch:{},mse:{}'.format(epoch+1,mse))
    print('best mse:{},at epoch:{}'.format(best_mse,best_epoch))
if __name__ == '__main__':
    main()
