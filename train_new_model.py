'''
model_train
@author:
Chong Chen (cstchenc@163.com)

@ created:
27/8/2017
@references:
Lei Zheng, Vahid Noroozi, and Philip S Yu. 2017. Joint deep modeling of users and items using reviews for recommendation.
In WSDM. ACM, 425-434.
'''

import numpy as np
import tensorflow as tf
import math
import pickle
from new_model import TransMemoryCNN
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='run model')
    parser.add_argument('--dataset',type=str,default='baby',help='specify the dataset')
    parser.add_argument('--embedding_size', type=int, default=100, help="Dimensionality of character embedding")
    parser.add_argument('--dim', type=int, default=100, help="factor dim of memory network")
    parser.add_argument('--num_mem', type=int, default=64, help="number of memory slots")
    parser.add_argument('--filter_sizes', type=str, default='[3]', help='filter size')
    parser.add_argument('--layer_size', type=str, default='[64,32]', help='full connected layer size')
    parser.add_argument("--num_filters", type=int,default=100,help="Number of filters per filter size")
    parser.add_argument("--dropout_keep_prob",type=float,default=0.5,help="Dropout keep probability ")
    parser.add_argument("--l2_reg_lambda", type=float,default=0.0, help="L2 regularizaion lambda")
    parser.add_argument("--lr", type=float,default=0.001, help="learning rate")
    parser.add_argument("--alpha", type=float,default=0.9, help="loss balance alpha")
    #parser.add_argument("--l2_reg_V",type=float,default=0.0,help="L2 regularizaion V")
    # Training parameters
    parser.add_argument("--batch_size",type=int,default=128,help="Batch Size ")
    parser.add_argument("--num_epochs", type=int,default=40, help="Number of training epochs ")
    #parser.add_argument("--evaluate_every",type=int,default=100,help="Evaluate model on dev set after this many steps ")
    #parser.add_argument("--checkpoint_every",type=int,default=100, help="Save model after this many steps ")
    # Misc Parameters
    #parser.add_argument("--allow_soft_placement",type=bool, default=True, help="Allow device soft device placement")
    #parser.add_argument("--log_device_placement",type=bool,default=False,help="Log placement of ops on devices")
    return parser.parse_args()

def train_step(model,batch_users,batch_items,batch_docs,batch_masks,batch_ratings,args):
    """
    A single training step
    """
    feed_dict = {
        model.user: batch_users,
        model.item: batch_items,
        model.doc: batch_docs,
        model.mask: batch_masks,
        model.rating: batch_ratings,
        model.dropout_keep_prob: args.dropout_keep_prob
    }
    _,loss,y = sess.run(
        [model.train_op,model.loss,model.y],
        feed_dict)
    err = np.array(batch_rating) - np.array(y)
    mse = np.mean(np.square(err))
    mae = np.mean(np.abs(err))
    return loss,mse,mae


def dev_step(model,batch_users,batch_items,batch_ratings):
    """
    Evaluates model on a dev set

    """
    feed_dict = {
        model.user: batch_users,
        model.item: batch_items,
        model.rating: batch_ratings,
        model.dropout_keep_prob: 1.0
    }
    y_ = sess.run(model.y_,feed_dict)
    err = np.array(batch_ratings) - np.array(y_)
    mse = np.mean(np.square(err))
    mae = np.mean(np.abs(err))
    return mse,mae
def generate_train_batch(data,batch_size):
    data_size = len(data)
    shuffle_indices = np.random.permutation(range(data_size))
    ll = int(len(data) / batch_size)
    for batch_num in range(ll):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size,data_size)
        batch_users,batch_items,batch_docs,batch_masks,batch_ratings= [],[],[],[],[]
        for idx in range(start_index,end_index):
            u,i,r,doc,mask = data[idx]
            batch_users.append(u)
            batch_items.append(i)
            batch_ratings.append(r)
            batch_docs.append(doc)
            batch_masks.append(mask)
        yield batch_users,batch_items,batch_docs,batch_masks,batch_rating
def generate_test_batch(data,batch_size):
    data_size = len(data)
    shuffle_indices = np.random.permutation(range(data_size))
    ll = int(len(data) / batch_size)
    for batch_num in range(ll):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size,data_size)
        batch_users,batch_items,batch_ratings= [],[],[]
        for idx in range(start_index,end_index):
            u,i,r = data[idx]
            batch_users.append(u)
            batch_items.append(i)
            batch_ratings.append(r)
        yield batch_users,batch_items,batch_rating
if __name__ == '__main__':
    args = parse_args()
    print('parameters:',args)
    print("Loading data...")
    prefix = './data/{}/{}'.format(args.dataset,args.dataset)
    with open(prefix+'_single_param.pkl','rb') as fp:
        para = pickle.load(fp)
    num_users = para['num_users']
    num_items = para['num_items']
    doc_length = para['doc_length']
    vocab_size = para['vocab_size']
    train_length = para['train_length']
    test_length = para['test_length']
    print('vocab size:{},doc length:{}'.format(vocab_size,doc_length))
    sess = tf.Session()
    model = TransMemoryCNN(sess,num_users,num_items,args.num_mem,args.embedding_size,args.dim,eval(args.filter_sizes),args.num_filters,vocab_size,doc_length,eval(args.layer_size),args.lr,args.l2_reg_lambda,args.alpha)
    best_mae = 5
    best_mse = 5
    train_mae = 0
    train_rmse = 0
    with open(prefix+'_single_train.pkl','rb') as fp:
        train_data = pickle.load(fp)
    with open(prefix+'_single_test.pkl','rb') as fp:
        test_data = pickle.load(fp)
    print('load data done..')
    batch_size = args.batch_size
    for epoch in range(args.num_epochs):
        # Shuffle the data at each epoch
        batch_num = 0
        train_loss = 0
        for batch_users,batch_items,batch_docs,batch_masks,batch_rating in generate_train_batch(train_data,batch_size):
            batch_num += 1
            loss,t_mse, t_mae = train_step(batch_users,batch_items,batch_docs,batch_masks,batch_rating,args)     
            train_mse += t_mse*len(batch_users)
            train_mae += t_mae*len(batch_users)
            train_loss += loss*len(batch_users)
            if batch_num % 300 == 0 and batch_num > 1:
                 loss_s = 0
                 mse_s = 0
                 mae_s = 0
                 for batch_users,batch_items,batch_ratings in generate_test_batch(test_data,batch_size):
                     mse, mae = dev_step(batch_users,batch_items,batch_ratings)
                     mse_s += len(batch_users) * mse
                     mae_s = mae_s + len(batch_users) * mae
                     print ("epoch:{},in training evaluation,rmse:{:g}, mae:{:g}".format(epoch,np.sqrt(mse_s / test_length),mae_s / test_length))
        print("epoch:{},training,loss:{},rmse:{},mae:{}".format(epoch,train_loss/train_length,train_rmse/train_length, train_mae / train_length))
        train_rmse = 0
        train_mae = 0
        mse_s = 0
        mae_s = 0
        for batch_users,batch_items,batch_ratings in generate_test_batch(test_data,batch_size):
              mse, mae = dev_step(batch_users,batch_items,batch_ratings)
              mse_s += len(batch_users) * mse
              mae_s = mae_s + len(batch_users) * mae
        print ("epoch:{},test,rmse:{:g}, mae:{:g}".format(epoch,np.sqrt(mse_s / test_length),mae_s / test_length))
        if mse_s/test_length < best_mse:
            best_mse = mse
        if mae_s/test_length < mae_s
            best_mae = mae
    print('best rmse:', np.sqrt(best_mse))
    print('best mae:', best_mae)
    print('end')
