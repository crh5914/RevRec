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
from tensorflow.contrib import learn
import datetime

import pickle
import DeepCoNN
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='run model')
    parser.add_argument('--dataset',type=str,default='baby',help='specify the dataset')
    parser.add_argument('--embedding_dim', type=int, default=300, help="Dimensionality of character embedding")
    parser.add_argument('--filter_sizes', type=str, default='[3]', help='filter size')
    parser.add_argument("--num_filters", type=int,default=100,help="Number of filters per filter size")
    parser.add_argument("--dropout_keep_prob",type=float,default=0.5,help="Dropout keep probability ")
    parser.add_argumentt("--l2_reg_lambda", type=float,default=0.0, help="L2 regularizaion lambda")
    parser.add_argument("--l2_reg_V",type=float,default=0.0,help="L2 regularizaion V")
    # Training parameters
    parser.add_argumentr("--batch_size",type=int,default=100,help="Batch Size ")
    parser.add_argument("--num_epochs", type=int,default=40, help="Number of training epochs ")
    parser.add_argument("--evaluate_every",type=int,default=100,help="Evaluate model on dev set after this many steps ")
    parser.add_argument("--checkpoint_every",type=int,default=100, help="Save model after this many steps ")
    # Misc Parameters
    parser.add_argument("--allow_soft_placement",type=bool, default=True, help="Allow device soft device placement")
    parser.add_argument("--log_device_placement",type=bool,default=False,help="Log placement of ops on devices")
    return parser.parse_args()

def train_step(u_batch, i_batch, uid, iid, y_batch):
    """
    A single training step
    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, loss, accuracy, mae = sess.run(
        [train_op, global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()

    # print("{}: step {}, loss {:g}, rmse {:g},mae {:g}".format(time_str, batch_num, loss, accuracy, mae))
    return accuracy, mae


def dev_step(u_batch, i_batch, uid, iid, y_batch, writer=None):
    """
    Evaluates model on a dev set

    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.dropout_keep_prob: 1.0
    }
    step, loss, accuracy, mae = sess.run(
        [global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step{}, loss {:g}, rmse {:g},mae {:g}".format(time_str, step, loss, accuracy, mae))

    return [loss, accuracy, mae]
def generate_batch(train_data,batch_size):
    train_data_size = len(train_data)
    shuffle_indices = np.random.permutation(np.range(train_data_size))
    ll = int(len(train_data) / batch_size)
    for batch_num in range(ll):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, train_data_size)
        uid, iid, y_batch, u_batch, i_batch = [],[],[],[],[]
        for idx in range(start_index,end_index):
            u,i,r,utext,ittext = train_data[idx]
            uid.append([u])
            iid.append([i])
            y_batch.append([r])
            u_batch.append(utext)
            i_batch.append(ittext)
        yield uid,iid,y_batch,u_batch,i_batch
if __name__ == '__main__':
    args = parse_args()
    print('parameters:',args)
    print("Loading data...")
    prefix = './data/{}/{}'.format(args.dataset,args.dataset)
    with open(prefix+'_aggregated_param.pkl','rb') as fp:
        param = pickle.load(fp)
    user_num = para['num_users']
    item_num = para['num_items']
    user_length = para['user_length']
    item_length = para['item_length']
    uvocab_size = para['uvocab_size']
    itvocab_size = para['itvocab_size']
    train_length = param['train_length']
    test_length = param['test_length']
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            deep = DeepCoNN.DeepCoNN(
                user_num=user_num,
                item_num=item_num,
                user_length=user_length,
                item_length=item_length,
                num_classes=1,
                user_vocab_size=uvocab_size,
                item_vocab_size=itvocab_size,
                embedding_size=args.embedding_dim,
                fm_k=8,
                filter_sizes=eval(args.filter_sizes),
                num_filters=args.num_filters,
                l2_reg_lambda=args.l2_reg_lambda,
                l2_reg_V=args.l2_reg_V,
                n_latent=32)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            # optimizer = tf.train.AdagradOptimizer(learning_rate=0.1, initial_accumulator_value=1e-8).minimize(deep.loss)
            optimizer = tf.train.AdamOptimizer(0.002, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(deep.loss)
            '''optimizer=tf.train.RMSPropOptimizer(0.002)
            grads_and_vars = optimizer.compute_gradients(deep.loss)'''
            train_op = optimizer  # .apply_gradients(grads_and_vars, global_step=global_step)
            sess.run(tf.global_variables_initializer())
            best_mae = 5
            best_rmse = 5
            train_mae = 0
            train_rmse = 0
            with open(prefix+'_aggregated_train.pkl','rb') as fp:
                train_data = pickle.load(fp)

            with open(prefix+'_aggregated_test.pkl','rb') as fp:
                test_data = pickle.load(fp)
            batch_size = args.batch_size
            for epoch in range(40):
                # Shuffle the data at each epoch
                batch_num = 0
                for uid,iid,y_batch,u_batch,i_batch in generate_batch(train_data,batch_size):
                    batch_num += 1
                    t_rmse, t_mae = train_step(u_batch, i_batch, uid, iid, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    train_rmse += np.square(t_rmse)*len(uid)
                    train_mae += t_mae*len(iid)
                    if batch_num % 1000 == 0 and batch_num > 1:
                        loss_s = 0
                        accuracy_s = 0
                        mae_s = 0
                        for uid,iid,y_batch,u_batch,i_batch in generate_batch(test_data,batch_size):
                            loss, accuracy, mae = dev_step(u_batch, i_batch,uid, iid, y_batch)
                            loss_s = loss_s + len(uid) * loss
                            accuracy_s = accuracy_s + len(uid) * np.square(accuracy)
                            mae_s = mae_s + len(uid) * mae
                        print ("epoch:{},in training evaluation,loss:{:g}, rmse:{:g}, mae:{:g}".format(epoch,loss_s / data_size_test,np.sqrt(accuracy_s / data_size_test),mae_s / data_size_test))
                print("epoch:{},training,rmse:{},mae:{}".format(epoch,train_rmse/train_length, train_mae / train_length))
                train_rmse = 0
                train_mae = 0
                loss_s = 0
                accuracy_s = 0
                mae_s = 0
                for uid, iid, y_batch, u_batch, i_batch in generate_batch(test_data, batch_size):
                    loss, accuracy, mae = dev_step(u_batch, i_batch, uid, iid, y_batch)
                    loss_s = loss_s + len(uid) * loss
                    accuracy_s = accuracy_s + len(uid) * np.square(accuracy)
                    mae_s = mae_s + len(uid) * mae
                print ("test,loss {:g}, rmse:{:g}, mae:{:g}".format(loss_s / test_length,np.sqrt(accuracy_s / test_length),mae_s / test_length))
                rmse = np.sqrt(accuracy_s / test_length)
                mae = mae_s / test_length
                if best_rmse > rmse:
                    best_rmse = rmse
                if best_mae > mae:
                    best_mae = mae
                print("")
            print('best rmse:', best_rmse)
            print('best mae:', best_mae)

    print('end')
