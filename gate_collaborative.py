import tensorflow as tf
import numpy as np
import pandas as pd


class GateCollaborativeRecommender:
    def __init__(self,sess,user_num,item_num,factor_num,vocab_size,review_length,filter_size,filter_num,lr):
        self.sess = sess
        self.user_num = user_num
        self.item_num = item_num
        self.factor_num = factor_num
        self.vocab_size = vocab_size
        self.review_length = review_length
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.lr = lr
        self.build_model()
    def build_model(self):
        self.user = tf.placeholder(shape=[None,],dtype=tf.int32)
        self.item = tf.placeholder(shape=[None,],dtype=tf.int32)
        self.text = tf.placeholder(shape=[None,self.review_length],dtype=tf.int32)
        self.rating = tf.placeholder(shape=[None,],dtype=tf.float32)
        self.phrase = tf.placeholder(False,dtype=tf.bool)
        with tf.name_scope('embedding/word_embedding'):
            word_embedding = tf.Variable(tf.random.uniform(shape=[self.vocab_size,self.factor_num],minval=-0.1,maxval=0.1))
        context = tf.nn.embedding_lookup(word_embedding,self.text)

        with tf.name_scope('embedding/user_embedding'):
            user_embedding = tf.Variable(tf.random_uniform(shape=[self.user_num,self.factor_num],minval=-0.1,maxval=0.1))
        uvec = tf.nn.embedding_lookup(user_embedding,self.user)

        with tf.name_scope('embedding/item_embedding'):
            item_embedding = tf.Variable(tf.random_uniform(shape=[self.user_num,self.factor_num],minval=-0.1,maxval=0.1))
        ivec = tf.nn.embedding_lookup(item_embedding,self.item)
        # convoluntional layers
        context = tf.expand_dims(context,axis=-1) # None*review_length*factor_num*1
        pools = []
        for size in self.filter_size:
            filter_kernal = [size,self.factor_num,self.filter_num,1]
            with tf.name_scope('conv_{}'.format(size)):
                filter_weights = tf.Variable(tf.random_normal(shape=filter_kernal,stddev=0.1))
                filter_biases =  tf.Variable(tf.random_normal(shape=[self.filter_num],stddev=0.1))
            conv = tf.nn.conv2d(context,filter_weights,strides=[1,1,1,1],padding='VALID')
            conv = tf.nn.bias_add(conv,filter_biases)
            pool_kernal = [1,self.review_length-size+1,1,1]
            pool = tf.nn.max_pool(conv,ksize=pool_kernal,strides=[1,1,1,1],padding='VALID')
            pools.append(pool)
        num_feature_total = self.filter_num * len(self.filter_size)
        pooled_total = tf.concat(pools,3)
        pooled_total = tf.reshape(pooled_total,[-1,num_feature_total])

        # gate
        with tf.name_scope('gate/user_gate'):
            Wxcr = tf.Variable(tf.random_normal(shape=[num_feature_total,self.factor_num]))
            Wxur = tf.Variable(tf.random_normal(shape=[self.factor_num,self.factor_num]))
            Wxch = tf.Variable(tf.random_normal(shape=[num_feature_total, self.factor_num]))
            Wxuh = tf.Variable(tf.random_normal(shape=[self.factor_num, self.factor_num]))
            bxr = tf.Variable(tf.constant(0.0,shape=[self.factor_num]))
            bxh = tf.Variable(tf.constant(0.0, shape=[self.factor_num]))
            Wxcz = tf.Variable(tf.random_normal(shape=[num_feature_total,self.factor_num]))
            Wxuz = tf.Variable(tf.random_normal(shape=[self.factor_num,self.factor_num]))
            bxz = tf.Variable(tf.constant(0.0,shape=[self.factor_num]))
        xr = tf.add_n(tf.matmul(pooled_total,Wxcr),tf.matmul(uvec,Wxur),bxr)
        xz = tf.add_n(tf.matmul(pooled_total,Wxcz),tf.matmul(uvec,Wxuz),bxz)
        uvec_hat = tf.tanh(tf.add_n(tf.matmul(pooled_total,Wxch),tf.maltiply(xr,tf.matmul(uvec,Wxuh)),bxh))
        uvec_final = tf.multiply(xz,uvec_hat) + tf.multiply((1-xz),uvec_hat)
        with tf.name_scope('gate/item_gate'):
            Wycr = tf.Variable(tf.random_normal(shape=[num_feature_total,self.factor_num]))
            Wyir = tf.Variable(tf.random_normal(shape=[self.factor_num,self.factor_num]))
            Wych = tf.Variable(tf.random_normal(shape=[num_feature_total, self.factor_num]))
            Wyuh = tf.Variable(tf.random_normal(shape=[self.factor_num, self.factor_num]))
            byr = tf.Variable(tf.constant(0.0,shape=[self.factor_num]))
            byh = tf.Variable(tf.constant(0.0, shape=[self.factor_num]))
            Wycz = tf.Variable(tf.random_normal(shape=[num_feature_total,self.factor_num]))
            Wyiz = tf.Variable(tf.random_normal(shape=[self.factor_num,self.factor_num]))
            byz = tf.Variable(tf.constant(0.0,shape=[self.factor_num]))
        yr = tf.add_n(tf.matmul(pooled_total,Wycr),tf.matmul(ivec,Wyir),byr)
        yz = tf.add_n(tf.matmul(pooled_total,Wycz),tf.matmul(uvec,Wyiz),byz)
        ivec_hat = tf.tanh(tf.add_n(tf.matmul(pooled_total, Wych),tf.maltiply(yr,tf.matmul(ivec, Wyuh)), byh))
        ivec_final = tf.multiply(yz, ivec_hat) + tf.multiply((1 - yz), ivec_hat)
        if self.phrase:
            final = tf.concat([uvec,ivec],axis=1)
        else:
            final = tf.concat([uvec_final,ivec_final],axis=1)
        with tf.name_scope('full_connected'):
            W1 = tf.Variable(tf.random_normal(shape=[2*self.factor_num,self.factor_num]))
            b1 = tf.Variable(tf.constant(0.0,shape=[self.factor_num]))
            W2 = tf.Variable(tf.random_normal(shape=[self.factor_num, 1]))
            b2 = tf.Variable(tf.constant(0.0, shape=[1]))
        f1 = tf.relu(tf.add(tf.matmul(final,W1),b1))
        f2 = tf.relu(tf.add(tf.matmul(f1, W2), b2))

        self.mse = tf.reduce_mean(tf.square(tf.subtract(tf.reduce_sum(f2,axis=1),self.rating)))
        self.mae = tf.reduce_mean(tf.abs(tf.subtract(tf.reduce_sum(f2, axis=1), self.rating)))
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.mse)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def fit(self,users,items,reviews,ratings):
        feed_dict = {self.user:users,self.item:items,self.reviews:reviews,self.rating:ratings,self.phase:False}
        _,mse,mae = tf.sess.run([self.opt,self.mse,self.mae],feed_dict=feed_dict)
        return mse,mae
    def predict(self,users,items,ratings):
        feed_dict = {self.user: users, self.item: items,self.rating: ratings,self.phrase:True}
        mse, mae = tf.sess.run([self.mse, self.mae], feed_dict=feed_dict)
        return mse,mae
if __name__ == '__main__':
    # parse command line arguments

    # load data from file

    # build up model

    # train the model

    # predict