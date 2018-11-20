import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from gensim.models import Word2Vec
import pickle
from tqdm import tqdm
class Corpus:
    def __init__(self,prefix):
        self.prefix = prefix
    def load(self):
        self.ratings = []
        self.docs = []
        with open(self.prefix+'.csv','r') as fp:
            for line in fp:
                vals = line.strip().split('_|_')
                words = vals[2].split(' ')
                label = float(vals[3])
                if label > 3.0:
                    label = 1.0
                else:
                    label = 0.0
                self.ratings.append(label)
                self.docs.append(words)
        #self.word2vec = Word2Vec(self.docs,min_count=1,iter=1000,size=16,alpha=0.005,window=5)
        with open(self.prefix+'_w2v.emb','rb') as fp:
            self.word2vec = pickle.load(fp)
        self.num_pos = np.sum(self.ratings)
        self.num_neg = len(self.ratings) - self.num_pos
        print('load data from file done. positive instances:{},negative instances:{}'.format(self.num_pos,self.num_neg))
    def build_up(self):
        self.load()
        self.build_dict()
        self.map_word()
        self.split(0.2)
    def build_dict(self):
        self.d = {}
        for doc in self.docs:
            for word in doc:
                self.d[word] = self.d.get(word, 0) + 1
    def map_word(self):
        self.word2idx = {w:i for i,w in enumerate(self.d)}
        self.num_words = len(self.word2idx)
        self.idx2word = {i:w for i,w in enumerate(self.d)}
        self.numeric_docs = []
        self.max_len = 0
        self.W = np.ones(shape=(len(self.word2idx),100))
        for word in self.word2idx:
            if word in self.word2vec:
                self.W[self.word2idx[word]-1] = self.word2vec[word]
                #print(word)
        for doc in self.docs:
            words = [self.word2idx[word] for word in doc if word in self.word2idx]
            if len(words) > self.max_len:
                self.max_len = len(words)
            self.numeric_docs.append(words)
        print('word dict size:{},max length:{}'.format(len(self.word2idx),self.max_len))
    def split(self,test_rate):
        idx = list(range(len(self.docs)))
        np.random.shuffle(idx)
        size = int((1-test_rate) * len(self.docs))
        self.tr_idxs = idx[:size]
        self.ts_idxs = idx[size:]
    def generate_train_test_batch(self,batch_size):
        for i in range(0,len(self.tr_idxs),batch_size):
            end = i + batch_size
            if end > len(self.tr_idxs):
                end = len(self.tr_idxs)
            batch_docs = []
            batch_masks = []
            batch_labels =[]
            for j in range(i,end):
                k = self.tr_idxs[j]
                doc,label = self.numeric_docs[k],self.ratings[k]
                #print(doc,label)
                mask = len(doc)
                doc = doc + [0]*(self.max_len - mask)
                batch_masks.append(mask)
                batch_labels.append(label)
                batch_docs.append(doc)
            yield batch_docs,batch_masks,batch_labels
    def generate_train_batch(self,batch_size):
        for i in range(0,len(self.tr_idxs),batch_size):
            end = i + batch_size
            if end > len(self.tr_idxs):
                end = len(self.tr_idxs)
            batch_docs = []
            batch_masks = []
            batch_labels =[]
            for j in range(i,end):
                k = self.tr_idxs[j]
                doc,label = self.numeric_docs[k],self.ratings[k]
                mask = len(doc)
                doc = doc + [0]*(self.max_len - mask)
                batch_masks.append(mask)
                batch_labels.append(label)
                batch_docs.append(doc)
            yield batch_docs,batch_masks,batch_labels
    def generate_test_batch(self,batch_size):
        for i in range(0,len(self.ts_idxs),batch_size):
            end = i + batch_size
            if end > len(self.ts_idxs):
                end = len(self.ts_idxs)
            batch_docs = []
            batch_masks = []
            batch_labels =[]
            for j in range(i,end):
                k = self.ts_idxs[j]
                doc,label = self.numeric_docs[k],self.ratings[k]
                mask = len(doc)
                doc = doc + [0]*(self.max_len - mask)
                batch_masks.append(mask)
                batch_labels.append(label)
                batch_docs.append(doc)
            yield batch_docs,batch_masks,batch_labels
class SentimentCNN:
    def __init__(self,sess,ds,dim,filter_sizes,num_filters,lr,l2_reg_lambda,keep_prob):
        self.sess = sess
        self.ds = ds
        self.num_words = ds.num_words
        self.dim = dim
        self.length = ds.max_len
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.lr = lr
        self.l2_reg_lambda = l2_reg_lambda
        self.keep_prob = keep_prob
        self.build_up()
    def build_up(self):
        self.x = tf.placeholder(shape=[None,self.length],dtype=tf.int32)
        self.mask = tf.placeholder(shape=[None,],dtype=tf.int32)
        self.y = tf.placeholder(shape=[None,],dtype=tf.float32)
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        l2_loss = tf.constant(0.0)
        # embedding layer
        self.word_embedding = tf.Variable(tf.random_uniform(shape=[self.num_words, self.dim], minval=-0.1,maxval=0.1))
        # embedding lookup
        word_vecs = tf.nn.embedding_lookup(self.word_embedding, self.x)
        mask = tf.expand_dims(tf.cast(tf.sequence_mask(self.mask,self.length),dtype=tf.float32),axis=-1)
        word_vecs = tf.multiply(mask,word_vecs)
        #word_vecs = tf.reduce_sum(word_vecs,axis=1)
        word_vecs = tf.expand_dims(word_vecs, -1)
        # text cnn
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            # Convolution Layer
            filter_shape = [filter_size, self.dim, 1, self.num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
            conv = tf.nn.conv2d(word_vecs, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            # Apply nonlinearity
            #h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(conv, ksize=[1, self.length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                    padding='VALID', name="pool")
            pooled_outputs.append(pooled)
        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        #num_filters_total = 16
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        #h_pool_flat = word_vecs
        h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        # Final (unnormalized) scores and predictions
        W1 = tf.random_normal(shape=[num_filters_total, num_filters_total//2], stddev=0.1)
        b1 = tf.Variable(tf.constant(0.1, shape=[num_filters_total//2]))
        W2 = tf.random_normal(shape=[num_filters_total//2, 1], stddev=0.1)
        b2 = tf.Variable(tf.constant(0.1, shape=[1]))
        f1 = tf.nn.sigmoid(tf.nn.xw_plus_b(h_drop, W1, b1))
        l2_loss += tf.nn.l2_loss(W1)
        l2_loss += tf.nn.l2_loss(b1)
        l2_loss += tf.nn.l2_loss(W2)
        l2_loss += tf.nn.l2_loss(b2)
        self.y_ = tf.sigmoid(tf.reduce_sum(tf.nn.xw_plus_b(f1, W2, b2),axis=1))
        log_loss = tf.losses.log_loss(self.y,self.y_)
        mse = tf.reduce_mean(tf.square(tf.subtract(self.y,self.y_)))
        self.loss = log_loss + self.l2_reg_lambda * l2_loss
        self.train_opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
    def train(self,epochs,batch_size):
        best_acc = 0
        best_epoch = 0
        self.sess.run(tf.assign(self.word_embedding,self.ds.W))
        for epoch in range(epochs):
            losses,count = 0.0,0
            for batch_docs,batch_masks,batch_labels in tqdm(self.ds.generate_train_batch(batch_size)):
                #batch_docs,batch_masks,batch_labels = shuffle(batch_docs,batch_masks,batch_labels)
                feed_dict = {self.x:batch_docs,self.y:batch_labels,self.mask:batch_masks,self.dropout_keep_prob:self.keep_prob}
                count += len(batch_labels)
                _,loss = self.sess.run([self.train_opt,self.loss],feed_dict=feed_dict)
                losses += len(batch_labels)*loss
            losses = losses / count
            acc = self.train_test(batch_size)
            accuracy = self.test(batch_size)
            print('epoch:{},loss:{},train accuracy:{},accuracy:{}'.format(epoch,losses,acc,accuracy))
            if best_acc < accuracy:
                best_acc = accuracy
                best_epoch = epoch
        print('best accuracy:{} taken at epoch:{}'.format(best_acc,best_epoch))
    def test(self,batch_size):
        ys = []
        ys_ = []
        for batch_docs,batch_masks,batch_labels in tqdm(self.ds.generate_test_batch(batch_size)):
            feed_dict = {self.x:batch_docs,self.y:batch_labels,self.mask:batch_masks,self.dropout_keep_prob:1.0}
            ys = ys + batch_labels
            y_ = self.sess.run(self.y_,feed_dict=feed_dict)
            #print(y_)
            y_ = [1 if s > 0.5 else 0 for s in y_]
            ys_ = ys_ + y_
        accuracy = accuracy_score(ys,ys_)
        return accuracy
    def train_test(self,batch_size):
        ys = []
        ys_ = []
        for batch_docs,batch_masks,batch_labels in self.ds.generate_train_test_batch(batch_size):
            feed_dict = {self.x:batch_docs,self.y:batch_labels,self.mask:batch_masks,self.dropout_keep_prob:1.0}
            ys = ys + batch_labels
            y_ = self.sess.run(self.y_,feed_dict=feed_dict)
            #print(y_)
            y_ = [1 if s > 0.5 else 0 for s in y_]
            ys_ = ys_ + y_
        accuracy = accuracy_score(ys,ys_)
        return accuracy
def main():
    sess = tf.Session()
    #file = './data/cutted_review_with_rating.csv'
    #file = './data/restaurant_review.csv'
    prefix = './data/baby/baby'
    ds = Corpus(prefix)
    ds.build_up()
    dim = 100
    filter_sizes = [2]
    num_filters = 64
    keep_prob = 0.8
    lr = 0.001
    l2_reg_lambda = 0.0
    batch_size = 256
    epochs = 100
    model = SentimentCNN(sess,ds,dim,filter_sizes,num_filters,lr,l2_reg_lambda,keep_prob)
    sess.run(model.word_embedding.assign(ds.W))
    model.train(epochs,batch_size)

if __name__ == '__main__':
    main()

