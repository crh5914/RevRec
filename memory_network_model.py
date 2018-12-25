import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from gensim.models import Word2Vec
import pickle
from nltk.corpus import stopwords
from tqdm import tqdm
class Corpus:
    def __init__(self,prefix,threshold=0.8,filter_stop=True):
        self.prefix = prefix
        self.threshold = threshold
        self.stop_words = []
        if filter_stop:
            self.stop_words = stopwords.words('english')
    def load(self):
        self.train_data = []
        self.test_data = []
        self.num_users = 0
        self.num_items = 0
        self.docs = []
        with open(self.prefix+'_train.csv','r') as fp:
            for line in fp:
                vals = line.strip().split(',')
                u,i,r = int(vals[0]),int(vals[1]),float(vals[3])
                self.num_users = max(u,self.num_users)
                self.num_items = max(i,self.num_items)
                nline = [u,i,r]
                words = vals[2].strip().split(' ')
                self.docs.append(words)
                self.train_data.append(nline)
        with open(self.prefix+'_test.csv','r') as fp:
            for line in fp:
                vals = line.strip().split(',')
                u,i,r = int(vals[0]),int(vals[1]),float(vals[3])
                self.num_users = max(u,self.num_users)
                self.num_items = max(i,self.num_items)
                nline = [u,i,r]
                self.test_data.append(nline)
    def build_up(self):
        self.load()
        self.build_dict()
        self.map_word()
        #self.padding_sentence()
        #self.split(0.2)
    def build_dict(self):
        d = {}
        for doc in self.docs:
            for word in doc:
                d[word] = d.get(word, 0) + 1
        vals = sorted(d.values(),reverse=True)
        threshold = vals[int(self.threshold*len(vals))]
        self.d = {}
        for word in d:
            if d.get(word,0) > threshold and word not in self.stop_words:
                self.d[word] = d[word]
    def map_word(self):
        self.word2idx = {w:(i+1) for i,w in enumerate(self.d)}
        self.num_words = len(self.word2idx) + 1
        self.idx2word = {i:w for i,w in enumerate(self.d)}
        self.numeric_docs = []
        self.max_len = 0
        for doc in self.docs:
            words = [self.word2idx[word] for word in doc if word in self.word2idx]
            #lens.append(len(words))
            if len(words) > self.max_len:
                self.max_len = len(words)
            self.numeric_docs.append(words)
        #lens = sorted(lens)
        self.max_len = 300
        print('word dict size:{},max length:{}'.format(len(self.word2idx),self.max_len))
    def generate_train_batch(self,batch_size):
        idx = list(range(len(self.numeric_docs)))
        np.random.shuffle(idx)
        for i in range(0,len(self.numeric_docs),batch_size):
            end = i + batch_size
            if end > len(self.numeric_docs):
                end = len(self.numeric_docs)
            batch_users,batch_docs,batch_masks,batch_items,batch_ratings =[],[],[],[],[]
            for j in idx[i:end]:
                u,it,r = self.train_data[j]
                doc = self.numeric_docs[j]
                if len(doc) > self.max_len:
                    mask = self.max_len
                    doc = doc[:self.max_len]
                else:
                    mask = len(doc)
                    doc = doc + [0]*(self.max_len - mask)
                batch_users.append(u-1)
                batch_items.append(it-1)
                batch_docs.append(doc)
                batch_masks.append(mask)
                batch_ratings.append(r)
            yield batch_users,batch_docs,batch_masks,batch_items,batch_ratings
    def generate_test_batch(self,batch_size):
        for i in range(0,len(self.test_data),batch_size):
            end = i + batch_size
            if end > len(self.test_data):
                end = len(self.test_data)
            batch_users,batch_items,batch_ratings = [],[],[]
            for j in range(i,end):
                u,i,r = self.test_data[j]
                batch_users.append(u-1)
                batch_items.append(i-1)
                batch_ratings.append(r)
            yield batch_users,batch_items,batch_ratings
class SentimentCNN:
    def __init__(self,sess,ds,num_mem,embedding_size,dim,filter_sizes,num_filters,layer_size,lr,l2_reg_lambda,keep_prob):
        self.sess = sess
        self.ds = ds
        self.embedding_size = embedding_size
        self.dim = dim
        self.num_mem = num_mem
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.layer_size = layer_size
        self.lr = lr
        self.l2_reg_lambda = l2_reg_lambda
        self.keep_prob = keep_prob
        self.build_up()
    def build_up(self):
        self.user = tf.placeholder(shape=[None,],dtype=tf.int32)
        self.doc = tf.placeholder(shape=[None,self.ds.max_len],dtype=tf.int32)
        self.mask = tf.placeholder(shape=[None,],dtype=tf.int32) 
        self.item = tf.placeholder(shape=[None,],dtype=tf.int32)
        self.rating = tf.placeholder(shape=[None,],dtype=tf.float32)
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.alpha = tf.placeholder(tf.float32)
        #l2_loss = tf.constant(0.0)
        # embedding layer
        self.word_embedding = tf.Variable(tf.random_normal(shape=(self.ds.num_words, self.embedding_size),stddev=0.01))
        user_embedding = tf.Variable(tf.random_normal(shape=(self.ds.num_users,self.dim),stddev=0.01))
        item_embedding = tf.Variable(tf.random_normal(shape=(self.ds.num_items,self.dim),stddev=0.01))
        memory = tf.Variable(tf.random_normal(shape=(self.num_mem,self.dim),stddev=0.01))
        user_bias = tf.Variable(tf.constant(0.0,shape=(self.ds.num_users,1)))
        item_bias = tf.Variable(tf.constant(0.0,shape=(self.ds.num_items,1)))
        global_bias = tf.Variable(tf.constant(0.0))
        # embedding lookup
        doc_vec = tf.nn.embedding_lookup(self.word_embedding, self.doc)
        user_vec = tf.nn.embedding_lookup(user_embedding,self.user)
        item_vec = tf.nn.embedding_lookup(item_embedding,self.item)
        bu = tf.nn.embedding_lookup(user_bias,self.user)
        bi = tf.nn.embedding_lookup(item_bias,self.item)
        uv = tf.concat([user_vec,item_vec],axis=1)
        AWh = tf.Variable(tf.random_normal(shape=(2*self.dim,self.num_mem),stddev=0.01),name="AWh")
        bh = tf.Variable(tf.constant(0.0,shape=(self.num_mem,)),name="bh")
        AWo = tf.Variable(tf.random_normal(shape=(self.num_mem,self.num_mem),stddev=0.01),name="AWo")
        bo = tf.Variable(tf.constant(0.0,shape=(self.num_mem,)),name="bo")
        key_hidden = tf.nn.tanh(tf.matmul(uv,AWh) + bh)
        raw_weights = tf.matmul(key_hidden,AWo) + bo
        key = tf.nn.softmax(raw_weights,axis=1)
        out = tf.matmul(key,memory)
        word_vecs = tf.expand_dims(doc_vec, -1)
        #text cnn
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            # Convolution Layer
            kernel_size = [filter_size, self.embedding_size]
            conv = tf.layers.conv2d(word_vecs,self.num_filters,kernel_size,activation=tf.nn.relu)
            # Maxpooling over the outputs
            pooled = tf.layers.max_pooling2d(conv,(self.ds.max_len - filter_size + 1, 1),(1,1))
            pooled_outputs.append(pooled)
        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        doc_feat = tf.reshape(h_pool, [-1, num_filters_total])
        feat = tf.layers.dense(doc_feat,self.dim,activation=tf.nn.relu,name='transform_layer')
        x = feat
        x_ = out
        W1 = tf.Variable(tf.random_normal(shape=(self.dim,self.layer_size[0]),stddev=0.01),name="W1")
        b1 = tf.Variable(tf.constant(0.0,shape=(self.layer_size[0],)),name="b1")
        W2 = tf.Variable(tf.random_normal(shape=(self.layer_size[0],1),stddev=0.01),name="W2")
        b2 = tf.Variable(tf.constant(0.0,shape=(1,)),name="b2")
        x_h = tf.nn.sigmoid(tf.add(tf.matmul(x,W1),b1))
        x_o = tf.matmul(x_h,W2) + b2
        x_h_ = tf.nn.sigmoid(tf.add(tf.matmul(x_,W1),b1))
        x_o_ = tf.matmul(x_h_,W2)+b2
        x_f = x_o + bu + bi + global_bias
        x_f_ = x_o_ + bu + bi + global_bias
        self.y = tf.reduce_sum(x_f,axis=1) 
        self.y_ = tf.reduce_sum(x_f_,axis=1)
        self.mse = tf.reduce_mean(tf.square(tf.subtract(self.rating,self.y)))
        #self.loss = tf.nn.l2_loss(tf.subtract(self.rating,self.y)) + tf.nn.l2_loss(tf.subtract(feat,out))
        regloss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(AWh) + tf.nn.l2_loss(AWo)
        self.loss = self.mse + tf.nn.l2_loss(tf.subtract(feat,out)) + self.l2_reg_lambda*regloss
        #self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.rating,self.y_)))
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_opt = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
    def train(self,epochs,batch_size):
        best_mse,best_mae,best_epoch = 10,10,0
        mse,mae = self.test(batch_size)
        print('init,mse:{},mae:{}'.format(mse,mae)) 
        for epoch in range(epochs):
            losses,count= 0.0,0
            for batch_users,batch_docs,batch_masks,batch_items,batch_ratings in tqdm(self.ds.generate_train_batch(batch_size)):
                feed_dict = {self.user:batch_users,self.doc:batch_docs,self.mask:batch_masks,self.item:batch_items,self.rating:batch_ratings,self.dropout_keep_prob:self.keep_prob,self.alpha:0.1}
                _,loss = self.sess.run([self.train_opt,self.loss],feed_dict=feed_dict)
                losses += len(batch_ratings)*loss
                count += len(batch_ratings)
            losses = losses / count
            tmse,tmae = self.test_train(batch_size)
            mse,mae = self.test(batch_size)
            print('epoch:{},loss:{},train_mse:{},train_mae:{},test_mse:{},test_mae:{}'.format(epoch,losses,tmse,tmae,mse,mae))
            if best_mse > mse:
                best_mse,best_mae,best_epoch = mse,mae,epoch
        print('best mse:{},mae:{},taken at epoch:{}'.format(best_mse,best_mae,best_epoch))
    def test_train(self,batch_size):
        ys_,ys = [],[]
        for batch_users,batch_docs,batch_masks,batch_items,batch_ratings in tqdm(self.ds.generate_train_batch(batch_size)):
            feed_dict = {self.user:batch_users,self.item:batch_items,self.rating:batch_ratings,self.dropout_keep_prob:1.0,self.alpha:0.1}
            y_ = self.sess.run(self.y_,feed_dict=feed_dict)
            ys_ += list(y_)
            ys += batch_ratings
        err = np.array(ys_) - np.array(ys)
        mse = np.mean(np.square(err))
        mae = np.mean(np.abs(err))
        return mse,mae
    def test(self,batch_size):
        ys_,ys = [],[]
        for batch_users,batch_items,batch_ratings in tqdm(self.ds.generate_test_batch(batch_size)):
            feed_dict = {self.user:batch_users,self.item:batch_items,self.rating:batch_ratings,self.dropout_keep_prob:1.0,self.alpha:0.1}
            y_ = self.sess.run(self.y_,feed_dict=feed_dict)
            ys_ += list(y_)
            ys += batch_ratings
        err = np.array(ys_)-np.array(ys)
        mse = np.mean(np.square(err))
        mae = np.mean(np.abs(err))
        return mse,mae
def main():
    sess = tf.Session()
    prefix = './data/baby/baby'
    #prefix = './data/beauty/beauty'
    ds = Corpus(prefix)
    ds.build_up()
    dim = 16
    embedding_size = 16
    filter_sizes = [2,3,5]
    num_filters = 16
    num_mem = 32
    keep_prob = 0.8
    lr = 0.0005
    l2_reg_lambda = 0.01
    batch_size = 256
    epochs = 100
    layer_size = [32]
    model = SentimentCNN(sess,ds,num_mem,embedding_size,dim,filter_sizes,num_filters,layer_size,lr,l2_reg_lambda,keep_prob)
    model.train(epochs,batch_size)

if __name__ == '__main__':
    main()

