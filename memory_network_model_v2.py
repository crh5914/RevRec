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
        #self.word2vec = Word2Vec(self.docs,min_count=1,iter=1000,size=16,alpha=0.005,window=5)
        with open(self.prefix+'_w2v.emb','rb') as fp:
            self.word2vec = pickle.load(fp)
    def build_up(self):
        self.load()
        self.build_dict()
        self.map_word()
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
    def generate_train_batch(self,batch_size):
        for i in range(0,len(self.numeric_docs),batch_size):
            end = i + batch_size
            if end > len(self.numeric_docs):
                end = len(self.numeric_docs)
            batch_users,batch_docs,batch_masks,batch_items,batch_ratings =[],[],[],[],[]
            for j in range(i,end):
                u,it,r = self.train_data[j]
                doc = self.numeric_docs[j]
                #print(doc,label)
                mask = len(doc)
                doc = doc + [len(self.word2idx)]*(self.max_len - mask)
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
    def __init__(self,sess,ds,num_mem,dim,filter_sizes,num_filters,lr,l2_reg_lambda,keep_prob):
        self.sess = sess
        self.ds = ds
        self.dim = dim
        self.num_mem = num_mem
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
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
        self.word_embedding = tf.Variable(tf.truncated_normal(shape=(self.ds.num_words, self.dim), stddev=0.01))
        user_embedding = tf.Variable(tf.truncated_normal(shape=(self.ds.num_users,self.dim),stddev=0.01))
        item_embedding = tf.Variable(tf.truncated_normal(shape=(self.ds.num_items,self.dim),stddev=0.01))
        # embedding lookup
        padding_embedding = tf.constant(0.0,shape=(1,self.dim))
        word_table = tf.concat([self.word_embedding,padding_embedding],axis=0)
        doc_vec = tf.nn.embedding_lookup(word_table, self.doc)
        user_vec = tf.nn.embedding_lookup(user_embedding,self.user)
        item_vec = tf.nn.embedding_lookup(item_embedding,self.item)
        trans_vec = tf.subtract(item_vec,user_vec)
        #word_vecs = tf.reduce_sum(word_vecs,axis=1)
        #sum_doc_vec =tf.div(tf.reduce_sum(doc_vec,axis=1),tf.expand_dims(tf.cast(self.mask,dtype=tf.float32),axis=-1)) 
        #sum_doc_vec = tf.reshape(tf.tile(tf.expand_dims(sum_doc_vec,axis=-1),(1,self.ds.max_len,1)),(-1,self.dim))
        #flat_doc_vec = tf.reshape(doc_vec,(-1,self.dim))
        #attention_in = tf.concat([flat_doc_vec,sum_doc_vec],axis=1)
        #attentions = tf.layers.dense(attention_in,1,activation=tf.nn.relu,name='word_attetion_layer')
        #raw_weights = tf.reshape(attentions,(-1,self.ds.max_len,1))
        #mask = tf.expand_dims(tf.cast(tf.sequence_mask(self.mask,self.ds.max_len),dtype=tf.float32),axis=-1)
        #exp_weights = tf.exp(mask*raw_weights)
        #weights = tf.div(exp_weights,tf.reduce_sum(exp_weights,axis=1,keep_dims=True))
        #doc = tf.reduce_sum(weights*doc_vec,axis=1)
        coeff = tf.expand_dims(tf.pow(tf.cast(self.mask,dtype=tf.float32),-1),axis=1)
        doc = coeff*tf.reduce_sum(doc_vec,axis=1)
        F = tf.layers.Dense(1,activation=tf.nn.relu,name='prediction')
        #rev_mem_key = tf.Variable(tf.truncated_normal(shape=(3*self.dim,self.num_mem),stddev=0.01))
        rev_mem_key = tf.Variable(tf.truncated_normal(shape=(3*self.dim,self.num_mem),stddev=0.01))
        query = tf.concat([user_vec,item_vec,doc],axis=1)
        #query = doc
        mem_slot = tf.Variable(tf.truncated_normal(shape=(self.num_mem,self.dim),stddev=0.01))
        train_rev_key = tf.matmul(query,rev_mem_key)
        train_rev_attention = tf.nn.softmax(train_rev_key)
        #train_rev_attention = tf.expand_dims(train_rev_attention,axis=-1)
        mem_vec = tf.matmul(train_rev_attention,mem_slot)
        self.rating_ = F(mem_vec)
        test_rating_ = F(trans_vec)
        self.y_ = tf.reduce_sum(F(trans_vec),axis=1)
        #self.test_mse = tf.reduce_mean(tf.square(tf.subtract(self.rating,test_rating_)))
        #self.test_mae = tf.reduce_mean(tf.abs(tf.subtract(self.rating,test_rating_)))  
        mse = tf.reduce_mean(tf.square(tf.subtract(self.rating,self.rating_)))
        transloss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(trans_vec,mem_vec))))
        self.loss = (1-self.alpha)*mse + self.alpha*transloss
        self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
    def train(self,epochs,batch_size):
        best_mse,best_mae,best_epoch = 10,10,0
        self.sess.run(tf.assign(self.word_embedding,self.ds.W))
        for epoch in range(epochs):
            losses,count = 0.0,0
            for batch_users,batch_docs,batch_masks,batch_items,batch_ratings in tqdm(self.ds.generate_train_batch(batch_size)):
                #batch_docs,batch_masks,batch_labels = shuffle(batch_docs,batch_masks,batch_labels)
                feed_dict = {self.user:batch_users,self.doc:batch_docs,self.mask:batch_masks,self.item:batch_items,self.rating:batch_ratings,self.dropout_keep_prob:self.keep_prob,self.alpha:0.7}
                count += len(batch_ratings)
                _,loss = self.sess.run([self.train_opt,self.loss],feed_dict=feed_dict)
                losses += len(batch_ratings)*loss
            losses = losses / count
            #acc = self.train_test(batch_size)
            mse,mae = self.test(batch_size)
            print('epoch:{},loss:{},mse:{},mae:{}'.format(epoch,losses,mse,mae))
            if best_mse > mse:
                best_mse,best_mae,best_epoch = mse,mae,epoch
        print('best mse:{},mae:{},taken at epoch:{}'.format(best_mse,best_mae,best_epoch))
    def test(self,batch_size):
        #mse,mae,count = 0,0,0
        ys_,ys = [],[]
        for batch_users,batch_items,batch_ratings in tqdm(self.ds.generate_test_batch(batch_size)):
            feed_dict = {self.user:batch_users,self.item:batch_items,self.rating:batch_ratings,self.dropout_keep_prob:1.0}
            y_ = self.sess.run(self.y_,feed_dict=feed_dict)
            ys_ += list(y_)
            ys += batch_ratings
            #count += len(batch_users)
            #mse += bmse * len(batch_users)
            #mae += bmae*len(batch_users)
        ys_,ys = np.array(ys_),np.array(ys)
        mse = np.mean(np.square(ys_-ys))
        mae = np.mean(np.abs(ys_-ys))
        return mse,mae
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
    prefix = './data/baby/baby'
    ds = Corpus(prefix)
    ds.build_up()
    dim = 100
    filter_sizes = [2]
    num_filters = 128
    num_mem = 32
    keep_prob = 0.8
    lr = 0.001
    l2_reg_lambda = 0.0
    batch_size = 256
    epochs = 100
    model = SentimentCNN(sess,ds,num_mem,dim,filter_sizes,num_filters,lr,l2_reg_lambda,keep_prob)
    sess.run(model.word_embedding.assign(ds.W))
    model.train(epochs,batch_size)

if __name__ == '__main__':
    main()

