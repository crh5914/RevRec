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
        #self.word2vec = Word2Vec(self.docs,min_count=1,iter=100,size=100,alpha=0.005,window=5)
        #with open(self.prefix+'_w2v.emb','rb') as fp:
            #self.word2vec = pickle.load(fp)
            #pickle.dump(self.word2vec,fp)
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
        #if '<PAD/>' not in self.d:
            #self.d['<PAD/>'] = 100
    #def build_dict(self):
        #self.d = {}
        #for doc in self.docs:
            #for word in doc:
                #self.d[word] = self.d.get(word, 0) + 1
    def map_word(self):
        self.word2idx = {w:(i+1) for i,w in enumerate(self.d)}
        self.num_words = len(self.word2idx) + 1
        self.idx2word = {i:w for i,w in enumerate(self.d)}
        self.numeric_docs = []
        self.max_len = 0
        #self.W = np.ones(shape=(len(self.word2idx),100))
        #for word in self.word2idx:
            #if word in self.word2vec:
                #self.W[self.word2idx[word]-1] = self.word2vec[word]
                #print(word)
        #lens = []
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
        #padding_embedding = tf.constant(0.0,shape=(1,self.embedding_size))
        #word_table = tf.concat([self.word_embedding,padding_embedding],axis=0)
        #doc_vec = tf.nn.embedding_lookup(word_table, self.doc)
        doc_vec = tf.nn.embedding_lookup(self.word_embedding, self.doc)
        user_vec = tf.nn.embedding_lookup(user_embedding,self.user)
        item_vec = tf.nn.embedding_lookup(item_embedding,self.item)
        bu = tf.nn.embedding_lookup(user_bias,self.user)
        bi = tf.nn.embedding_lookup(item_bias,self.item)
        uv = tf.concat([user_vec,item_vec],axis=1)
        key_hidden = tf.layers.dense(uv,self.num_mem,activation=tf.nn.tanh,name='key_hidden_layer')
        raw_weights = tf.layers.dense(key_hidden,self.num_mem,name='key_out_layer')
        key = tf.nn.softmax(raw_weights,axis=1)
        out = tf.matmul(key,memory)
        #transvec = tf.layers.dense(out,self.)
        word_vecs = tf.expand_dims(doc_vec, -1)
        #mr = tf.multiply(user_vec,item_vec)
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
        #x = tf.concat([feat,mr],axis=1)
        x = feat
        #x_ = tf.concat([out,mr],axis=1)
        x_ = out
        layer = {}
        for i,s in enumerate(self.layer_size):
            layer['layer%d'%i] = tf.layers.Dense(s,activation=tf.nn.sigmoid,name='full_connected_layer%d'%i)
        layer['predict_layer'] = tf.layers.Dense(1,activation=None,name='prediction_layer')
        for k in range(len(self.layer_size)):
            x = layer['layer%d'%k](x)
            x_ = layer['layer%d'%k](x_)
        x = layer['predict_layer'](x) + bu + bi + global_bias
        x_ = layer['predict_layer'](x_) + bu + bi + global_bias
        self.y = tf.reduce_sum(x,axis=1) #+ mr
        self.y_ = tf.reduce_sum(x_,axis=1)# + mr
        #self.loss = tf.reduce_mean(tf.square(tf.subtract(self.rating,self.y)))
        self.mse = tf.reduce_mean(tf.square(tf.subtract(self.rating,self.y)))
        #self.loss = tf.nn.l2_loss(tf.subtract(self.rating,self.y)) + tf.nn.l2_loss(tf.subtract(feat,out))
        self.loss = self.mse + tf.nn.l2_loss(tf.subtract(feat,out))
        #self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.rating,self.y_)))
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_opt = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
    def train(self,epochs,batch_size):
        best_mse,best_mae,best_epoch = 10,10,0
        #self.sess.run(tf.assign(self.word_embedding,self.ds.W))
        mse,mae = self.test(batch_size)
        print('init,mse:{},mae:{}'.format(mse,mae)) 
        for epoch in range(epochs):
            losses,count= 0.0,0
            for batch_users,batch_docs,batch_masks,batch_items,batch_ratings in tqdm(self.ds.generate_train_batch(batch_size)):
                #batch_docs,batch_masks,batch_labels = shuffle(batch_docs,batch_masks,batch_labels)
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
        print('best rmse:{},mae:{},taken at epoch:{}'.format(np.sqrt(best_mse),best_mae,best_epoch))
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
        #mse,mae,count = 0,0,0
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
    def valid(self,batch_size):
        ys_,ys = [],[]
        for batch_users,batch_docs,batch_masks,batch_items,batch_ratings in tqdm(self.ds.get_valid_batch(batch_size)):
            feed_dict = {self.user:batch_users,self.doc:batch_docs,self.mask:batch_masks,self.item:batch_items,self.rating:batch_ratings,self.dropout_keep_prob:self.keep_prob,self.alpha:0.1}
            y_ = self.sess.run(self.y,feed_dict=feed_dict)
            ys_ += list(y_)
            ys += batch_ratings
        ys_,ys = np.array(ys_),np.array(ys)
        mse = np.mean(np.square(ys_-ys))
        mae = np.mean(np.abs(ys_-ys))
        return mse,mae
def main():
    sess = tf.Session()
    #prefix = './data/baby/baby'
    prefix = './data/beauty/beauty'
    ds = Corpus(prefix)
    ds.build_up()
    dim = 16
    embedding_size = 16
    filter_sizes = [2,3,5]
    num_filters = 16
    num_mem = 32
    keep_prob = 0.8
    lr = 0.0005
    l2_reg_lambda = 0.0
    batch_size = 256
    epochs = 100
    layer_size = [32]
    model = SentimentCNN(sess,ds,num_mem,embedding_size,dim,filter_sizes,num_filters,layer_size,lr,l2_reg_lambda,keep_prob)
    #print(ds.max_len)
    #sess.run(model.word_embedding.assign(ds.W))
    model.train(epochs,batch_size)

if __name__ == '__main__':
    main()

