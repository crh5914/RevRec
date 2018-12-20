import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
class DataSet:
    def __init__(self,prefix,length):
        self.prefix = prefix
        self.length = length
        self.build_up()
    def tokenize(self,s):
        s = s.lower()
        s = s.replace('[^a-zA-Z]',' ')
        s = s.replace('\s{2,}',' ')
        return s.split()
    def build_up(self):
        self.num_user,self.num_item = 0,0
        self.data_train = []
        self.data_test = []
        self.docs = []
        word_freq = {}
        #self.user_bias = {}
        #self.item_bias = {}
        #self.global_bias = []
        with open(self.prefix+'train.csv','r') as fp:
            for line in fp:
                u,i,doc,r = line.split(',')[:4]
                words = doc.split()
                for word in words:
                    word_freq[word] = word_freq.get(word,0) + 1
                u,i,r = int(u),int(i),float(r)
                #if u not in self.user_bias:
                #    self.user_bias[u] = []
                #self.user_bias[u].append(r)
                #if i not in self.item_bias:
                #    self.item_bias[i] = []
                #self.global_bias.append(r)
                #self.item_bias[i].append(r)
                self.num_user = max(self.num_user,u)
                self.num_item = max(self.num_item,i)
                self.docs.append(words)
                self.data_train.append([u,i,r])
        with open(self.prefix+'test.csv','r') as fp:
            for line in fp:
                u,i,doc,r = line.split(',')[:4]
                u,i,r = int(u),int(i),float(r)
                self.num_user = max(self.num_user,u)
                self.num_item = max(self.num_item,i)
                self.data_test.append([u,i,r])
        total_freq = sum(word_freq.values())
        threshold = int(0.0001 * total_freq)
        vocab = [w for w in word_freq if word_freq[w] >= total_freq]
        self.word2id = {w:i+1 for i,w in enumerate(vocab)}
        self.numeric_docs = []
        for doc in self.docs:
            ndoc = [self.word2id[word] for word in doc if word in self.word2id]
            self.numeric_docs.append(ndoc)
        self.vocab_size = len(self.word2id) + 1
        print('vocab size:{}'.format(self.vocab_size))
    def generate_train_batch(self,batch_size):
        data_train,docs_train = shuffle(self.data_train,self.numeric_docs)
        user_batch,item_batch,mask_batch,doc_batch,rating_batch = [],[],[],[],[]
        for idx in range(len(data_train)):
            u,i,r = data_train[idx]
            doc = self.numeric_docs[idx]
            if len(doc) > self.length:
                doc = doc[:self.length]
                mask = len(doc)
            else:
                mask = len(doc)
                doc = doc + [0]*(self.length - mask)
            user_batch.append(u-1)
            item_batch.append(i-1)
            mask_batch.append(mask)
            doc_batch.append(doc)
            rating_batch.append(r)
            if len(user_batch) == batch_size:
                yield user_batch,item_batch,doc_batch,mask_batch,rating_batch
                user_batch, item_batch, mask_batch, doc_batch, rating_batch = [], [], [], [], []
        if len(user_batch) > 0:
            yield user_batch, item_batch, doc_batch, mask_batch, rating_batch

    def generate_test_batch(self, batch_size):
        data_test = shuffle(self.data_test)
        user_batch, item_batch, rating_batch = [], [], []
        for idx in range(len(data_test)):
            u, i, r = data_test[idx]
            user_batch.append(u-1)
            item_batch.append(i-1)
            rating_batch.append(r)
            if len(user_batch) == batch_size:
                yield user_batch, item_batch, rating_batch
                user_batch, item_batch,rating_batch = [], [], []
        if len(user_batch) > 0:
            yield user_batch, item_batch,rating_batch

class TransRev:
    def __init__(self,sess,ds,batch_size,num_factor,alpha,mu,lr):
        self.sess = sess
        self.ds = ds
        self.batch_size = batch_size
        self.vocab_size = ds.vocab_size
        self.num_user = ds.num_user
        self.num_item = ds.num_item
        self.num_factor = num_factor
        self.alpha = alpha
        self.mu = mu
        self.lr = lr
        self.test_mse,self.test_mae = 10,10
        self.build_up()
    def build_up(self):
        self.user = tf.placeholder(shape=(None,),dtype=tf.int32)
        self.item = tf.placeholder(shape=(None,),dtype=tf.int32)
        self.rev = tf.placeholder(shape=(None,None),dtype=tf.int32)
        self.mask = tf.placeholder(shape=(None,),dtype=tf.int32)
        self.rating = tf.placeholder(shape=(None,),dtype=tf.float32)
        user_embedding = tf.Variable(tf.random_normal(shape=(self.num_user,self.num_factor),stddev=0.01))
        item_embedding = tf.Variable(tf.random_normal(shape=(self.num_item,self.num_factor),stddev=0.01))
        word_embedding = tf.Variable(tf.random_normal(shape=(self.vocab_size,self.num_factor),stddev=0.01))
        common_rev = tf.Variable(tf.constant(0.0,shape=(1,self.num_factor)))
        user_bias = tf.Variable(tf.constant(0.0,shape=(self.num_user,1)))
        item_bias = tf.Variable(tf.constant(0.0,shape=(self.num_item,1)))
        global_bias = tf.Variable(tf.constant(0.0))
        ei = tf.nn.embedding_lookup(item_embedding,self.item)
        eu = tf.nn.embedding_lookup(user_embedding,self.user)
        bu = tf.nn.embedding_lookup(user_bias,self.user)
        bi = tf.nn.embedding_lookup(item_bias,self.item)
        vt = tf.nn.embedding_lookup(word_embedding,self.rev)
        #h_rev = tf.div(tf.reduce_sum(vt,axis=1),tf.cast(self.mask,dtype=tf.float32)) + common_rev
        h_rev = tf.reduce_mean(vt,axis=1) + common_rev
        W = tf.Variable(tf.random_normal(shape=(self.num_factor,1),stddev=0.01))
        r_hat = tf.reduce_sum(tf.matmul(tf.sigmoid(h_rev),W) + bu + bi + global_bias,axis=1)
        self.y = tf.reduce_sum(tf.matmul(tf.sigmoid(ei-eu),W) + bu + bi + global_bias,axis=1)
        mse = tf.reduce_mean(tf.square(r_hat-self.rating))
        transloss = tf.nn.l2_loss(eu + h_rev - ei)
        regloss = tf.nn.l2_loss(W) + tf.nn.l2_loss(common_rev) + tf.nn.l2_loss(ei) + tf.nn.l2_loss(eu) + tf.nn.l2_loss(vt)
        self.loss = mse + self.alpha*transloss + self.mu*regloss
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.learner = optimizer.minimize(self.loss)
    def train(self):
        loss,count = 0,0
        for user_batch, item_batch, doc_batch, mask_batch, rating_batch in self.ds.generate_train_batch(self.batch_size):
            feed_dict = {self.user:user_batch,self.item:item_batch,self.rev:doc_batch,self.mask:mask_batch,self.rating:rating_batch}
            _,nloss = self.sess.run([self.learner,self.loss],feed_dict=feed_dict)
            loss += nloss * len(user_batch)
            count += len(user_batch)
        loss = loss/count
        rs,rs_ = [],[]
        for user_batch, item_batch, doc_batch, mask_batch, rating_batch in self.ds.generate_train_batch(self.batch_size):
            feed_dict = {self.user: user_batch, self.item: item_batch,self.rating: rating_batch}
            r = self.sess.run(self.y, feed_dict=feed_dict)
            rs_ += list(r)
            rs += rating_batch
        err = np.array(rs_) - np.array(rs)
        mse = np.mean(np.square(err))
        mae = np.mean(np.abs(err))
        print('Trainning,loss:{},mse:{},mae:{}'.format(loss,mse,mae))
    def test(self):
        rs, rs_ = [], []
        for user_batch, item_batch, rating_batch in self.ds.generate_test_batch(self.batch_size):
            feed_dict = {self.user: user_batch, self.item: item_batch,self.rating: rating_batch}
            r = self.sess.run(self.y, feed_dict=feed_dict)
            rs_ += list(r)
            rs += rating_batch
        err = np.array(rs_) - np.array(rs)
        mse = np.mean(np.square(err))
        mae = np.mean(np.abs(err))
        if mse < self.test_mse:
            self.test_mse = mse
        if mae < self.test_mae:
            self.test_mae = mae
        print('Testing,mse:{},mae:{}'.format(mse, mae))
def main():
    ds = DataSet('./data/baby/baby_',300)
    batch_size,num_factor = 64,16
    epoch = 1000
    sess = tf.Session()
    model = TransRev(sess,ds,batch_size,num_factor,1,0.01,0.005)
    init = tf.global_variables_initializer()
    sess.run(init)
    for n in range(epoch):
        print('epoch '+str(n+1))
        print('-'*50)
        model.train()
        model.test()
    print('best mse:{},best mae:{}'.format(model.test_mse,model.test_mae))
if __name__ == '__main__':
    main()

