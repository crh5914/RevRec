import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from gensim.models import Word2Vec
import pickle
from nltk.corpus import stopwords
from tqdm import tqdm
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
    parser.add_argument("--l2_reg_lambda", type=float,default=1e-3, help="L2 regularizaion lambda")
    parser.add_argument("--lr", type=float,default=0.001, help="learning rate")
    parser.add_argument("--alpha", type=float,default=0.9, help="loss balance alpha")
    parser.add_argument("--batch_size",type=int,default=128,help="Batch Size ")
    parser.add_argument("--num_epochs", type=int,default=40, help="Number of training epochs ")
    return parser.parse_args()

class Corpus:
    def __init__(self, prefix, threshold=0.8, filter_stop=True):
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
        self.train_users = {}
        self.train_items = {}
        with open(self.prefix + '_train.csv', 'r') as fp:
            for line in fp:
                vals = line.strip().split(',')
                u, i, r = int(vals[0]), int(vals[1]), float(vals[3])
                self.num_users = max(u, self.num_users)
                self.num_items = max(i, self.num_items)
                nline = [u, i, r]
                self.train_users[u-1] = self.train_users.get(u-1,0) + 1
                self.train_items[i-1] = self.train_items.get(i-1,0) + 1
                words = vals[2].strip().split(' ')
                self.docs.append(words)
                self.train_data.append(nline)
        with open(self.prefix + '_test.csv', 'r') as fp:
            for line in fp:
                vals = line.strip().split(',')
                u, i, r = int(vals[0]), int(vals[1]), float(vals[3])
                self.num_users = max(u, self.num_users)
                self.num_items = max(i, self.num_items)
                nline = [u, i, r]
                self.test_data.append(nline)

    def build_up(self):
        self.load()
        self.build_dict()
        self.map_word()
        self.padding_sentence()
    def build_dict(self):
        d = {}
        for doc in self.docs:
            for word in doc:
                d[word] = d.get(word, 0) + 1
        vals = sorted(d.values(), reverse=True)
        threshold = vals[int(self.threshold * len(vals))]
        self.d = {}
        for word in d:
            if d.get(word, 0) > threshold and word not in self.stop_words:
                self.d[word] = d[word]
    def map_word(self):
        self.word2idx = {w: i for i, w in enumerate(self.d)}
        self.num_words = len(self.word2idx)
        self.idx2word = {i: w for i, w in enumerate(self.d)}
        self.numeric_docs = []
        #self.max_len = 0
        lens = []
        for doc in self.docs:
            words = [self.word2idx[word] for word in doc if word in self.word2idx]
            lens.append(len(words))
            #if len(words) > self.max_len:
            #    self.max_len = len(words)
            self.numeric_docs.append(words)
        lens = sorted(lens)
        self.max_len = lens[int(0.95 * len(lens))]
        print('word dict size:{},max length:{}'.format(len(self.word2idx), self.max_len))

    def padding_sentence(self):
        docs = []
        self.length = []
        for doc in self.numeric_docs:
            if len(doc) > self.max_len:
                doc = doc[:self.max_len]
                self.length.append(self.max_len)
                docs.append(doc)
            else:
                self.length.append(len(doc))
                doc = doc + [len(self.word2idx)] * (self.max_len - len(doc))
                docs.append(doc)
        self.numeric_docs = docs

    def get_train_batch(self, batch_size):
        self.train_data,self.numeric_docs = shuffle(self.train_data,self.numeric_docs)
        for i in range(0, len(self.numeric_docs), batch_size):
            end = i + batch_size
            if end > len(self.numeric_docs):
                end = len(self.numeric_docs)
            batch_users, batch_docs, batch_masks, batch_items,batch_ratings = [], [],[], [], [], []
            for j in range(i, end):
                u, it, r = self.train_data[j]
                doc = self.numeric_docs[j]
                mask = self.length[j]
                batch_users.append(u - 1)
                batch_items.append(it - 1)
                batch_docs.append(doc)
                batch_masks.append(mask)
                batch_ratings.append(r)
            yield batch_users, batch_docs, batch_masks, batch_items,batch_ratings
    def generate_test_batch(self, batch_size):
        for i in range(0, len(self.test_data), batch_size):
            end = i + batch_size
            if end > len(self.test_data):
                end = len(self.test_data)
            batch_users, batch_items, batch_ratings = [], [], []
            for j in range(i, end):
                u, i, r = self.test_data[j]
                if (u-1) not in self.train_users or (i-1) not in self.train_items:
                    continue
                batch_users.append(u - 1)
                batch_items.append(i - 1)
                batch_ratings.append(r)
            yield batch_users, batch_items, batch_ratings


class MarginTransModel:
    def __init__(self, sess, ds, num_mem, embedding_size, dim, filter_sizes, num_filters, layer_size, lr, l2_reg_lambda,
                 keep_prob):
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
        self.user = tf.placeholder(shape=[None, ], dtype=tf.int32)
        self.doc = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.mask = tf.placeholder(shape=[None, ], dtype=tf.int32)
        self.item = tf.placeholder(shape=[None, ], dtype=tf.int32)
        self.rating = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.alpha = tf.placeholder(tf.float32)
        # l2_loss = tf.constant(0.0)
        # embedding layer
        self.word_embedding = tf.Variable(
            tf.random_uniform(shape=(self.ds.num_words, self.embedding_size), minval=-0.1, maxval=0.1))
        user_embedding = tf.Variable(tf.truncated_normal(shape=(self.ds.num_users, self.dim), stddev=0.01))
        item_embedding = tf.Variable(tf.truncated_normal(shape=(self.ds.num_items, self.dim), stddev=0.01))
        memory = tf.Variable(tf.truncated_normal(shape=(self.num_mem, self.dim), stddev=0.01))
        # embedding lookup
        padding_embedding = tf.constant(0.0, shape=(1, self.embedding_size))
        word_table = tf.concat([self.word_embedding, padding_embedding], axis=0)
        doc_vec = tf.nn.embedding_lookup(word_table, self.doc)
        user_vec = tf.nn.embedding_lookup(user_embedding, self.user)
        item_vec = tf.nn.embedding_lookup(item_embedding, self.item)
        # pos instance
        uv = tf.concat([user_vec, item_vec], axis=1)
        key_hidden = tf.layers.dense(uv, self.num_mem, activation=tf.nn.relu, name='key_hidden_layer')
        raw_weights = tf.layers.dense(key_hidden, self.num_mem, name='key_out_layer')
        key = tf.nn.softmax(raw_weights, axis=1)
        out = tf.matmul(key, memory)

        word_vecs = tf.expand_dims(doc_vec, -1)
        # text cnn
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            # Convolution Layer
            kernel_size = [filter_size, self.embedding_size]
            conv = tf.layers.conv2d(word_vecs, self.num_filters, kernel_size, activation=tf.nn.relu)
            # Maxpooling over the outputs
            pooled = tf.layers.max_pooling2d(conv, (self.ds.max_len - filter_size + 1, 1), (1, 1))
            pooled_outputs.append(pooled)
        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        doc_feat = tf.reshape(h_pool, [-1, num_filters_total])
        T = tf.Variable(tf.random_normal(shape=(num_filters_total,self.dim),stddev=0.01),name='T')
        bt = tf.Variable(tf.constant(0.0,shape=(self.layer_size[0],)),name='bt')
        feat = tf.nn.relu(tf.matmul(doc_feat,T) + bt)
        #feat = tf.layers.dense(doc_feat, self.dim, activation=tf.nn.relu, name='transform_layer')

        W1 = tf.Variable(tf.random_normal(shape=(self.dim,self.layer_size[0]),stddev=0.01),name='W1')
        b1 = tf.Variable(tf.constant(0.0,shape=(self.layer_size[0],)),name='b1')
        W2 = tf.Variable(tf.random_normal(shape=(self.layer_size[0],self.layer_size[1]), stddev=0.01), name='W2')
        b2 = tf.Variable(tf.constant(0.0, shape=(self.layer_size[1],)), name='b2')
        W3 = tf.Variable(tf.random_normal(shape=(self.layer_size[1],1), stddev=0.01), name='W3')
        b3 = tf.Variable(tf.constant(0.0, shape=(1,)), name='b3')

        f1 = tf.nn.relu(tf.matmul(feat,W1) + b1)
        f2 = tf.nn.relu(tf.matmul(f1, W2) + b2)
        f3 = tf.nn.relu(tf.matmul(f2, W3) + b3)

        f_1 = tf.nn.relu(tf.matmul(out, W1) + b1)
        f_2 = tf.nn.relu(tf.matmul(f_1, W2) + b2)
        f_3 = tf.nn.relu(tf.matmul(f_2, W3) + b3)
        self.y = tf.reduce_sum(f3,axis=1)
        self.y_ = tf.reduce_sum(f_3,axis=1)
        reg_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
        self.mse = tf.reduce_mean(tf.square(tf.subtract(self.rating, self.y)))
        self.transloss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(feat,out)),axis=1))
        self.loss = self.mse + self.transloss + self.l2_reg_lambda*reg_loss
        self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train(self, epochs, batch_size):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        best_mse, best_mae, best_epoch = 10, 10, 0
        mse, mae = self.test(batch_size)
        print('init,rmse:{},mae:{}'.format(np.sqrt(mse), mae))
        for epoch in range(epochs):
            batch_num = 0
            losses, tmse, count = 0.0, 0, 0
            for batch_users, batch_docs, batch_masks, batch_items,batch_ratings in tqdm(
                    self.ds.get_train_batch(batch_size)):
                feed_dict = {self.user: batch_users, self.doc: batch_docs, self.mask: batch_masks,
                             self.item: batch_items,self.rating: batch_ratings, self.dropout_keep_prob: self.keep_prob,
                             self.alpha: 0.1}
                count += len(batch_ratings)
                _, bmse, loss = self.sess.run([self.train_opt, self.mse, self.loss], feed_dict=feed_dict)
                losses += len(batch_ratings) * loss
                tmse += len(batch_ratings) * bmse
                if batch_num % 300 == 0:
                    mse, mae = self.test(batch_size)
                    print('in trainning,test_rmse:{},test_mae:{}'.format(np.sqrt(mse), mae))
            losses = losses / count
            tmse = tmse / count
            mse, mae = self.test(batch_size)
            print(
                'epoch:{},loss:{},train_mse:{},test_rmse:{},test_mae:{}'.format(epoch, losses, tmse, np.sqrt(mse), mae))
            if best_mse > mse:
                best_mse, best_mae, best_epoch = mse, mae, epoch
        print('best rmse:{},mae:{},taken at epoch:{}'.format(np.sqrt(best_mse), best_mae, best_epoch))

    def test(self, batch_size):
        # mse,mae,count = 0,0,0
        ys_, ys = [], []
        for batch_users, batch_items, batch_ratings in tqdm(self.ds.generate_test_batch(batch_size)):
            feed_dict = {self.user: batch_users, self.item: batch_items, self.rating: batch_ratings,
                         self.dropout_keep_prob: 1.0, self.alpha: 0.1}
            y_ = self.sess.run(self.y_, feed_dict=feed_dict)
            #print(y_,batch_ratings)
            ys_ += list(y_)
            ys += batch_ratings
        ys_, ys = np.array(ys_), np.array(ys)
        mse = np.mean(np.square(ys_ - ys))
        mae = np.mean(np.abs(ys_ - ys))
        return mse, mae

def main():
    args = parse_args()
    sess = tf.Session()
    prefix = './data/{}/{}'.format(args.dataset,args.dataset)
    ds = Corpus(prefix)
    ds.build_up()
    model = MarginTransModel(sess, ds, args.num_mem, args.embedding_size, args.dim, eval(args.filter_sizes), args.num_filters, eval(args.layer_size), args.lr,
                         args.l2_reg_lambda, args.keep_prob)
    model.train(args.num_epochs, args.batch_size)


if __name__ == '__main__':
    main()

