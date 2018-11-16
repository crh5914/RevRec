import tensorflow as tf

class SimpleModel:
    def __init__(self,sess,num_users,num_items,length,vocab_size,dim,layer_size,lr,alpha):
        self.sess = sess
        self.num_users = num_users
        self.num_items = num_items
        self.length = length
        self.vocab_size = vocab_size
        self.dim = dim
        self.layer_size = layer_size
        self.lr = lr
        self.alpha = alpha
        self.build_up()
    def build_up(self):
        self.user = tf.placeholder(shape=[None,],dtype=tf.int32)
        self.review = tf.placeholder(shape=[None,self.length],dtype=tf.int32)
        self.rev_length = tf.placeholder(shape=[None,1],dtype=tf.int32)
        self.item = tf.placeholder(shape=[None,],dtype=tf.int32)
        self.rating = tf.placeholder(shape=[None,],dtype=tf.float32)
        self.phase = tf.placeholder(tf.bool)
        self.word_emb = tf.Variable(tf.random_normal(shape=(self.vocab_size,self.dim),stddev=0.1),name='word_emb')
        padd_emb = tf.constant(0.0,shape=(1,self.dim))
        word_table = tf.concat([padd_emb,self.word_emb],axis=0)
        user_emb = tf.Variable(tf.random_normal(shape=(self.num_users,self.dim),stddev=0.1),name='user_emb')
        item_emb = tf.Variable(tf.random_normal(shape=(self.num_items,self.dim),stddev=0.1),name='item_emb')
        uvec = tf.nn.embedding_lookup(user_emb,self.user)
        itvec = tf.nn.embedding_lookup(item_emb,self.item)
        senvec = tf.nn.embedding_lookup(word_table,self.review)
        revfeat = tf.reduce_mean(senvec,axis=1)
        #revfeat = tf.div(,tf.cast(self.rev_length,dtype=tf.float32))
        W = tf.Variable(tf.random_normal(shape=(self.dim,self.dim),stddev=0.1))
        b = tf.Variable(tf.constant(0.0,shape=(self.dim,)))
        transvec = tf.subtract(uvec,itvec)
        feat = tf.nn.tanh(tf.add(tf.matmul(revfeat,W),b))
        prev_dim = self.dim
        fc_W = {}
        layer = feat
        pred_layer = transvec
        for i,s in enumerate(self.layer_size):
            if i > 0:
                prev_dim = self.layer_size[i-1]
            fc_W['layer_W'+str(i)] = tf.Variable(tf.random_normal(shape=(prev_dim,s),stddev=0.1))
            fc_W['layer_b'+str(i)] = tf.Variable(tf.constant(0.0,shape=(s,))) 
            layer = tf.nn.relu(tf.add(tf.matmul(layer,fc_W['layer_W'+str(i)]),fc_W['layer_b'+str(i)]))
            pred_layer = tf.nn.relu(tf.add(tf.matmul(pred_layer,fc_W['layer_W'+str(i)]),fc_W['layer_b'+str(i)]))
        transloss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(transvec,feat)),axis=1))
        self.y_ = tf.reduce_sum(layer,axis=1)
        self.test_y_ = tf.reduce_sum(pred_layer,axis=1)
        self.mse = tf.reduce_mean(tf.square(tf.subtract(self.y_,self.rating))) 
        self.test_mse = tf.reduce_mean(tf.square(tf.subtract(self.test_y_,self.rating)))   
        self.loss = (1-self.alpha)*transloss + self.alpha*self.mse
        self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
    def train(self,us,its,revs,revlen,rs):
        feed_dict = {self.user:us,self.review:revs,self.rev_length:revlen,self.item:its,self.rating:rs,self.phase:False}
        _,mse,loss = self.sess.run([self.train_opt,self.mse,self.loss],feed_dict=feed_dict)
        return mse,loss
    def test(self,us,its,rs):
        feed_dict = {self.user:us,self.item:its,self.rating:rs,self.phase:True}
        y_,mse = self.sess.run([self.test_y_,self.test_mse],feed_dict=feed_dict)
        return y_,mse

