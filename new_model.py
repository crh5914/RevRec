class TransMemoryCNN:
    def __init__(self,sess,num_users,num_items,num_mem,embedding_size,dim,filter_sizes,num_filters,vocab_size,doc_length,layer_size,lr,l2_reg_lambda,alpha):
        self.sess = sess
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.dim = dim
        self.num_mem = num_mem
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.layer_size = layer_size
        self.vocab_size = vocab_size
        self.doc_length = doc_length
        self.lr = lr
        self.l2_reg_lambda = l2_reg_lambda
        self.alpha = alpha
        self.build_up()
    def build_up(self):
        self.user = tf.placeholder(shape=[None,],dtype=tf.int32)
        self.doc = tf.placeholder(shape=[None,self.doc_length],dtype=tf.int32)
        self.mask = tf.placeholder(shape=[None,],dtype=tf.int32) 
        self.item = tf.placeholder(shape=[None,],dtype=tf.int32)
        self.rating = tf.placeholder(shape=[None,],dtype=tf.float32)
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        #l2_loss = tf.constant(0.0)
        # embedding layer
        self.word_embedding = tf.Variable(tf.random_uniform(shape=(self.vocab_size, self.embedding_size),minval=-0.1,maxval=0.1))
        user_embedding = tf.Variable(tf.truncated_normal(shape=(self.num_users,self.dim),stddev=0.01))
        item_embedding = tf.Variable(tf.truncated_normal(shape=(self.num_items,self.dim),stddev=0.01))
        memory = tf.Variable(tf.truncated_normal(shape=(self.num_mem,self.dim),stddev=0.01))
        # embedding lookup
        doc_vec = tf.nn.embedding_lookup(self.word_embedding, self.doc)
        user_vec = tf.nn.embedding_lookup(user_embedding,self.user)
        item_vec = tf.nn.embedding_lookup(item_embedding,self.item)
        uv = tf.concat([user_vec,item_vec],axis=1)
        key_hidden = tf.layers.dense(uv,self.num_mem,activation=tf.nn.tanh,name='key_hidden_layer')
        raw_weights = tf.layers.dense(key_hidden,self.num_mem,name='key_out_layer')
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
            pooled = tf.layers.max_pooling2d(conv,(self.doc_length - filter_size + 1, 1),(1,1))
            pooled_outputs.append(pooled)
        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        doc_feat = tf.reshape(h_pool, [-1, num_filters_total])
        feat = tf.layers.dense(doc_feat,self.dim,activation=tf.nn.relu,name='transform_layer')
        x = tf.nn.dropout(feat,keep_prob=self.dropout_keep_prob)
        x_ = tf.nn.dropout(out,keep_prob=self.dropout_keep_prob)
        layer = {}
        for i,s in enumerate(self.layer_size):
            layer['layer%d'%i] = tf.layers.Dense(s,activation=tf.nn.relu,name='full_connected_layer%d'%i)
        layer['predict_layer'] = tf.layers.Dense(1,activation=tf.nn.relu,name='prediction_layer')
        for k in range(len(self.layer_size)):
            x = layer['layer%d'%k](x)
            x_ = layer['layer%d'%k](x_)
            x = tf.nn.dropout(x,keep_prob=self.dropout_keep_prob)
            x_ = tf.nn.dropout(x_,keep_prob=self.dropout_keep_prob)
        x = layer['predict_layer'](x)
        x_ = layer['predict_layer'](x_)
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.y = tf.reduce_sum(x,axis=1)
        self.y_ = tf.reduce_sum(x_,axis=1)
        self.mse = tf.reduce_mean(tf.square(tf.subtract(self.rating,self.y)))
        self.loss = (1-self.alpha)*tf.nn.l2_loss(tf.subtract(self.rating,self.y)) + self.alpha*tf.nn.l2_loss(tf.subtract(feat,out)) + self.l2_reg_lambda*reg_loss
        self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
