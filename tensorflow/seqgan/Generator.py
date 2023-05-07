import tensorflow as tf

class Generator(object):

    def __init__(self, config):
        """ Basic Set up

        Args:
           num_emb: output vocabulary size
           batch_size: batch size for generator
           emb_dim: LSTM hidden unit dimension
           sequence_length: maximum length of input sequence
           start_token: special token used to represent start of sentence
           initializer: initializer for LSTM kernel and output matrix
        """
        self.num_emb = config.num_emb
        self.batch_size = config.gen_batch_size
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.sequence_length = config.sequence_length
        self.start_token = tf.constant(config.start_token, dtype=tf.int32, shape=[self.batch_size])
        self.initializer = tf.random_normal_initializer(stddev=0.1)

    '''
    构建生成器的输入
    '''
    def build_input(self, name):
        """ Buid input placeholder

        Input:
            name: name of network
        Output:
            self.input_seqs_pre (if name == pretrained)
            self.input_seqs_mask (if name == pretrained, optional mask for masking invalid token)
            self.input_seqs_adv (if name == 'adversarial')
            self.rewards (if name == 'adversarial')
        """
        assert name in ['pretrain', 'adversarial', 'sample']
        if name == 'pretrain':
            self.input_seqs_pre = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_seqs_pre")
            self.input_seqs_mask = tf.placeholder(tf.float32, [None, self.sequence_length], name="input_seqs_mask")
        elif name == 'adversarial':
            self.input_seqs_adv = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_seqs_adv")
            self.rewards = tf.placeholder(tf.float32, [None, self.sequence_length], name="reward")

    '''
    生成器预训练结构，使用交叉损失进行训练
    '''
    def build_pretrain_network(self):
        """ Buid pretrained network

        Input:
            self.input_seqs_pre
            self.input_seqs_mask
        Output:
            self.pretrained_loss
            self.pretrained_loss_sum (optional)
        """
        # 建立预训练网络，使用MLE交叉熵损失
        self.build_input(name='pretrain')
        self.pretrained_loss = 0.0
        
        with tf.variable_scope('teller'):   # teller 变量作用域
            with tf.variable_scope('lstm'): # lstm 变量作用域
                '''
                构建 LSTM 结构

                num_units 是LSTM单元中的单元数
                        这个参数的大小就是LSTM输出的向量的维度
                state_is_tuple 表示是否返回(c_state, m_state)
                        当值为True时，接受并返回的是状态二元组
                '''
                lstm1 = tf.nn.rnn_cell.LSTMCell(
                    num_units=self.hidden_dim, 
                    state_is_tuple=True
                )

            with tf.variable_scope('embedding'):
                '''
                构建词向量矩阵

                输入的词向量，其实是一个句子的编码
                '''
                embedding = tf.get_variable(
                    name='embedding', 
                    shape=[self.num_emb, self.emb_dim], 
                    initializer=self.initializer
                ) 
        pass

    '''
    生成器对抗训练结构，使用Policy Gradient反向传播奖励
    '''
    def build_adversarial_network(self):
        pass

    '''
    利用生成器生成样例序列数据
    '''
    def build_sample_network(self):
        pass

    '''
    构建完整的生成器
    '''
    def build(self):
        self.build_pretrain_network()
        self.build_adversarial_network()
        self.build_sample_network()

    '''
    生成样例序列数据
    '''
    def generate(self, sess):
        # 生成样例序列数据
        return sess.run(self.sample_word_list_reshape)