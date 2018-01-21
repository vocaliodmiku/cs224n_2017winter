import tensorflow as tf
from addition import *
import numpy as np
from tqdm import tqdm
import time


length = 6  # 限制每一个评论的长度
target_lenth = 10 #每一个目标的长度
hidden_size = 300  # 隐藏层的神经元个数
embedding_size = 300 # 词向量的维度
learning_rate = 0.01  # 梯度下降的学习率
batch_size = 6  # 样本的大小
keep_prob = 0.5



"""载入词向量 上下文 目标文本数据"""

embeddings_var = np.loadtxt('glovec.txt')  # 载入预训练好的词向量矩阵
embeddings_var = embeddings_var.astype(np.float32)


def load_data(filename1, filename2, filename3):
    context_data = []
    with open(filename1, "r") as f1:
        c_data_list = f1.readlines()
        for i in c_data_list:
            tep = list(map(int, i.split(' ')[:len(i) + 1]))
            context_data.append(tep)  # 载入编号好的文本

    tartget_data = []
    with open(filename2, "r") as f2:
        t_data_list = f2.readlines()
        for i in t_data_list:
            tep = list(map(int, i.split(' ')[:len(i) + 1]))
            tartget_data.append(tep)  # 载入编号好的目标

    Label_data = np.loadtxt(filename3)  # 载入情感极性文本
    Label_data = Label_data.astype(int)
    return context_data, Label_data, tartget_data


(X_train, Y_train, T_train) = load_data('context_data_3602.txt', 'target_data_3602.txt', 'polarity_3602.txt')
(X_test, Y_test, T_test) = load_data('context_data_1120.txt', 'target_data_1120.txt', 'polarity_1120.txt')


"""上下文和target进行补零操作"""
x_train, y_train, target_train = Rand(X_train, Y_train, T_train)
x_test, y_test, target_test = Rand(X_test, Y_test, T_test)

x_train, x_train_len = zero_padding(x_train, length) # 对训练数据和测试数据分别进行补零，长度为20
x_test, x_test_len = zero_padding(x_test, length)


t_train = padding_target(target_train)                   # 对目标数据分别进行补零
t_test = padding_target(target_test)


"""定义占位符"""
batch_sample = tf.placeholder(tf.int32, shape=[None, length])
batch_target = tf.placeholder(tf.int32, shape=None)

#batch_target = tf.placeholder(tf.int32, shape=None)

seq_len_ph = tf.placeholder(tf.int32, shape=None)  # 这个是评论文本的实际长度
y = tf.placeholder(tf.int32, shape=[None, 3])      # 将真实标签传进去,三个类别

# 需要修改的文件位置tf.nn.rnn_cell_impl
"""初始化词向量"""
with tf.name_scope('embedding'):
    batch_Embedded = tf.nn.embedding_lookup(embeddings_var, batch_sample)  # 取出词向量
    batch_Target = tf.nn.embedding_lookup(embeddings_var, batch_target)  # 取出目标向量
    batch_Target = tf.reduce_mean(batch_Target, 1) #目标求平均
    batch_Target = tf.reshape(batch_Target,[batch_size,embedding_size]) # 将目标向量转置

"""定义双向RNN模型"""
with tf.name_scope('rnn_model'):
    # rnn_outputs, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.BasicLSTMCell(hidden_size,target_word=batch_Target),
    #                                                  inputs=batch_Embedded, sequence_length=seq_len_ph,
    #                                                  dtype=tf.float32)                                                  #rnn_outputs是[batch_size, seq_length, hidden_size]

    rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        tf.contrib.rnn.BasicLSTMCell(hidden_size, target_word=batch_Target),
        tf.contrib.rnn.BasicLSTMCell(hidden_size, target_word=batch_Target),
        inputs=batch_Embedded, sequence_length=seq_len_ph,
        dtype=tf.float32)
    outputs = tf.concat(rnn_outputs, 2)  # 如果是双向RNN，返回是一个tuple形式。那么就需要连接起来
    mean_hidden = tf.reduce_mean(outputs, 1)  # 将250个时间序列进行求平均值
with tf.name_scope('fully_connected'):
        # drop_hidden = tf.nn.dropout(mean_hidden, 0.5)
        # print(drop_hidden.shape)
    W = tf.Variable(tf.random_normal([2 * hidden_size, 3]))  # 随机初始化W权重
    b = tf.Variable(tf.random_normal([3]))  # 随机初始化b权重
    y_hat = tf.nn.xw_plus_b(mean_hidden, W, b)
inti5 = tf.global_variables_initializer()

"""训练与测试"""
with tf.Session() as sess:
    sess.run(inti5)
    train_data = [[1,2,3,4,5,6],
                  [2,4,5,6,7,8],
                  [3,4,5,6,7,8],
                  [0,0,0,0,0,0],
                  [0,0,0,0,0,0],
                  [0,0,0,0,0,0]]
    train_target = [1,1,1,0,0,0]
    train_seq = [6,6,6,0,0,0]
    loss_train = sess.run(mean_hidden,feed_dict={batch_sample: train_data, seq_len_ph: train_seq, batch_target: train_target})
