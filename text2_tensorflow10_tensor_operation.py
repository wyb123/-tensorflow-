import tensorflow as tf
import numpy as np

t1 = [[[1,1,1],[2,2,2]],[[3,3,3],[4,4,4]],[[5,5,5],[6,6,6]]]
t2 = [[7,8,9],[1,2,3]]
t3 = [[1, 2, 3], [4, 5, 6]]
t4 = [[7, 8, 9], [10, 11, 12]]
y = tf.constant([0,2,-1])
t5 = [0, 2, -1, 1]
t6 = [0,1,1,2]

with tf.Session() as sess:
    #对输入数据进行切片操作
    # print(sess.run(tf.slice(t1,[1,0,0],[1,1,3])))
    # 沿着某一个维度将tensor分离成num_or_size_splits
    # Value是一个shape为[5,30]的张量
    # 沿着第一列将value按[4,15,11]分成3个张量
    # tf.split()
    # print(sess.run(tf.concat([t1,t2]),axis = 0))
    print(sess.run(tf.concat([t3, t4], 0)))#0为纵向，1为横向
    print(sess.run(tf.concat([t3, t4], 1)))#0为纵向，1为横向
    print(sess.run(tf.stack([t3,t4],1)))#合并类似上面的
    # print(sess.run(tf.unstack()))#拆分
    print("====================")
    print(sess.run(tf.gather(y,[2,0])))
    print("=============================")
    print(sess.run(tf.one_hot(t5, 3, 5.0, off_value=None,
            axis=-1)))
    # 统计非零个数
    print("=========================")
    print(sess.run(tf.count_nonzero(t6,axis=0)))

