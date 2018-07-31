import tensorflow as tf
# division分割或者segment,分割相关函数
x = tf.constant([[1,2,3,4],[-1,-2,-3,-4],[5,6,7,8]])
y = tf.constant([0,0,1])

with tf.Session() as sess:
    #按照指定的维度，分割张量data中的值，将x按照[0,0,1]的维度来分割
    print(sess.run(tf.segment_sum(x,y)))
    print(sess.run(tf.segment_prod(x,y)))
    #根据分段计算各个片段的最小值
    print(sess.run(tf.segment_min(x,y)))
    print(sess.run(tf.segment_max(x,y)))
    print(sess.run(tf.segment_mean(x,y)))
    # 下面是id顺序无序的函数
    print(sess.run(tf.unsorted_segment_sum(x,y)))

    # print(sess.run(tf.sparse_segment_sum(x,y)))
    # print(sess.run(tf.segment_min(x,y)))
