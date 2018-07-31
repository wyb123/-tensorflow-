import tensorflow as tf
x = 2
L = [[2,32,1],
     [2,1,1]]
tensor = [[1,2,3],[4,5,6]]
with tf.Session() as sess:
    #1.tensor的类型操作
    print(tf.to_float(x,name = "ToFloat"))
    print(sess.run(tf.string_to_number(string_tensor="12",out_type=None,name = None)))
    print(sess.run(tf.to_double(x,name="ToDouble")))
    print(sess.run(tf.to_float(x,name= "ToFloat")))
    #将x转成dtype所说的类型
    print(sess.run(tf.cast(x,dtype = "float",name = None)))
    #2.tensor的数值操作
    print(sess.run(tf.ones(shape=(3,2),dtype=tf.int16)))
    print(sess.run(tf.ones([3,2],tf.int16)))
    print(sess.run(tf.zeros([3, 4], tf.int32)))
    print(sess.run(tf.ones_like(L)))
    print(sess.run(tf.zeros_like(tensor)))
    print(sess.run(tf.fill([3,2],2)))
    print(sess.run(tf.constant(1,shape =[2,3])))
    #正太分布随机数，均值mean，标准差stddev
    print(sess.run(tf.random_normal([2,3],mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name = 'baobao')))
    #截断正太分布随机数，均值mean,标准差stddev，只保留【mean-2*stddev,mean+2*stddev】范围内的随机数
    print(sess.run(tf.truncated_normal([2,3],mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name = "tongxue")))
    #均匀分布随机数，范围为[minval,maxval]
    print(sess.run(tf.random_uniform([2,3],minval=0,maxval=None,dtype=tf.float32,seed=None,name = 'baboa1')))
    # 将输入值按照size尺寸随机剪辑
    # print(sess.run(tf.random_crop(L,size=2,seed=None,name = None)))
    # 设置随机数种子
    # print(sess.run(tf.set_random_seed(seed=2)))
    print(sess.run(tf.linspace(1.0,5.0,5)))
    # delta=0.5随机数设置增量为0.5，即每0.5值增加一次
    print(sess.run(tf.range(1,5,delta=0.5)))
    print(sess.run(tf.range(1,5)))#默认delta= 1

# a = tf.random_uniform([1])
# b = tf.random_normal([1])
#
# print("Session 1")
# with tf.Session() as sess1:
#     print(sess1.run(a))  # generates 'A1'
#     print(sess1.run(a))  # generates 'A2'
#     print(sess1.run(b))  # generates 'B1'
#     print(sess1.run(b))  # generates 'B2'
