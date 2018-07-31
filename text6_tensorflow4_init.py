import tensorflow as tf
value = 1.0
x,y = 1.0,2.0
# 初始化学习参数
# zeros_initializer = Zeros
# ones_initializer = Ones
# constant_initializer = Constant
# random_uniform_initializer = RandomUniform
# random_normal_initializer = RandomNormal
# truncated_normal_initializer = TruncatedNormal
# uniform_unit_scaling_initializer = UniformUnitScaling
# variance_scaling_initializer = VarianceScaling
# orthogonal_initializer = Orthogonal
# 初始化一切提供的值
a = tf.constant_initializer(value)
# 从x到y均匀初始化
#uniform统一的,相同的均衡的
b = tf.random_uniform_initializer(x,y)
# 用所给平均值和标准差初始化均匀分布
c = tf.random_normal_initializer(mean,stddev)#uniform统一的,相同的均衡的
# 正太随机分布
b = tf.random_normal_initializer(mean=0.0,stddev=1.0,seed=None,dtype = tf.float32)
# 截断正太分布随机数
b = tf.truncated_normal_initializer(mean=0.0,stddev=1.0,seed=None,dtype=tf.float32)
# 均匀分布随机数
b = tf.random_uniform_initializer(minval=0,maxval=None,seed=None,dtype=tf.float32)
# 满足均匀分布，但不影响输出数量级的随机数
b = tf.uniform_unit_scaling_initializer(factor=1.0,seed=None,dtype=tf.float32)
# 初始化为1
b = tf.zeros_initializer(shape,dtype = tf.float32,partition_info = None)
# 生成正交矩阵的随机数，当需要生成的参数是二维时，这个正交矩阵是由均匀分布的随机数矩阵经过svd分解而来
b = tf.orthogonal_initializer(gain=1.0,dtype=tf.float32,seed=None)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(a)
    print(b)

  # print(c)
    # print(d)
    # print(e)
    # print(f)
    # print(g)
