import tensorflow as tf
with tf.name_scope('name_scope_1'):
    var1 = tf.get_variable(name='var1', shape=[1], initializer=None, dtype=tf.float32)
    var2 = tf.Variable(name='var2', initial_value=[1], dtype=tf.float32)
    var21 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    print(var1.name)
    print(var2.name)
    print(var21.name)

