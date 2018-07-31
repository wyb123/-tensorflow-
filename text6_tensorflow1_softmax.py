import tensorflow as tf
labels = [[0,0,1],[0,1,0]]
logits = [[2,0.5,6],[0.1,0,3]]
# 将交叉熵经过两次softmax
logits_scaled = tf.nn.softmax(logits)
logits_scaled2  = tf.nn.softmax(logits_scaled)
result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
result2 = tf.nn.softmax_cross_entropy_with_logits(labels = labels,logits = logits_scaled)
# 自建公式实验:将做两次softmax的值放到自建组合的公式里得到正确的值
result3 = -tf.reduce_sum(labels*tf.log(logits_scaled),1)

with tf.Session() as sess:
    print("scaled = ",sess.run(logits_scaled))
    print("scaled2 = ",sess.run(logits_scaled2))

    print("rel1 =",sess.run(result1),"\n")
    # 正确的方式
    print("rel2 = ",sess.run(result2),"\n")
    print("rel3 = ",sess.run(result3),"\n")
