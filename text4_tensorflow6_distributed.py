#配置分布式tensorflow
# 分布式tensorflow的角色和原理
# ps:作为分布式训练的服务端，等待各个终端来连接
# worker：作为分布式训练的运算终端
#chief supervisors：在众多终端中必须保存一个作为主要的运算终端。
#该终端在运算终端中，最先启动，保存各个运算终端运算后的所有学习参数，将其保存或者载入
#在实际运行中，各个角色的网络构建部分代码必须100%相同，三者的分工如下
# 服务端作为一个多方协调者，等待各个原酸终端来连接
# chief supervisors会在启动是统一管理全局的学习参数，进行初始化或从模型载入，
# tensorboard可视化中的summary日志等任何参数信息，整个过程都是通过RPC协议来通信的

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
plotdata = { "batchsize":[], "loss":[] }
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


#生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声
#图形显示
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

tf.reset_default_graph()
# 1.为每个角色添加IP地址和端口，创建server
print("====1.为每个角色添加IP地址和端口，创建server====")
# （1）定义IP和端口#host主机
print("(1)定义IP和端口")
strps_hosts = "localhost:1681"
#本机器域名写法等同于127.0.0.1（本机IP）
strworker_hosts = "localhost:1682,localhost:1683"
# (2)定义角色名称
print("(2)定义角色名称")
strjob_name = "ps"
task_index = 0#task任务序列索引
#(3)将字符串转成数组
print("(3)将字符串转成数组")
ps_hosts = strps_hosts.split(',')
worker_hosts = strworker_hosts.split(',')
# (4)ClusterSpec特殊集群
print("(4)ClusterSpec特殊集群，将分布式服务端和运算终端整合在一起调用")
# 将分布式服务端和运算终端整合在一起调用
cluster_spec = tf.train.ClusterSpec({'ps':ps_hosts,'worker':worker_hosts})
# (5)创建server
print("(5)创建server，将创建的主机域名和工作内容，和任务索引放入，默认serverstart = True")
# Creates a new server with the given definition
# def __init__(self,
#                server_or_cluster_def,
#                job_name=None,
#                task_index=None,
#                protocol=None,
#                config=None,
#                start=True):
server = tf.train.Server({'ps':ps_hosts,'worker':worker_hosts},
                         job_name=strjob_name,task_index = task_index)
# 2.为ps角色添加等待函数
if strjob_name =="ps":
    print("wait")
    # join：Blocks until the server has shut down.
    # This method currently blocks forever.
    server.join()
    #ps角色使用server.join函数将线程挂起，开始接收连接消息
# 3.创建网络结构
# device:使用默认的图形包装器Wrapper for `Graph.device()` using the default graph.
with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d"%task_index,cluster=cluster_spec)):
    # replica_device_setter返回副本构建所需要的设备函数
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    #模型参数
    w = tf.Variable(tf.random_normal([1]),name="weight")
    b = tf.Variable(tf.zeros([1],name = "bias"))
    # 获取迭代次数
    global_step = tf.train.get_or_create_global_step()
    # 返回并创建全局步骤张量(如果需要)。
    #前向结构
    z = tf.multiply(X,w)+b
    tf.summary.histogram("z",z)
    #反向优化
    cost = tf.reduce_mean(tf.square(Y-z))
    tf.summary.scalar("loss_function",cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost,global_step = global_step)
    saver = tf.train.Saver(max_to_keep=1)
    merged_summary_op = tf.summary.merge_all()#合并所有summary
    init = tf.global_variables_initializer()
# 4.创建Supervisor，管理session
training_epochs = 20
display_step = 2
sv = tf.train.Supervisor( #Create a `Supervisor`#
                    is_chief=(task_index == 0),
                    logdir="log/distributed/",
                    init_op = init,
                    summary_op=None,
                    saver = saver,
                    global_step = global_step,
                    save_model_secs=5
                    )
#连接目标角色创建session
with sv.managed_session(server.target) as sess:

# 在tf.train.Supervisor函数中，is_chief表明了是否为chief supervisors角色。
# 这里将task_index = 0的worker设置为chief supervisors
# logdir 为检查点文件和summary文件保存的路径
# init_op表示使用初始化变量的函数
# target:目标
# 5.迭代训练
    print("sess ok")
    print(global_step.eval(session=sess))
    for epoch in range(global_step.eval(session=sess),training_epochs*len(train_X)):
        for (x,y) in zip(train_X,train_Y):
            _,epoch = sess.run([optimizer,global_step],feed_dict={X:x,Y:y})
            # 生成summary
            summary_str = sess.run(merged_summary_op,feed_dict={X:x,Y:y});
            # 将summary写入文件
            sv.summary_computed(sess,summary_str,global_step=epoch)
            if epoch % display_step ==0:
                loss = sess.run(cost,feed_dict = {X:train_X,Y:train_Y})
                print("Epoch:", epoch + 1, "cost=", loss, "W=", sess.run(W),
                      "b=", sess.run(b))
                if not (loss =="NA"):
                    plotdata["batchsize"].append(epoch)
                    plotdata["loss"].append(loss)
    print("Finished!")
    sv.saver.save(sess,"log/minist_with_summaries/"+"sv.cpk",global_step=epoch)
sv.stop()
# 6.建立worker文件
# 7.部署运行