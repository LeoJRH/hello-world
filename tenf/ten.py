import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# add

# a = tf.constant(5.0)
# b = tf.constant(6.0)
#
# sum1 = tf.add(a,b)
# print(sum1)
#
# plt = tf.placeholder(tf.float32,[2,3,4])
# print(plt)
#
#
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#     # print(sess.run(plt, feed_dict={plt:[[1,2,3],[4,5,36],[2,3,8]]}))
#     # print(a.graph)
#     print("-------------")
#     print(a.shape)
#     print(plt.shape)
#     print("-------------")
#     print(a.name)
#     print("----------------")
#     print(a.op)
#shape:
#0:()
#1:(5)
#2:(5,6)
#3:(2,3,4)

# plt = tf.placeholder(tf.float32,[None,2])
#
# print(plt)
#
#
# plt.set_shape([3,2])
# print(plt)
# plt_reshape = tf.reshape(plt,[1,6]) #can't change elements like [3,3] or [1,2,3],not same
#
#
# print(plt_reshape)


# variable value op can be keep all time
# define variable op ,First init in session
a = tf.constant(3.0)

b = tf.constant(4.0)

c = tf.add(a,b)
var = tf.Variable(tf.random_normal([2,3],mean=0.0,stddev=1.0))

print(a, var)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    #    OP---init
    sess.run(init_op)


    #put graph in events

    filewrite = tf.summary.FileWriter("./",graph=sess.graph)
    print(sess.run([c, var]))



