import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# Graph:show the memory token
a = tf.constant(5.0)
b = tf.constant(6.0)

sum1 = tf.add(a,b)

graph = tf.get_default_graph()
print(graph)

# var1 = 2
# var2 = 3
# sum2 = var1+var2 #Wrong example value define
#Not op value

var1 = 2.0
sum2 = a+var1 #reload + and make it op
print(sum2)
with tf.Session() as sess:
    print(sess.run([sum2]))
    # print(a.graph)
    # print(sum1.graph)
    # print(sess.graph)







#Build a graph :with

# g = tf.Graph()
# print(g)
# with g.as_default():
#     c = tf.constant(11.0)
#     print(c.graph)
#op:tensorflow API is op
#tensor: represent data
