import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#1.Prepare x_train and target
#2.Build model w and b
#3.Loss function
#4.gradient descent


#Build myregression---


def myregression():

    with tf.variable_scope("data"): #like define func to see func of a part

        x = tf.random_normal([100,1],mean=1.75,stddev=0.5,name="x_data")

        #matrix multiplication has to be two-dimensional
        y_true = tf.matmul(x,[[0.9]]) + 2 #JUST DEFINE first as data

    #Build regression model- one y =w.x + b
    #Random value to
    #trainable:default true,can be change
    with tf.variable_scope("model"):

        weight = tf.Variable(tf.random_normal([1,1], mean=0.0,stddev=1.0,name="Weight"))
        bias = tf.Variable(0.0, name="b")

        y_predict = tf.matmul(x,weight) + bias

     # Loss Function
    with tf.variable_scope("loss"):
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

    # gradient descent learning_rate:0~1,2,3,4,5,7,10

    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    #0.1 make affect to speed and  accuracy


    #Collect var and merge var
    tf.summary.scalar("losses",loss)

    tf.summary.histogram("weights",weight)



    #op merge
    merged = tf.summary.merge_all()



    #define op varible
    init_op = tf.global_variables_initializer()


    saver = tf.train.Saver()


    #run func by session
    with tf.Session() as sess:
        sess.run(init_op)

        #print random value at first time

        print("Random value weight: %f,Bias: %f" % (weight.eval(),bias.eval()))
        filewriter = tf.summary.FileWriter("./",graph=sess.graph)


        #load model and show para from last time

        if os.path.exists("./ckpt/checkpoint"):
            saver.restore(sess,"./ckpt/testmodel")


        for i in range(1000):
            sess.run(train_op)
            summary = sess.run(merged)

            filewriter.add_summary(summary,i)



            print("Now is weight: %f,Bias: %f" % (weight.eval(),bias.eval()))
        saver.save(sess, "./ckpt/testmodel")



if __name__ == '__main__':
    myregression()