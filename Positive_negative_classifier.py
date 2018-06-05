import tensorflow as tf
import numpy as np
from Positive_negative_classification import assemble

train_x, train_y, test_x, test_y = assemble('D:/user/Documents/Python codes/Datasets/pos.txt', 'D:/user/Documents/Python codes/Datasets/neg.txt')

n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 1000
n_nodes_hl4 = 1000

n_classes = 2 # positive & negative
batch_size = 100

x = tf.placeholder(tf.float32, [None, len(train_x[0])], name='x')
y = tf.placeholder(tf.float32, name='y')

def ann(data):
    weights = {'hl1':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
               'hl2':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
               'hl3':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
               'hl4':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
               'out':tf.Variable(tf.random_normal([n_nodes_hl4, n_classes]))}

    biases = {'hl1':tf.Variable(tf.random_normal([n_nodes_hl1])),
              'hl2':tf.Variable(tf.random_normal([n_nodes_hl2])),
              'hl3':tf.Variable(tf.random_normal([n_nodes_hl3])),
              'hl4':tf.Variable(tf.random_normal([n_nodes_hl4])),
              'out':tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, weights['hl1']), biases['hl1'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, weights['hl2']), biases['hl2'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, weights['hl3']), biases['hl3'])
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3, weights['hl4']), biases['hl4'])
    l4 = tf.nn.relu(l4)

    output = tf.add(tf.matmul(l4, weights['out']), biases['out'])
    return output

def train(data):
    logits = ann(data)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    saver = tf.train.Saver()

    hm_epochs = 15
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})
                epoch_loss = epoch_loss + c
                i = i + batch_size
            print('Epoch', epoch+1, 'completed out of', hm_epochs, ', loss:', epoch_loss)
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        print('Accuracy:', sess.run(accuracy, feed_dict={x:test_x, y:test_y}))
        path = saver.save(sess, 'D:/user/Documents/Python codes/ANN_codes/Amygdala_model/Amygdala_model', global_step=15)
        print('\nSuccessfully saved to: ' + path)

train(x)
