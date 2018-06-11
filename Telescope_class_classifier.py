import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

csv = 'D:/user/Documents/Python codes/Datasets/Telescope data.csv'

def read_file(file):
    data = pd.read_csv(file)
    df = pd.DataFrame(data)
    print(df.shape)
    X = df[df.columns[:9]].values
    y = df[df.columns[10]]

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    print(X.shape)
    return (X, Y)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

X, Y = read_file(csv)
X, Y = shuffle(X, Y, random_state=1)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=1)
print('Shape of train_x is ' + str(train_x.shape))
print('Shape of train_y is ' + str(train_y.shape))
print('Shape of test_x is ' + str(test_x.shape))
print('Shape of test_y is ' + str(test_y.shape))

LR = 0.01
training_epoch = 100
n_dim = X.shape[1]
print('%d' %(n_dim))
n_classes = 2

n_hl1 = 20
n_hl2 = 20
n_hl3 = 20
n_hl4 = 20

x = tf.placeholder(tf.float32, [None, n_dim])
y_ = tf.placeholder(tf.float32, [None, n_classes])

weights = {'wh1':tf.Variable(tf.random_normal([n_dim, n_hl1])),
           'wh2':tf.Variable(tf.random_normal([n_hl1, n_hl2])),
           'wh3':tf.Variable(tf.random_normal([n_hl2, n_hl3])),
           'wh4':tf.Variable(tf.random_normal([n_hl3, n_hl4])),
           'whout':tf.Variable(tf.random_normal([n_hl4, n_classes]))}

biases = {'bh1':tf.Variable(tf.random_normal([n_hl1])),
          'bh2':tf.Variable(tf.random_normal([n_hl2])),
          'bh3':tf.Variable(tf.random_normal([n_hl3])),
          'bh4':tf.Variable(tf.random_normal([n_hl4])),
          'bhout':tf.Variable(tf.random_normal([n_classes]))}

def ann(data):
    l1 = tf.add(tf.matmul(data, weights['wh1']), biases['bh1'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, weights['wh2']), biases['bh2'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, weights['wh3']), biases['bh3'])
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3, weights['wh4']), biases['bh4'])
    l4 = tf.nn.relu(l4)

    output = tf.add(tf.matmul(l4, weights['whout']), biases['bhout'])
    return output

logits = ann(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_))
optimizer = tf.train.GradientDescentOptimizer(LR).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epoch):
        sess.run(optimizer, feed_dict={x:train_x, y_:train_y})
        c = sess.run(cost, feed_dict={x:train_x, y_:train_y})
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        pred_y = sess.run(logits, feed_dict={x:test_x})
        accuracy = sess.run(accuracy, feed_dict={x:train_x, y_:train_y})

        print('Epoch:', epoch+1, 'completed out of', training_epoch, 'Loss:', c)
    corr = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
    print('Test accuracy:', (sess.run(accuracy, feed_dict={x:test_x, y_:test_y})))

#read_file(csv)
