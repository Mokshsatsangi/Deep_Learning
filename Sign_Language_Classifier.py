import cv2
import numpy as np
from random import shuffle
from collections import Counter
from tqdm import tqdm
import os
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

TRAIN_A_DIR = 'D:/user/Documents/Python codes/Datasets/Sign Language/Training/A'
TRAIN_B_DIR = 'D:/user/Documents/Python codes/Datasets/Sign Language/Training/B'
TRAIN_C_DIR = 'D:/user/Documents/Python codes/Datasets/Sign Language/Training/C'
TRAIN_FIVE_DIR = 'D:/user/Documents/Python codes/Datasets/Sign Language/Training/Five'
TRAIN_POINT_DIR = 'D:/user/Documents/Python codes/Datasets/Sign Language/Training/Point'
TRAIN_V_DIR = 'D:/user/Documents/Python codes/Datasets/Sign Language/Training/V'

TEST_A_DIR = 'D:/user/Documents/Python codes/Datasets/Sign Language/Testing/A'
TEST_B_DIR = 'D:/user/Documents/Python codes/Datasets/Sign Language/Testing/B'
TEST_C_DIR = 'D:/user/Documents/Python codes/Datasets/Sign Language/Testing/C'
TEST_FIVE_DIR = 'D:/user/Documents/Python codes/Datasets/Sign Language/Testing/Five'
TEST_POINT_DIR = 'D:/user/Documents/Python codes/Datasets/Sign Language/Testing/Point'
TEST_V_DIR = 'D:/user/Documents/Python codes/Datasets/Sign Language/Testing/V'

LR = 1e-3
IMG_SIZE = 50
MODEL_NAME = 'Sign_Language-{}-{}.model'.format(LR, 'Covnet')

def create_train_data():
    training_data = []
    for img1 in tqdm(os.listdir(TRAIN_A_DIR)):
        label1 = [1, 0, 0, 0, 0, 0]
        path1 = os.path.join(TRAIN_A_DIR, img1)
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img1), np.array(label1)]) # ==> [i[0], i[1]]

    for img2 in tqdm(os.listdir(TRAIN_B_DIR)):
        label2 = [0, 1, 0, 0, 0, 0]
        path2 = os.path.join(TRAIN_B_DIR, img2)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.resize(img2, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img2), np.array(label2)]) # ==> [i[0], i[1]]

    for img3 in tqdm(os.listdir(TRAIN_C_DIR)):
        label3 = [0, 0, 1, 0, 0, 0]
        path3 = os.path.join(TRAIN_C_DIR, img3)
        img3 = cv2.imread(path3, cv2.IMREAD_GRAYSCALE)
        img3 = cv2.resize(img3, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img3), np.array(label3)]) # ==> [i[0], i[1]]

    for img4 in tqdm(os.listdir(TRAIN_FIVE_DIR)):
        label4 = [0, 0, 0, 1, 0, 0]
        path4 = os.path.join(TRAIN_FIVE_DIR, img4)
        img4 = cv2.imread(path4, cv2.IMREAD_GRAYSCALE)
        img4 = cv2.resize(img4, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img4), np.array(label4)]) # ==> [i[0], i[1]]

    for img5 in tqdm(os.listdir(TRAIN_POINT_DIR)):
        label5 = [0, 0, 0, 0, 1, 0]
        path5 = os.path.join(TRAIN_POINT_DIR, img5)
        img5 = cv2.imread(path5, cv2.IMREAD_GRAYSCALE)
        img5 = cv2.resize(img5, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img5), np.array(label5)]) # ==> [i[0], i[1]]

    for img6 in tqdm(os.listdir(TRAIN_V_DIR)):
        label6 = [0, 0, 0, 0, 0, 1]
        path6 = os.path.join(TRAIN_V_DIR, img6)
        img6 = cv2.imread(path6, cv2.IMREAD_GRAYSCALE)
        img6 = cv2.resize(img6, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img6), np.array(label6)]) # ==> [i[0], i[1]]

    shuffle(training_data)
    np.save('trained_data.npy', training_data)
    return training_data

def create_test_data():
    testing_data = []
    for img1 in tqdm(os.listdir(TEST_A_DIR)):
        path1 = os.path.join(TEST_A_DIR, img1)
        img_num1 = img1.split('-')[0]
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img1), img_num1]) # ==> [i[0], i[1]]

    for img2 in tqdm(os.listdir(TEST_B_DIR)):
        path2 = os.path.join(TEST_B_DIR, img2)
        img_num2 = img2.split('-')[0]
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.resize(img2, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img2), img_num2]) # ==> [i[0], i[1]]

    for img3 in tqdm(os.listdir(TEST_C_DIR)):
        path3 = os.path.join(TEST_C_DIR, img3)
        img_num3 = img3.split('-')[0]
        img3 = cv2.imread(path3, cv2.IMREAD_GRAYSCALE)
        img3 = cv2.resize(img3, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img3), img_num3]) # ==> [i[0], i[1]]

    for img4 in tqdm(os.listdir(TEST_FIVE_DIR)):
        path4 = os.path.join(TEST_FIVE_DIR, img4)
        img_num4 = img4.split('-')[0]
        img4 = cv2.imread(path4, cv2.IMREAD_GRAYSCALE)
        img4 = cv2.resize(img4, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img4), img_num4]) # ==> [i[0], i[1]]

    for img5 in tqdm(os.listdir(TEST_POINT_DIR)):
        path5 = os.path.join(TEST_POINT_DIR, img5)
        img_num5 = img5.split('-')[0]
        img5 = cv2.imread(path5, cv2.IMREAD_GRAYSCALE)
        img5 = cv2.resize(img5, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img5), img_num5]) # ==> [i[0], i[1]]

    for img6 in tqdm(os.listdir(TEST_V_DIR)):
        path6 = os.path.join(TEST_V_DIR, img6)
        img_num6 = img6.split('-')[0]
        img6 = cv2.imread(path6, cv2.IMREAD_GRAYSCALE)
        img6 = cv2.resize(img6, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img6), img_num6]) # ==> [i[0], i[1]]

    shuffle(testing_data)
    np.save('tested_data.npy', testing_data)
    return testing_data

train_data = create_train_data()
test_data = create_test_data()
print('Initiating data...')

tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 6, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

train = train_data[:-500]
test = train_data[-500:]
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

'''model.fit({'input': X}, {'targets': Y}, n_epoch=100, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.save('Sign_Language_Interpreter.model')'''

model.load('Sign_Language_Interpreter.model')
ans = input('Again (y/n)? ')
while ans == 'y':
    if ans == 'y':
        num = int(input('Enter a number of that image: '))
        prediction = model.predict([test_x[num]])
        print(prediction)
        if prediction[0][0] == float(1):
            print('A')
        elif prediction[0][1] == float(1):
            print('B')
        elif prediction[0][2] == float(1):
            print('C')
        elif prediction[0][3] == float(1):
            print('five')
        elif prediction[0][4] == float(1):
            print('point')
        elif prediction[0][5] == float(1):
            print('V')
        ans = input('Again (y/n)? ')
    elif ans == 'n':
        break
