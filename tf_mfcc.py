import librosa
import librosa.display
import scipy.io.wavfile
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
from glob import glob
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import tqdm
import cv2
from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
import numpy as np
import keras.backend as K
from scipy.spatial import distance
from PIL import Image
from keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

os.chdir('D:/data_science/kaggle_sound_classification')
os.getcwd()
train = pd.read_csv('train.csv')
train.columns
train.manually_verified.value_counts()
train = train.loc[train['manually_verified'] == 1,:]
data, samp_rate = librosa.load('D:/data_science/kaggle_sound_classification/audio_train/00c9e799.wav')
plt.figure(figsize=(12,4))
import IPython.display as ipd
ipd.Audio('D:/data_science/kaggle_sound_classification/audio_train/00c9e799.wav')
librosa.display.waveplot(data, sr=samp_rate)
train.label.value_counts()
train_path = 'D:/data_science/kaggle_sound_classification/audio_train/'

train_data = []
for i in tqdm((train['fname'].values)):
    filename = os.path.join(train_path + i)
    data,samp_rate = librosa.load(filename, res_type = 'kaiser_fast')
    mfcc = np.mean(librosa.feature.mfcc(y = data,n_mfcc=60).T,axis=0)
    train_data.append(mfcc)

train_data = np.array(train_data)

print('data_loaded')
y = train.label    
class labelOnehotEnc():
    def __init__(self):
        self.le = LabelEncoder()
        self.oe = OneHotEncoder(sparse=False)   
    def label_fit(self,x):
        feature = self.le.fit_transform(x)
        self.oe = OneHotEncoder(sparse=False)
        return self.oe.fit_transform(feature.reshape(-1,1))
    def onehot_inverse(self,x):
        self.indecies = []
        for t in range(len(x)):
            ind = np.argmax((x[t]))
            self.indecies.append(ind)
        return self.le.inverse_transform(self.indecies)
    def inverse_label(self,x):
        return self.le.inverse_transform(x.reshape(len(x),1))

leohe = labelOnehotEnc()
y_enc = leohe.label_fit(y)
train_X,test_X,train_y,test_y = train_test_split(train_data,y_enc, test_size = 0.1, random_state = 20)
print('data ready for modelling')
#Tensorflow model
tf.reset_default_graph()
tf.__version__
import __future__
nclass = train_y.shape[1]
input = 60
batch_size = 64
h_1 = 256
h_2 = 256
h_3 = 256
X = tf.placeholder(tf.float32, shape=[None, 60])
Y = tf.placeholder(tf.float32, [None, nclass])
y_true_cls = tf.argmax(Y, dimension=1) 
weight_1 = tf.get_variable("W1", shape=[input, h_1],
           initializer=tf.contrib.layers.xavier_initializer())
bias_1 = tf.get_variable("b1", shape=[h_1],
           initializer=tf.contrib.layers.xavier_initializer())
weight_2 = tf.get_variable("W2", shape=[h_1,h_2],
           initializer=tf.contrib.layers.xavier_initializer())
bias_2 = tf.get_variable("b2", shape=[h_2],
           initializer=tf.contrib.layers.xavier_initializer())
weight_3 = tf.get_variable("W3", shape=[h_2,nclass],
           initializer=tf.contrib.layers.xavier_initializer())
bias_3 = tf.get_variable("b3", shape=[nclass],
           initializer=tf.contrib.layers.xavier_initializer())

#hidden-layer-1
l1 = tf.add(tf.matmul(X,weight_1),bias_1)
l1 = tf.nn.relu(l1)
#Batch-norm
epsilon = 1e-3
batch_mean2, batch_var2 = tf.nn.moments(l1,[0])
scale2 = tf.Variable(tf.ones([h_1],dtype=tf.float32),dtype=tf.float32)
beta2 = tf.Variable(tf.zeros([h_1],dtype=tf.float32),dtype=tf.float32)
l1 = tf.nn.batch_normalization(l1,batch_mean2,batch_var2,beta2,scale2,epsilon)
l1 = tf.nn.dropout(
    l1,
    0.7,
    noise_shape=None,
    seed=10,
    name=None
)
#hidden-layer-2
l2 = tf.add(tf.matmul(l1,weight_2),bias_2)
l2 = tf.nn.relu(l2)
#Batch-norm
epsilon = 1e-3
batch_mean2, batch_var2 = tf.nn.moments(l1,[0])
scale2 = tf.Variable(tf.ones([h_1],dtype=tf.float32),dtype=tf.float32)
beta2 = tf.Variable(tf.zeros([h_1],dtype=tf.float32),dtype=tf.float32)
l2 = tf.nn.batch_normalization(l2,batch_mean2,batch_var2,beta2,scale2,epsilon)
l2 = tf.nn.dropout(
    l2,
    0.75,
    noise_shape=None,
    seed=10,
    name=None
)
#Final-layer
y = tf.add(tf.matmul(l2,weight_3),bias_3)
with tf.variable_scope("Softmax"):
    y_pred = tf.nn.softmax(y)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_pred))

# Use Adam Optimizer
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Accuracy# Accura 
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
# Initialize the FileWriter
writer = tf.summary.FileWriter("Training_FileWriter/")
writer1 = tf.summary.FileWriter("Validation_FileWriter/")
# Add the cost and accuracy to summary
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', acc)

# Merge all summaries together
merged_summary = tf.summary.merge_all() 

#training
display_step = 100
n_epoch = 10
init = tf.global_variables_initializer()
step = 0
batch_size = 64
# Start training
no_of_batches = int(len(train_X)/batch_size)    
for step in range(1,n_epoch):
    ptr = 0
    for j in range(1, no_of_batches):
        batch_x, batch_y = train_X[ptr:ptr+batch_size], train_y[ptr:ptr+batch_size]
        ptr+=batch_size
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        batch_x = batch_x.reshape((batch_size,60))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, accuracy = sess.run([cost, acc], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " +                   "{:.4f}".format(loss) + ", Training Accuracy= " +                   "{:.3f}".format(accuracy))
    print("Epoch -",str(step))       
    loss, accuracy = sess.run([cost, acc], feed_dict={X: test_X,
                                                                 Y: test_y})
    print("Step " + str(step) + ", step Loss= " +                   "{:.4f}".format(loss) + ", step Test Accuracy= " +                   "{:.3f}".format(accuracy))
print('optimization complete')
sess.close()   