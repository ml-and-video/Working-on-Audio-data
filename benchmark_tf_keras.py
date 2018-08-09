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
from keras.layers import Convolution2D, MaxPooling2D,LeakyReLU
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
from hyperas.distributions import choice, uniform, conditional
import hyperopt
import hyperas
from hyperas.distributions import choice, uniform, conditional
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from sklearn.cross_validation import train_test_split
from tqdm import tqdm,tqdm_pandas
import librosa
from tqdm._tqdm_notebook import tqdm_notebook

#TQDM build
def tqdm_pandas(t):
  from pandas.core.frame import Series
  def inner(series, func, *args, **kwargs):
      t.total = series.size
      def wrapper(*args, **kwargs):
          t.update(1)
          return func(*args, **kwargs)
      result = series.apply(wrapper, *args, **kwargs)
      t.close()
      return result
  Series.progress_apply = inner

tqdm_pandas(tqdm_notebook())
tqdm.pandas(desc="my bar!")

#data prep and feature engg 
def data():
    os.chdir('D:/data_science/kaggle_sound_classification')
    train = pd.read_csv('train.csv')
    train = train.iloc[1:300,:]
    train_path = 'D:/data_science/kaggle_sound_classification/audio_train/'
    def get_feature(fname):
        b,_ = librosa.load(fname, res_type = 'kaiser_fast')
        try:
            mfcc = np.mean(librosa.feature.mfcc(y = b,n_mfcc=60).T,axis=0)
            mels = np.mean(librosa.feature.melspectrogram(b, sr = SAMPLE_RATE).T,axis = 0)
            stft = np.abs(librosa.stft(b))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr = SAMPLE_RATE).T,axis = 0)
            contrast=np.mean(librosa.feature.spectral_contrast(S=stft, sr=SAMPLE_RATE).T,axis=0)
            tonnetz=np.mean(librosa.feature.tonnetz(librosa.effects.harmonic(b), sr = SAMPLE_RATE).T,axis = 0)
            ft2 = librosa.feature.zero_crossing_rate(b)[0]
            ft3 = librosa.feature.spectral_rolloff(b)[0]
            ft4 = librosa.feature.spectral_centroid(b)[0]
            ft5 = librosa.feature.spectral_contrast(b)[0]
            ft6 = librosa.feature.spectral_bandwidth(b)[0]
            ft2_trunc = np.hstack([np.mean(ft2),np.std(ft2), skew(ft2), np.max(ft2), np.min(ft2)])
            ft3_trunc = np.hstack([np.mean(ft3),np.std(ft3), skew(ft3), np.max(ft3), np.min(ft3)])
            ft4_trunc = np.hstack([np.mean(ft4),np.std(ft4), skew(ft4), np.max(ft4), np.min(ft4)])
            ft5_trunc = np.hstack([np.mean(ft5),np.std(ft5), skew(ft5), np.max(ft5), np.min(ft5)])
            ft6_trunc = np.hstack([np.mean(ft6),np.std(ft6), skew(ft6), np.max(ft6), np.min(ft6)])
            return pd.Series(np.hstack((mfcc,mels,chroma,contrast,tonnetz,ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc)))
        except:
            print('bad file')
            return pd.Series([0]*238)
    train_files = glob(train_path + '*.wav')    
    train['fname'] = train_files   
    train_data = train['fname'].progress_apply(lambda x: get_feature(x))
    print('data loaded')

    train_data = np.array(train_data)
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
            return self.le.inverse_transform(x)

    leohe = labelOnehotEnc()
    y_enc = leohe.label_fit(y)
    train_X,test_X,train_y,test_y = train_test_split(train_data,y_enc, test_size = 0.1, random_state = 20)
    return train_X, test_X, train_y, test_y

print('data ready for model')  

#Keras model with grid search  
def create_model(train_X, test_X, train_y, test_y):
    model = Sequential()
    model.add(Dense(500, input_shape=(238,),kernel_initializer= {{choice(['glorot_uniform','random_uniform'])}}))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
    model.add(Activation({{choice(['relu','sigmoid','tanh'])}}))
    model.add(Dropout({{uniform(0, 0.3)}}))

    model.add(Dense({{choice([128,256])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 0.4)}}))

    model.add(Dense({{choice([128,256])}}))
    model.add(Activation({{choice(['relu','tanh'])}}))
    model.add(Dropout(0.3))

    model.add(Dense(41))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer={{choice(['rmsprop', 'adam'])}})
    model.summary()
    early_stops = EarlyStopping(patience=3, monitor='val_acc')
    ckpt_callback = ModelCheckpoint('keras_model', 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='auto')

    model.fit(train_X, train_y, batch_size={{choice([128,264])}}, nb_epoch={{choice([10,20])}}, validation_data=(test_X, test_y), callbacks=[early_stops,ckpt_callback])
    score, acc = model.evaluate(test_X, test_y, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    import gc; gc.collect()
    best_run, best_model = optim.minimize(model=create_model,data = data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    train_X, test_X, train_y, test_y = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(test_X, test_y))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

#Best run val loss-1.38, val_acc- 62%

#Tensorflow model- built with best hyperparameter

#Tensorflow model
tf.reset_default_graph()
tf.__version__
import __future__
nclass = train_y.shape[1]
input = 238
batch_size = 64
h_1 = 256
h_2 = 256
h_3 = 256
X = tf.placeholder(tf.float32, shape=[None, input])
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
display_step = 10
n_epoch = 10
init = tf.global_variables_initializer()
step = 0
batch_size = 256
# Start training
no_of_batches = int(len(train_X)/batch_size)    
for step in range(1,n_epoch):
    ptr = 0
    for j in range(1, no_of_batches):
        batch_x, batch_y = train_X[ptr:ptr+batch_size], train_y[ptr:ptr+batch_size]
        ptr+=batch_size
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        batch_x = batch_x.reshape((batch_size,input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0:
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
#Tensorflow val_loss 3.38, val_acc 3.2%(as good as random!)

#prediction on test files from keras model

#test data prep
test_path = 'D:/data_science/kaggle_sound_classification/audio_train/'
test_files = glob(test_path + '*.wav')

SAMPLE_RATE = 22050
def get_feature(fname):
    b,_ = librosa.load(fname, res_type = 'kaiser_fast')
    try:
        mfcc = np.mean(librosa.feature.mfcc(y = b,n_mfcc=60).T,axis=0)
        mels = np.mean(librosa.feature.melspectrogram(b, sr = SAMPLE_RATE).T,axis = 0)
        stft = np.abs(librosa.stft(b))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr = SAMPLE_RATE).T,axis = 0)
        contrast=np.mean(librosa.feature.spectral_contrast(S=stft, sr=SAMPLE_RATE).T,axis=0)
        tonnetz=np.mean(librosa.feature.tonnetz(librosa.effects.harmonic(b), sr = SAMPLE_RATE).T,axis = 0)
        ft2 = librosa.feature.zero_crossing_rate(b)[0]
        ft3 = librosa.feature.spectral_rolloff(b)[0]
        ft4 = librosa.feature.spectral_centroid(b)[0]
        ft5 = librosa.feature.spectral_contrast(b)[0]
        ft6 = librosa.feature.spectral_bandwidth(b)[0]
        ft2_trunc = np.hstack([np.mean(ft2),np.std(ft2), skew(ft2), np.max(ft2), np.min(ft2)])
        ft3_trunc = np.hstack([np.mean(ft3),np.std(ft3), skew(ft3), np.max(ft3), np.min(ft3)])
        ft4_trunc = np.hstack([np.mean(ft4),np.std(ft4), skew(ft4), np.max(ft4), np.min(ft4)])
        ft5_trunc = np.hstack([np.mean(ft5),np.std(ft5), skew(ft5), np.max(ft5), np.min(ft5)])
        ft6_trunc = np.hstack([np.mean(ft6),np.std(ft6), skew(ft6), np.max(ft6), np.min(ft6)])
        return pd.Series(np.hstack((mfcc,mels,chroma,contrast,tonnetz,ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc)))
        #d = np.hstack([mfcc,mels,chroma,contrast,tonnetz,ft2_trunc,ft3_trunc,ft4_trunc,ft5_trunc,ft6_trunc])
        #features = np.empty((0,238))
        #d = np.vstack([features,d])
    except:
        print('bad file')
        return pd.Series([0]*238)    
    #return d    

test = pd.DataFrame()
test['fname'] = test_files
test_audio = test['fname'].progress_apply(lambda x: get_feature(x))
print('done loading test mfcc')    
test_audio = np.array(test_audio)


#prediction
pred = model.predict_proba(test_audio)
pred = np.array(pred)

prediction = []
for i in tqdm(range(pred.shape[0])):
    np.argsort(pred)[i][::-1]
    pred_i = np.argsort(pred)[i][::-1][:5]
    prediction.append(pred_i)
prediction = np.array(prediction)
prediction = pd.DataFrame(prediction)

#Top-3 prediction
predicted_tag = []
for i in tqdm(range(pred.shape[0])):
    tag = " ".join(prediction.iloc[i,0:3])
    predicted_tag.append(tag)
predicted_tag = pd.Series(predicted_tag)
sub = pd.read_csv('D:/data_science/kaggle_sound_classification/sample_submission.csv')
sub['label'] = predicted_tag
sub.to_csv('D:/data_science/kaggle_sound_classification/pred_submission/submission.csv')