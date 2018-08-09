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
from tqdm import tqdm
import librosa

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
def create_model(train_X, test_X, train_y, test_y):
    model = Sequential()
    model.add(Dense(500, input_shape=(60,),kernel_initializer= {{choice(['glorot_uniform','random_uniform'])}}))
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