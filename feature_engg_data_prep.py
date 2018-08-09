# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 11:50:29 2018

@author: Sourish
"""

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
from keras import applications,models, losses,optimizers
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
import IPython.display as ipd
from tqdm import tqdm,tqdm_pandas
import shutil
from scipy.stats import kurtosis
from scipy.stats import skew

os.chdir('D:/data_science/kaggle_sound_classification')
os.getcwd()
train = pd.read_csv('train.csv')
test = pd.read_csv("pred_submission/sample_submission.csv")

#Basic Exploratory analysis
train.columns
train.manually_verified.value_counts()
grp = train.groupby(['label','manually_verified']).count()
plot = grp.unstack().reindex(grp.unstack().sum(axis = 1).sort_values().index).plot(kind = 'bar',stacked = True,figsize=(12,4),title = 'audio_clip_taggingwise')
plot.set_xlabel('audio class')
plot.set_ylabel('manual-auto')
train = train.loc[train['manually_verified'] == 1,:]
data, samp_rate = librosa.load('D:/data_science/kaggle_sound_classification/audio_train/00ad7068.wav')  
plt.figure(figsize=(12,4))
ipd.Audio('D:/data_science/kaggle_sound_classification/audio_train/00c9e799.wav')
librosa.display.waveplot(data, sr=samp_rate)
train.label.value_counts()
train_path = 'D:/data_science/kaggle_sound_classification/audio_train/'
test_path = 'D:/data_science/kaggle_sound_classification/audio_test/'

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

#Feature engineering
SAMPLE_RATE = 22050
from scipy.stats import skew
print(os.listdir(os.getcwd()))    
tqdm.pandas
train_files = os.listdir(train_path)
test_files = os.listdir(test_path)
train_files = glob(train_path + '*.wav')
test_files = glob(test_path + '*.wav')
SAMPLE_RATE = 22050
def get_feature(fname):
    #b,_ = librosa.load(fname, res_type = 'kaiser_fast')
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


train_data = train['fname'].progress_apply(lambda x: get_feature(x))
    
#Few more features- basic statistical features
def get_stat_feature(fname):
    #b,_ = librosa.load(fname, res_type = 'kaiser_fast')
    b,_ = librosa.load(i, res_type = 'kaiser_fast')
    try:
        #basic statistical features
        length = len(b)
        mean = np.mean(b)
        minimum = np.min(b)
        maximum = np.max(b)
        std = np.std(b)
        rms = np.sqrt(np.mean(b**2))
        kurt = kurtosis(b)
        Skew = skew(b)
        #Audio length feature
        data,samp_rate = librosa.effects.trim(b,top_db = 40)
        len_init = len(data) 
        ratio_init = len_init/length
        splits = librosa.effects.split(b, top_db=40)
        if len(splits) > 1:
            b = np.concatenate([b[x[0]:x[1]] for x in splits]) 
        len_final = len(b) 
        ratio_final = len_final/length
        #return pd.Series([mean,minimum,maximum,std,rms,kurt,Skew,len_init,ratio_init,len_final,ratio_final])
        return pd.Series(np.hstack((mean,minimum,maximum,std,rms,kurt,Skew,len_init,ratio_init,len_final,ratio_final)))
    except:
        print("Bad file at {}".format(fname))
        return pd.Series([0]*11)      

train_data1 = train['fname'].progress_apply(lambda x: get_stat_feature(x)) 

#Concatenate all features
train_data = np.concatenate([train_data1, train_data], axis = 1)
train_data = pd.DataFrame(train_data)
train_data.to_csv('D:/data_science/kaggle_sound_classification/pred_submission/train_with_feature.csv')

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

#test data prep
test_path = 'D:/data_science/kaggle_sound_classification/audio_train/'
test_files = glob(test_path + '*.wav')
test = pd.DataFrame()
test['fname'] = test_files
test_audio = test['fname'].progress_apply(lambda x: get_feature(x))
test_audio1 = test['fname'].progress_apply(lambda x: get_stat_feature(x))
train_data = np.concatenate([train_data1, train_data], axis = 1)
train_data = pd.DataFrame(train_data)
print('done loading test mfcc')    
test_audio = np.array(test_audio)

test_data.to_csv('D:/data_science/kaggle_sound_classification/pred_submission/test_with_feature.csv')

 

   