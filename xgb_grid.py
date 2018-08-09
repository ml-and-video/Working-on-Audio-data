import librosa
import librosa.display
import scipy.io.wavfile
import cv2
from glob import glob
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tqdm
import cv2
from scipy.spatial import distance
from PIL import Image
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm import tqdm,tqdm_pandas
from hyperas.distributions import choice, uniform, conditional
import hyperopt
import hyperas
from hyperas.distributions import choice, uniform, conditional
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe,hp,fmin
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb
from ctypes import *
from sklearn import datasets, metrics, model_selection
from sklearn.grid_search import GridSearchCV 
from xgboost.sklearn import XGBClassifier

os.chdir('D:/data_science/kaggle_sound_classification')
os.getcwd()
train = pd.read_csv('train.csv',nrows = 200)
y = train.label
train_data = pd.read_csv('D:/data_science/kaggle_sound_classification/train_with_feature.csv',nrows = 200)
train_data = train_data.iloc[:,1:239]
train_data = np.array(train_data)  
le = LabelEncoder()
y = le.fit_transform(y)
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
train_X,test_X,train_y,test_y = train_test_split(train_data,y, test_size = 0.1, random_state = 20)
train_y_enc,test_y_enc = train_test_split(y_enc, test_size = 0.1, random_state = 20)
del train_data
print('data prepared')

def objective(params):
    params = {
        'max_depth': int(params['max_depth']),
        'gamma': "{:.3f}".format(params['gamma']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
    }

    clf = xgb.XGBClassifier(
        n_estimators=50,objective= 'multi:softprob',
        learning_rate=0.05,
        n_jobs=4,
        **params
    )

    eval_set  = [(train_X,train_y), (test_X,test_y)]
    clf.fit(train_X, train_y,
            eval_set=eval_set, eval_metric="mlogloss",
            early_stopping_rounds=10,verbose = 10)

    pred = clf.predict_proba(test_X)
    logloss = log_loss(test_y_enc, pred)
    print("logloss {:.3f} params {}".format(logloss, params))

    return{'loss':logloss, 'status': STATUS_OK }


space = {
    'max_depth': hp.quniform('max_depth', 2, 8, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'gamma': hp.uniform('gamma', 0.0, 0.5),
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials)

print(best)