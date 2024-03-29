import numpy as np
import matplotlib
import math
import os
from matplotlib import pyplot as plt
import IPython.display as ipd
import librosa
import librosa.display
from keras.models import load_model
model = load_model('mandarin_model.h5')


def mp3tomfcc(file_path, max_pad):
  audio, sample_rate = librosa.core.load(file_path)
  mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
  pad_width = max_pad - mfcc.shape[1]
  mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
  return mfcc

def mand_deepL(sound):
    mfcss = []

    myTest3 = sound

    mfccs2 = []
    mfccs2.append(mp3tomfcc(myTest3, 60))
    mfccs2 = np.asarray(mfccs2)
    # print(mfccs2.shape)

    X2 = mfccs2
    dim_1 = mfccs2.shape[1]
    dim_2 = mfccs2.shape[2]
    channels = 1

    X2 = X2.reshape((mfccs2.shape[0], dim_1, dim_2, channels))

    myTonePrediction = np.argmax(model.predict(X2), axis=-1)
    # result =  "Predicted tone: " + str(myTonePrediction)
    # countDownLabel['text'] = result
    return myTonePrediction
