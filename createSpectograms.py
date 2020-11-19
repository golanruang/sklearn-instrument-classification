import librosa
import librosa.display
from os import listdir
from os.path import isfile, join, expanduser
import os

import matplotlib.pyplot as plt
import numpy as np

import cv2

home = expanduser("~")
path = home + '/' + 'Documents/sklearn-instrument-classification/nsynth-test/audio'

for f in os.listdir(path):                          # for every file in the path given
    index=f.rfind('.')
    newname=f[:index]

    y, sr = librosa.load("nsynth-test/audio/" + f)
    S=librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    #fig, ax = plt.subplots()

    S_dB = librosa.power_to_db(S, ref=np.max)

    # img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)

    #plt.matshow(S_dB)

    img=np.array(S_dB) * -1

#plt.show()

    cv2.imwrite('./images/' + newname + '.png', img)
    print("img written")
    # find datatype of the np array
    # uint8 and int8 multiply by -1
    # if float (0-1) --> 1-np.array(S_dB)
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')

    # ax.set(title='Mel-frequency spectrogram')
