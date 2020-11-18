import librosa
import librosa.display
# from os import listdir
# from os.path import isfile, join
# import os

import matplotlib.pyplot as plt
import numpy as np

import cv2

# home = os.path.expanduser("~")
# path = input('Enter path: ')                         # input for path that audio files/this file are in
# path = home + '/' + 'Documents/sklearn-instrument-classification/nsynth-test/audio'

# files=[]
# for f in os.listdir(path):                          # for every file in the path given
# index=f.rfind('.')
# newname=f[:index]

y, sr = librosa.load('nsynth-test/audio/string_acoustic_012-042-050.wav')
S=librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

fig, ax = plt.subplots()

S_dB = librosa.power_to_db(S, ref=np.max)

# img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)

plt.matshow(S_dB)

img=np.array(S_dB) * 255

#plt.show()

cv2.imwrite('./images/string_acoustic_012-042-050.png', img)
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')
    #
    # ax.set(title='Mel-frequency spectrogram')
