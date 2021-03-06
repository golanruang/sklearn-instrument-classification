from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification

import librosa
import librosa.display

from os import listdir
from os.path import isfile, join, expanduser
import os

import matplotlib.pyplot as plt
import numpy as np

import cv2
# need to flatten to 1x64 by reshape
# numpy.reshape()
# should be uint8

# Fourier transform transforms signal into sine waves that when you add it together you get the signal
# y axis is frequency
# x axis is time
# MEL spectrogram -->
# amplitude of each sine wave is the hue

def getGraphs():
    home = expanduser("~")
    path = home + '/' + 'Documents/sklearn-instrument-classification/images'
    graphs=[]
    labels=[]
    for file in os.listdir(path):
        if ".png" not in file:
            continue
        #print(file)
        img=cv2.imread("images/"+file,0)
        image=img.flatten()
        graphs.append(image)
        instrumentName="_".join(file.split("_")[:-1])
        labels.append(instrumentName)
    return graphs,labels

def main():
    graphs,labels=getGraphs()
    # graphs=np.array(graphs)
    # graphs=graphs.flatten()
    #graphs=np.reshape(graphs)/255

    trainX, testX, trainY, testY = train_test_split(
    graphs, labels, test_size=0.3, shuffle=True
    )

    # SVMs work by taking some features and using this features to define the model
    # usually the model doesn't process these features
    # somehow the model is doing the analysis?
    # it tries to find a line that separates the data

    classifier = SVC(max_iter=10000)
    classifier.fit(trainX,trainY)
    preds = classifier.predict(testX)

    correct = 0
    incorrect = 0
    for pred, gt in zip(preds, testY):
        if pred == gt:
            correct += 1
        else:
            incorrect += 1

    print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect)}")

    #plot_confusion_matrix(classifier, testX, testY)
    #plt.show()

    # usually used for removing outliers (image noise)
    # there's no noise because this is a spectrogram so the scaler really isn't doing anything
    pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])

    pipe.fit(trainX, trainY)

    # classifier = SVC(max_iter=10000)
    # classifier.fit(trainX,trainY)
    preds = pipe.predict(testX)

    correct = 0
    incorrect = 0
    for pred, gt in zip(preds, testY):
        if pred == gt:
            correct += 1
        else:
            incorrect += 1

    print(f"After Standard Scaler\nCorrect: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect)}")

    plot_confusion_matrix(pipe, testX, testY)
    #plt.show()


main()
# make into long list
# strip off numbers
# read string --> cut off the last x digits so that it's only the instrument name
# convert it to a set()
# make it back into a list
