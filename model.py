from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC

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

def getGraphs():
    home = expanduser("~")
    path = home + '/' + 'Documents/sklearn-instrument-classification/images'
    graphs=[]
    labels=[]
    for file in os.listdir(path):
        if ".png" not in file:
            continue
        print(file)
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

    plot_confusion_matrix(classifier, testX, testY)
    pyplot.show()


main()
# make into long list
# strip off numbers
# read string --> cut off the last x digits so that it's only the instrument name
# convert it to a set()
# make it back into a list
