import os
import random
import nltk
from nltk import word_tokenize

#First define a function that produces features from a given object
#In this case a script of a TED Talk cut up to the first laugh (or the end of a sentence if there were no laughs)
#The features extracted for the talk are every word in the text
#Note that featuresets are lists. That's what the classifier takes as input
def langFeatures(filename):
    rf = open(filename, 'r')
    D = {} #dictionary of keys

    for line in rf:
        for word in nltk.word_tokenize(line):
            #the feature list is the words in the script
            D[word] = True

    rf.close
    return D


def isPositive(filename):
    if "+" in filename:
        return True
    else:
        return False


def extractFolder(path):
    files = os.listdir(path)
    random.shuffle(files)

    #Training is 1/5 of the data set, so we will cut it off there
    cutOff = len(files)//5

    featureSets = []

    for filename in os.listdir(path):
        featureSets.append((langFeatures((path+filename)), isPositive(filename)))

    #splits training and test sets
    train, test = featureSets[:cutOff], featureSets[cutOff:]

    #NLTK's built-in implementation of the Naive Bayes classifier is trained
    classifier = nltk.NaiveBayesClassifier.train(train)

    #now, it is tested on the test set and the accuracy reported
    print ("Accuracy: ", nltk.classify.accuracy(classifier, test))

    #this is a nice function that reports the top most impactful features the NB classifier found
    print (classifier.show_most_informative_features(20))


extractFolder("Laugh_Data/Data/")
