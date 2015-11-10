import os
import random
import nltk
import DataCreator
from nltk import word_tokenize

#First define a function that produces features from a given object
#In this case a script of a TED Talk cut up to the first laugh (or the end of a sentence if there were no laughs)
#The features extracted for the talk are every word in the text
#Note that featuresets are lists. That's what the classifier takes as input
def langFeatures(data):
    D = {} #dictionary of keys

    #ONLY FOR TESTING! REMOVE LATER
    #D[0] = data[0]

    for word in data[1:]:
        #the feature list is the words in the script
        D[word] = True

    return D


def extractFeatures(positives, negatives, verbose, useBayes, useTree):
    featureSets = []

    for data in positives:
        featureSets.append((langFeatures(data), True))

    for data in negatives:
        featureSets.append((langFeatures(data), False))

    #Testing is 1/4 of the data set, so we will cut it off there
    cutOff = len(featureSets)//4

    random.shuffle(featureSets)

    #splits training and test sets
    train, test = featureSets[cutOff:], featureSets[:cutOff]

    if useBayes:
        print("Running Naive Bayes classifier")
        #NLTK's built-in implementation of the Naive Bayes classifier is trained
        classifier = nltk.NaiveBayesClassifier.train(train)

        #now, it is tested on the test set and the accuracy reported
        print ("Bayes Accuracy: ", nltk.classify.accuracy(classifier, test))

        if verbose:
            #this is a nice function that reports the top most impactful features the NB classifier found
            print (classifier.show_most_informative_features(20))

    if useTree:
        print("Running Decision Tree classifier")
        #NLTK's built-in implementation of the Decision Tree classifier is trained
        classifier = nltk.DecisionTreeClassifier.train(train)

        #now, it is tested on the test set and the accuracy reported
        print ("DTree Accuracy: ", nltk.classify.accuracy(classifier, test))

        if verbose:
            print("Printing tree")
            #print(classifier.pretty_format())
            for (feats, cor) in test:
                classification = classifier.classify(feats)
                print("Correct: ", cor, " Result: ", classification)#, "for ", feats[0])


#TODO Handle command line arguments. 1 for the amount of test files, another for if we are making a new meta file or not, and
#what classifier to use
(positives, negatives) = DataCreator.createDataFrom("parsed_websites/", "Ted_Meta.txt", "Ted_Laughs.txt", 100, False)
print("Extracting Features\n")
extractFeatures(positives, negatives, True, True, True)
