import random
import time
import collections
import nltk
from nltk.metrics import precision, recall, f_measure

import FeatureExtractor
import DataCreator
from loadingbar import printPercentage

def runClassifiers(positives, negatives, verbose, useBayes, useTree, useEntropy):
    featureSets = []
    onDataSet = 0
    numDataSets = len(positives + negatives)

    for data in positives:
        featureSets.append((FeatureExtractor.langFeatures(data), True))
        onDataSet += 1
        printPercentage(onDataSet/numDataSets * 100, "Extracting Features: ")

    for data in negatives:
        featureSets.append((FeatureExtractor.langFeatures(data), False))
        onDataSet += 1
        printPercentage(onDataSet/numDataSets * 100, "Extracting Features: ")

    print("\n")

    #Testing is 1/4 of the data set, so we will cut it off there
    cutOff = len(featureSets)//4

    random.shuffle(featureSets)

    #splits training and test sets
    train, test = featureSets[cutOff:], featureSets[:cutOff]

    numDataSets = len(test)

    if useBayes:
        timeStart = time.time()

        print("Running Naive Bayes classifier")
        #NLTK's built-in implementation of the Naive Bayes classifier is trained
        classifier = nltk.NaiveBayesClassifier.train(train)

        #now, it is tested on the test set
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)

        onDataSet = 0
        for i, (feats, label) in enumerate(test):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
            onDataSet += 1
            printPercentage(onDataSet/numDataSets * 100, "Extracting Features: ")

        #get the time it takes to run Naive Bayes
        print ("\nTime to run in seconds: ", time.time() - timeStart)

        #report the accuracy
        print ("Bayes Precision: ", precision(refsets[True], testsets[True]))
        print ("Bayes Recall: ", recall(refsets[True], testsets[True]))
        print ("Bayes F-Measure: ", f_measure(refsets[True], testsets[True]))

        if verbose:
            #this is a nice function that reports the top most impactful features the NB classifier found
            print (classifier.show_most_informative_features(20))

    if useTree:
        timeStart = time.time()

        print("Running Decision Tree classifier")
        #NLTK's built-in implementation of the Decision Tree classifier is trained
        classifier = nltk.DecisionTreeClassifier.train(train)

        #now, it is tested on the test set
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)

        onDataSet = 0
        for i, (feats, label) in enumerate(test):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
            onDataSet += 1
            printPercentage(onDataSet/numDataSets * 100, "Extracting Features: ")

        #get the time to run Decision tree
        print ("\nTime to run in seconds: ", time.time() - timeStart)

        #now, it is tested on the test set and the accuracy reported
        print ("DTree Precision: ", precision(refsets[True], testsets[True]))
        print ("DTree Recall: ", recall(refsets[True], testsets[True]))
        print ("DTree F-Measure: ", f_measure(refsets[True], testsets[True]))

        if verbose:
            print("Printing tree")
            #print(classifier.pretty_format())
            for (feats, cor) in test[:20]:
                classification = classifier.classify(feats)
                print("Correct: ", cor, " Result: ", classification)#, "for ", feats[0])

    if useEntropy:
        timeStart = time.time()

        print("Running Maximum Entropy classifier")
        #NLTK's built-in implementation of the Naive Bayes classifier is trained
        classifier = nltk.MaxentClassifier.train(train)

        #now, it is tested on the test set
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)

        onDataSet = 0
        for i, (feats, label) in enumerate(test):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
            onDataSet += 1
            printPercentage(onDataSet/numDataSets * 100, "Extracting Features: ")

        #get the time to run Decision tree
        print ("\nTime to run in seconds: ", time.time() - timeStart)

         #now, it is tested on the test set and the accuracy reported
        print ("Entropy Precision: ", precision(refsets[True], testsets[True]))
        print ("Entropy Recall: ", recall(refsets[True], testsets[True]))
        print ("Entropy F-Measure: ", f_measure(refsets[True], testsets[True]))

        if verbose:
            #this is a nice function that reports the top most impactful features the NB classifier found
            print (classifier.show_most_informative_features(20))
            #this is a function that explains the effect of each feature in the set
            #print (classifier.explain())