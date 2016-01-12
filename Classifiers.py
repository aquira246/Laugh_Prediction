import random
import time
import collections
import nltk
import sys
from nltk.metrics import precision, recall, f_measure
from tabulate import tabulate

import FeatureExtractor
from loadingbar import printPercentage


# helper function to run tests on the classifier passed in
def assess_classifier(classifier, test_data, text):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    numDataSets = len(test_data)
    onDataSet = 0

    # enumerate through the test data and classify them
    for i, (feats, label) in enumerate(test_data):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
        onDataSet += 1
        printPercentage(onDataSet/numDataSets * 100, "Extracting Features: ")

    # calculate the precision and recall
    laugh_precision = precision(refsets[True], testsets[True])
    laugh_recall = recall(refsets[True], testsets[True])

    non_laugh_precision = precision(refsets[False], testsets[False])
    non_laugh_recall = recall(refsets[False], testsets[False])

    acc = nltk.classify.accuracy(classifier, test_data)

    return [text, acc, laugh_precision, laugh_recall, non_laugh_precision, non_laugh_recall]


def runClassifiers(positives, negatives, verbose, useBayes, useTree, useEntropy):
    featureSets = []
    onDataSet = 0
    numDataSets = len(positives + negatives)
    table = []

    for data in positives:
        featureSets.append((FeatureExtractor.langFeatures(data), True))
        onDataSet += 1
        printPercentage(onDataSet/numDataSets * 100, "Extracting Features: ", False)

    for data in negatives:
        featureSets.append((FeatureExtractor.langFeatures(data), False))
        onDataSet += 1
        printPercentage(onDataSet/numDataSets * 100, "Extracting Features: ", False)

    sys.stdout.flush()
    print("\n")

    # Testing is 1/4 of the data set, so we will cut it off there
    cutOff = len(featureSets)//4

    random.shuffle(featureSets)

    # splits training and test sets
    train_data, test_data = featureSets[cutOff:], featureSets[:cutOff]

    if useBayes:
        print("Running Naive Bayes classifier")
        timeStart = time.time()

        # NLTK's built-in implementation of the Naive Bayes classifier is trained
        classifier = nltk.NaiveBayesClassifier.train(train_data)

        # get the time it takes to train Naive Bayes
        print ("\nTime to train in seconds: ", time.time() - timeStart)

        # store the accuracy in the table
        table.append(assess_classifier(classifier, test_data, "Naive Bayes"))

        if verbose:
            # this is a nice function that reports the top most impactful features the NB classifier found
            print("\n\n")
            print (classifier.show_most_informative_features(20))

    if useTree:
        print("Running Decision Tree classifier")
        timeStart = time.time()

        # NLTK's built-in implementation of the Decision Tree classifier is trained
        classifier = nltk.DecisionTreeClassifier.train(train_data)

        # get the time to train Decision tree
        print ("\nTime to train in seconds: ", time.time() - timeStart)

        # store the accuracy in the table
        table.append(assess_classifier(classifier, test_data, "Decision Tree"))

        if verbose:
            print("Printing tree")
            # print(classifier.pretty_format())
            for (feats, cor) in test_data[:20]:
                classification = classifier.classify(feats)
                print("Correct: ", cor, " Result: ", classification)#, "for ", feats[0])

    if useEntropy:
        print("Running Maximum Entropy classifier")
        timeStart = time.time()

        # NLTK's built-in implementation of the Max Entropy classifier is trained
        classifier = nltk.MaxentClassifier.train(train_data)

        # get the time to train Maximum Entropy
        print ("\nTime to train in seconds: ", time.time() - timeStart)

        # store the accuracy in the table
        table.append(assess_classifier(classifier, test_data, "Maximum Entropy"))

        if verbose:
            # this is a nice function that reports the top most impactful features the NB classifier found
            print (classifier.show_most_informative_features(20))
            # this is a function that explains the effect of each feature in the set
            # print (classifier.explain())

    print("\n", tabulate(table, headers=["Classifier", "accuracy", "laugh precision", "laugh recall", "non-laugh precision", "non-laugh recall"]))
