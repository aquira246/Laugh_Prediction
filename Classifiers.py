import random
import time
import collections
import nltk
import sys
from nltk.metrics import precision, recall, f_measure
from tabulate import tabulate

import sklearn
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

import FeatureExtractor
from loadingbar import printPercentage

NUM_CLASSIFIERS = 5


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
        # printPercentage(onDataSet/numDataSets * 100, "Extracting Features: ")

    # calculate the precisionl, recall, f-measure
    laugh_precision = precision(refsets[True], testsets[True])
    laugh_recall = recall(refsets[True], testsets[True])
    laugh_f1 = f_measure(refsets[True], testsets[True])

    non_laugh_precision = precision(refsets[False], testsets[False])
    non_laugh_recall = recall(refsets[False], testsets[False])
    non_laugh_f1 = f_measure(refsets[False], testsets[False])

    acc = nltk.classify.accuracy(classifier, test_data)

    return [text, acc, laugh_precision, laugh_recall, laugh_f1, non_laugh_precision, non_laugh_recall, non_laugh_f1]


def runClassifiers(positives, negatives, featuresToUse, outFile, verbose, classifiersToUse):
    featureSets = []
    onDataSet = 0
    numDataSets = len(positives + negatives)
    table = []

    short = NUM_CLASSIFIERS - len(classifiersToUse)
    for x in range(short):
        classifiersToUse.append(False)

    # print which features we are using
    print("Using these features: ", FeatureExtractor.featuresToString(featuresToUse))

    for data in positives:
        featureSets.append((FeatureExtractor.langFeatures(data, featuresToUse), True))
        onDataSet += 1
        # printPercentage(onDataSet/numDataSets * 100, "Extracting Features: ", False)

    for data in negatives:
        featureSets.append((FeatureExtractor.langFeatures(data, featuresToUse), False))
        onDataSet += 1
        # printPercentage(onDataSet/numDataSets * 100, "Extracting Features: ", False)

    # sys.stdout.flush()
    # print("\n")

    # Testing is 1/4 of the data set, so we will cut it off there
    cutOff = len(featureSets)//4

    random.shuffle(featureSets)

    # splits training and test sets
    train_data, test_data = featureSets[cutOff:], featureSets[:cutOff]

    if classifiersToUse[0]:
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

    if classifiersToUse[1]:
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

    if classifiersToUse[2]:
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

    if classifiersToUse[3]:
        print("Running SVM classifier")
        timeStart = time.time()

        # Scikit-learn's SVC classifier, wrapped up in NLTK's wrapper class
        classifier = SklearnClassifier(SVC(), sparse=False).train(train_data)

        # get the time to train a Support Vector Machine
        print ("\nTime to train in seconds: ", time.time() - timeStart)

        # store the accuracy in the table
        table.append(assess_classifier(classifier, test_data, "Support Vector Machine"))

    if classifiersToUse[4]:
        print("Running AdaBoost classifier")
        timeStart = time.time()

        # Scikit-learn's AdaBoost classifier wrapped up in NLTK's wrapper class
        # The main parameters to tune to obtain good results are:
        # n_estimators and the complexity of the base estimators
        clf = AdaBoostClassifier()
        classifier = SklearnClassifier(clf, sparse=False).train(train_data)

        # get the time to train
        print ("\nTime to train in seconds: ", time.time() - timeStart)

        # store the accuracy in the table
        table.append(assess_classifier(classifier, test_data, "AdaBoost"))


    if (outFile == ""):
        print("\n", FeatureExtractor.featuresToString(featuresToUse))
        print(tabulate(table, headers=["Classifier", "accuracy", "laugh precision", "laugh recall", "laugh f1", "non-laugh precision", "non-laugh recall", "non-laugh f1"]))
    else:
        with open(outFile, 'a') as out:
            out.write("\n")
            out.write(FeatureExtractor.featuresToString(featuresToUse))
            out.write(tabulate(table, headers=["Classifier", "accuracy", "laugh precision", "laugh recall", "laugh f1", "non-laugh precision", "non-laugh recall", "non-laugh f1"]))
            out.write("\n")
