import random
import nltk
import DataCreator
import loadingbar
import numpy
import time
import collections
from loadingbar import printPercentage
from nltk import word_tokenize
from nltk.tag import pos_tag, map_tag
from nltk.util import ngrams

#First define a function that produces features from a given object
#This function takes in a paragraph. It then breaks it up and uses it
#for the feature sets
#The features extracted for the talk are:
#1. every word in the text
#2. ngram for words and characters
#3. POS tag
#4. Sentiment Analysis
#Note that featuresets are lists. That's what the classifier takes as input
def langFeatures(data):
    D = {} #dictionary of keys

    text = nltk.word_tokenize(data)

    wordCount = len(text)
    verbCount = 0
    nounCount = 0
    adjCount = 0

    if True:
        #featureset of just words
        for word in text:
            #the feature list is the words in the script
            D[word] = True

    if False:
        #create word ngrams
        word_bigrams = ngrams(text, 2)
        word_trigrams = ngrams(text, 3)
        word_quadgrams = ngrams(text, 4)
        #create character ngrams
        char_text = list(data)
        char_bigrams = ngrams(char_text, 2)
        char_trigrams = ngrams(char_text, 3)
        char_quadgrams = ngrams(char_text, 4)

        #combine the ngrams
        D["word_bigrams"] =  word_bigrams
        D["word_trigrams"] = word_trigrams
        D["word_quadgrams"] = word_quadgrams
        D["char_bigrams"] = char_bigrams
        D["char_trigrams"] = char_trigrams
        D["char_quadgrams"] = char_quadgrams

    if True:
        #POS tag based feature set
        #get the parts of speech tags
        parts_of_speech = nltk.pos_tag(text)
        for (word, pos) in parts_of_speech:
            #simplify the POS tag
            tag = map_tag('en-ptb', 'universal', pos)
            #increment pos counters
            if "NOUN" in tag:
                nounCount += 1
            elif "ADJ" in tag:
                adjCount += 1
            elif "VERB" in tag:
                verbCount += 1

        if wordCount == 0:
            D["Empty"] = True
            return D

        #record the percentages the pos
        np = nounCount/wordCount
        ap = adjCount/wordCount
        vp = verbCount/wordCount

        #check the documentation for binning explanation
        #bin the nouns and add them to dictionary
        if np < .145:
            D["noun_percentage"] = 0
        elif np < .255:
            D["noun_percentage"] = 1
        else:
            D["noun_percentage"] = 2

        #bin the adjectives and add them to dictionary
        if ap < .028:
            D["adj_percentage"] = 0
        elif ap < .096:
            D["adj_percentage"] = 1
        else:
            D["adj_percentage"] = 2

        #bin the verbs and add them to dictionary
        if vp < .13:
            D["verb_percentage"] = 0
        elif vp < .22:
            D["verb_percentage"] = 1
        else:
            D["verb_percentage"] = 2

    return D


def extractFeatures(positives, negatives, verbose, useBayes, useTree, useEntropy):
    featureSets = []
    onDataSet = 0
    numDataSets = len(positives + negatives)

    for data in positives:
        featureSets.append((langFeatures(data), True))
        onDataSet += 1
        printPercentage(onDataSet/numDataSets * 100, "Extracting Features: ")

    for data in negatives:
        featureSets.append((langFeatures(data), False))
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
        print ("Bayes Precision: ", nltk.metrics.precision(refsets[True], testsets[True]))
        print ("Bayes Recall: ", nltk.metrics.recall(refsets[True], testsets[True]))
        print ("Bayes F-Measure: ", nltk.metrics.f_measure(refsets[True], testsets[True]))

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
        print ("DTree Precision: ", nltk.metrics.precision(refsets[True], testsets[True]))
        print ("DTree Recall: ", nltk.metrics.recall(refsets[True], testsets[True]))
        print ("DTree F-Measure: ", nltk.metrics.f_measure(refsets[True], testsets[True]))

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
        print ("Entropy Precision: ", nltk.metrics.precision(refsets[True], testsets[True]))
        print ("Entropy Recall: ", nltk.metrics.recall(refsets[True], testsets[True]))
        print ("Entropy F-Measure: ", nltk.metrics.f_measure(refsets[True], testsets[True]))

        if verbose:
            #this is a nice function that reports the top most impactful features the NB classifier found
            print (classifier.show_most_informative_features(20))
            #this is a function that explains the effect of each feature in the set
            #print (classifier.explain())
