import sys
import copy
import nltk, re, pprint
import math, random, statistics
from nltk import sent_tokenize
from nltk import word_tokenize
import TedMeta
import FeatureCollection

CUT_OUTLIER_PERCENTAGE = .05
MAX_DATA_COUNT = 500
POS_PARAGRAPHS_FROM_DATA = 3
NEG_PARAGRAPHS_FROM_DATA = 2


# TODO add parts of speech
def splitFile(md):
    rf = open(md.filename, 'r')

    # skip first 6 lines since they aren't important
    talk = " ".join(rf.readlines()[6:])

    # remove the Audio: Laughing and the applause
    talk = talk.replace("(Applause)", "")
    talk = talk.replace("(Audio: Laughing)", "")

    sents = sent_tokenize(talk)  # the talk turned into sentences
    passedSents = []             # all of the sentences passed
    numSents = len(sents)        # the number of sentences in the talk
    sentsSinceLastLaugh = 0      # the number of sentences since the last laugh
    laughCount = 0               # the laughs counted so far
    positives = []               # all of the positives
    negatives = []               # all of the negatives

    for i in range(numSents):
        # create a FeatureCollection for each sentence
        features = FeatureCollection.FeatureCollection(md.name)

        # if there is laughter in the sentence and it is not at the beginning OR
        # it is at the start of the next (if there is one) sentence
        if ("(Laughter)" in sents[i][3:]) or (i != numSents - 1 and ("(Laughter)" in sents[i+1][:11])):
            features.positive = True
        else:
            features.positive = False

        # remove the laughter from the sentence
        curSent = sents[i].replace("(Laughter)", "")

        # increase the distance since the last laugh
        sentsSinceLastLaugh += 1

        # put this sentence at the end of the passed sentences list
        passedSents.append(curSent)  # TODO might need to change it to a deep copy and not just a reference to curSent

        features.length = len(word_tokenize(curSent))
        features.depth = i/numSents
        features.laughsUntilNow = laughCount
        features.sentsSinceLaugh = sentsSinceLastLaugh
        features.sentences = copy.copy(passedSents)  # need a shallow copy so that we can modify passedSents
        features.numSents = len(features.sentences)

        # if the sentence was positive, we need to reset the distance since last laugh and increment laugh count
        if features.positive:
            laughCount += 1
            sentsSinceLastLaugh = 0
            positives.append(features)
        else:
            negatives.append(features)

    rf.close

    return [positives, negatives]


def creatingData(metadata, lengths):
    # cut out talks that are too long or short
    cutPoint = int(math.floor(len(metadata)*CUT_OUTLIER_PERCENTAGE))
    trimmedLengths = lengths[cutPoint:-cutPoint]

    if (len(metadata) < 40):
        trimmedLengths = lengths

    posData = []
    negData = []

    for wordCount in trimmedLengths:
        for md in metadata:
            if md.wordCount == wordCount:
                matchedFile = md
                break

        metadata.remove(matchedFile)
        pieces = splitFile(matchedFile)
        posData = posData + pieces[0]
        negData = negData + pieces[1]

    # comment out the below 5 lines when testing
    minLen = min(min(len(posData), len(negData)), MAX_DATA_COUNT)
    random.shuffle(posData)
    random.shuffle(negData)
    posData = posData[:minLen]
    negData = negData[:minLen]

    print("Number of Positive Data: ", len(posData), "\n")
    print("Number of Negative Data: ", len(negData), "\n")

    return [posData, negData]


def createDataFrom(location, metadataFile, laughDataFile, createFiles):
    if createFiles:
        print("Creating meta files\n")
        (metadata, lengths) = \
            TedMeta.createTedMetaFile(location, metadataFile, laughDataFile)
    else:
        print("Using previous meta files to get data\n")
        (metadata, lengths) = \
            TedMeta.useTedMetaFiles(location, metadataFile)

    print("Creating data\n")
    return creatingData(metadata, lengths)
