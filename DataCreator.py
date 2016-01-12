import sys
import nltk, re, pprint
import math, random, statistics
from nltk import sent_tokenize
import TedMeta
import FeatureCollection

CUT_OUTLIER_PERCENTAGE = .05
MAX_DATA_COUNT = 500
POS_PARAGRAPHS_FROM_DATA = 3
NEG_PARAGRAPHS_FROM_DATA = 2

def splitFile(metadata):
    rf = open(metadata.filename, 'r')

    talk = " ".join(rf[6:])
    sents = nltk.sent_tokenize(talk)

    rf.close
    return [positives, negatives]


def matchMetaWithLengths(metadata, lengths):
    ret = []

    for wordCount in lengths:
        for md in metadata:
            if md.wordCount == wordCount:
                matchedFile = md

        metadata.remove(matchedFile)
        ret.append(splitFile(matchedFile))

    return ret


def generateDataGroups(fromPosData, fromNegData):
    random.shuffle(fromPosData)
    random.shuffle(fromNegData)

    posData = []
    negData = []

    # numPosData = MAX_DATA_COUNT
    # numNegData = MAX_DATA_COUNT

    # for data in fromPosData:
    #     if numPosData <= 0:
    #         break

    #     numData = min(len(data[0]), POS_PARAGRAPHS_FROM_DATA)
    #     paragraphs = random.shuffle(data[0])
    #     posData = posData + paragraphs[:numData]
    #     numPosData -= numData

    #     numData = min(len(data[1]), NEG_PARAGRAPHS_FROM_DATA)
    #     paragraphs = random.shuffle(data[1])
    #     negData = negData + paragraphs[:numData]
    #     numNegData -= numData

    # for data in fromNegData:
    #     if numNegData <= 0:
    #         break

    #     numData = min(len(data[1]), NEG_PARAGRAPHS_FROM_DATA)
    #     paragraphs = random.shuffle(data[1])
    #     negData = negData + paragraphs[:numData]
    #     numNegData -= numData

    # if numPosData > 0 or numNegData > 0:
    #     print("ERROR! Invalid number of positive or negative data")

    for data in fromPosData + fromNegData:
        posData = posData + data[0]
        negData = negData + data[0]

    minLen = min(min(len(posData), len(negData)), MAX_DATA_COUNT)
    random.shuffle(posData)
    random.shuffle(negData)
    posData = posData[:minLen]
    negData = negData[:minLen]

    print("Number of Positive Data: ", len(posData), "\n")
    print("Number of Negative Data: ", len(negData), "\n")

    return [posData, negData]


def splitData(metadata, posLengths, lengthNonLaugh):
    #cut out talks that are too long or short
    cutPoint = int(math.floor(len(metadata)*CUT_OUTLIER_PERCENTAGE))
    trimmedPosLengths = posLengths[cutPoint:-cutPoint]

    cutPoint = int(math.floor(len(lengthNonLaugh)*CUT_OUTLIER_PERCENTAGE))
    trimmedNonLaughs = lengthNonLaugh[cutPoint:-cutPoint]

    negMetaData = []
    posMetaData = []

    for md in metadata:
        if md.firstLaughAt == -1:
            negMetaData.append(md)
        else:
            posMetaData.append(md)

    fromPosData = matchMetaWithLengths(posMetaData, trimmedPosLengths)
    fromNegData = matchMetaWithLengths(negMetaData, trimmedNonLaughs)



    return generateDataGroups(fromPosData, fromNegData)


def createDataFrom(location, metadataFile, laughDataFile, createFiles):
    if createFiles:
        print("Creating meta files\n")
        (metadata, posLengths, negLengths) = \
            TedMeta.createTedMetaFile(location, metadataFile, laughDataFile)
    else:
        print("Using previous meta files to get data\n")
        (metadata, posLengths, negLengths) = \
            TedMeta.useTedMetaFiles(location, metadataFile, laughDataFile)

    print("Dividing data into groups\n")
    return splitData(metadata, posLengths, negLengths)
