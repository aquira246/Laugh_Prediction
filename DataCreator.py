import sys
import nltk, re, pprint
import math, random, statistics
from nltk import word_tokenize
from urllib import request
import TedMeta

CUT_OUTLIER_PERCENTAGE = .05


def validNegativeLengths(pos, negs):

    if len(negs) < len(pos):
        print("Not enough Negs")
        return False

    for i in range(len(pos)):
        if negs[i] < pos[i]:
            return False

    return True


def getLaughLocations(firstLaughs, nonLaughs, testCount):
    fl_mean = statistics.mean(firstLaughs)
    fl_std_dev = statistics.stdev(firstLaughs, fl_mean)

    print("Mean " + str(fl_mean) + "\n")
    print("Std Dev " + str(fl_std_dev) + "\n")
    print ("laughs smallest location: " + str(firstLaughs[0]) + "\nLargest location: " + str(firstLaughs[-1]) + "\n")
    print ("non-laughs smallest length: " + str(nonLaughs[0]) + "\nLargest length: " + str(nonLaughs[-1]) + "\n")
    print ("non-laugh count: " + str(len(nonLaughs)) + "\n")

    if len(nonLaughs) < testCount:
        print("ERROR! Can't create data, need more non-laughs")
        return 0

    divisions_values = [fl_mean - fl_std_dev*3, fl_mean - fl_std_dev*2, fl_mean - fl_std_dev,\
    fl_mean, fl_mean + fl_std_dev, fl_mean + fl_std_dev*2, fl_mean + fl_std_dev*3]

    #TODO
    #math.ceil(testCount*.001), math.floor(testCount*.021),\
    #math.floor(testCount*.136), math.floor(testCount*.341),\
    #math.floor(testCount*.341), math.floor(testCount*.136),\
    #math.floor(testCount*.021), math.ceil(testCount*.001)
    divisions_amounts = [1, 2, 13, 34, 34, 13, 2, 1]
    #divisions_amounts = [1, 4, 27, 68, 68, 27, 4, 1]

    bottomGroup = []
    neg3q = []
    neg2q = []
    neg1q = []
    pos1q = []
    pos2q = []
    pos3q = []
    topGroup = []

    for x in firstLaughs:
        if x < divisions_values[0]:
            bottomGroup.append(x)
        elif x < divisions_values[1]:
            neg3q.append(x)
        elif x < divisions_values[2]:
            neg2q.append(x)
        elif x < divisions_values[3]:
            neg1q.append(x)
        elif x < divisions_values[4]:
            pos1q.append(x)
        elif x < divisions_values[5]:
            pos2q.append(x)
        elif x < divisions_values[6]:
            pos3q.append(x)
        else:
            topGroup.append(x)

    print (divisions_values)

    random.shuffle(bottomGroup)
    random.shuffle(neg3q)
    random.shuffle(neg2q)
    random.shuffle(neg1q)
    random.shuffle(pos1q)
    random.shuffle(pos2q)
    random.shuffle(pos3q)
    random.shuffle(topGroup)

    groups = [bottomGroup, neg3q, neg2q, neg1q, pos1q, pos2q, pos3q, topGroup]

    laughGroup = []
    hits = 0

    for i in range(0, len(divisions_amounts)):
        hits += divisions_amounts[i]

        if len(groups[i]) < hits:
            laughGroup = laughGroup + groups[i]
            hits -= len(groups[i])
        else:
            laughGroup = laughGroup + groups[i][:hits]
            hits = 0

    laughGroup.sort()
    return laughGroup

def trimFile(metadata, length):
    rf = open(metadata.filename, 'r')

    words = []
    wordCount = 0
    lastWord = ""

    words.append(metadata.name)

    for paragraph in rf:
        if re.match(r'^Title: .+', paragraph) is not None:
            pass
        elif re.match(r'^Author: .+', paragraph) is not None:
            pass
        elif re.match(r'^Tags: .+', paragraph) is not None:
            pass
        else:
            hitCount = False
            for word in nltk.word_tokenize(paragraph):
                wordCount += 1
                if (wordCount >= length):
                    hitCount = True

                if not (word is "Laughter" and lastWord is "("):
                    words.append(word)
                    if hitCount and word is ".":
                        break
                lastWord = word

    rf.close
    return words


def divideDataIntoGroups(metadata, firstLaughs, lengthNonLaugh, numTestFiles):
    cutOutMeta = math.floor(len(metadata)*CUT_OUTLIER_PERCENTAGE)
    trimmedFirstLaughs = firstLaughs[cutOutMeta:-cutOutMeta]

    cutOutNonLaughs = math.floor(len(lengthNonLaugh)*CUT_OUTLIER_PERCENTAGE*2)
    #if a talk is too long, all the better for the non laughs
    trimmedNonLaughs = lengthNonLaugh[cutOutNonLaughs:]

    laughLocs = getLaughLocations(trimmedFirstLaughs, trimmedNonLaughs, numTestFiles)
    laughLocs.sort()
    trimmedNonLaughs.sort()
    negativeLengths = trimmedNonLaughs[-(len(laughLocs)):]

    if not validNegativeLengths(laughLocs, negativeLengths):
        print("WARNING: Some of the negative laughs are not long enough")
        return []

    posNegMatch = []
    for i in range(len(laughLocs)):
        posNegMatch.append((laughLocs[i], negativeLengths[i]))

    posData = []
    negData = []

    for (pos, neg) in posNegMatch:
        for md in metadata:
            if md.firstLaughAt == pos:
                posFile = md
            elif md.firstLaughAt == -1 and md.wordCount == neg:
                negFile = md

        metadata.remove(posFile)
        metadata.remove(negFile)
        posData.append(trimFile(posFile, pos))
        negData.append(trimFile(negFile, neg))

    return (posData, negData)


def createDataFrom(location, metadataFile, laughDataFile, numTestFiles, createFiles):
    if createFiles:
        print("Creating meta files\n")
        (metadata, firstLaughs, lengthNonLaugh) = TedMeta.createTedMetaFile(location, metadataFile, laughDataFile)
    else:
        print("Using previous meta files to get data\n")
        (metadata, firstLaughs, lengthNonLaugh) = TedMeta.useTedMetaFiles(location, metadataFile, laughDataFile)

    print("Dividing data into groups\n")
    return divideDataIntoGroups(metadata, firstLaughs, lengthNonLaugh, numTestFiles)
