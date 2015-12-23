import sys
import nltk, re, pprint
import math, random, statistics
from nltk import word_tokenize
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
    divisions_amounts = [
    math.ceil(testCount*.001), math.floor(testCount*.021),\
    math.floor(testCount*.136), math.floor(testCount*.341),\
    math.floor(testCount*.341), math.floor(testCount*.136),\
    math.floor(testCount*.021), math.ceil(testCount*.001)]
    #divisions_amounts = [1, 2, 13, 34, 34, 13, 2, 1]
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

def trimFile(metadata):
    rf = open(metadata.filename, 'r')

    prevPara = ""
    prevIsPositive = False
    positives = []
    negatives = []


    for paragraph in rf:
        if re.match(r'^Title: .+', paragraph) is not None:
            pass
        elif re.match(r'^Author: .+', paragraph) is not None:
            pass
        elif re.match(r'^Tags: .+', paragraph) is not None:
            pass
        else:
            paragraph = paragraph.replace("(Applause)", "")
            paragraph = paragraph.replace("(Audio: Laughing)", "")

            if prevIsPositive or ("(Laughter)" in paragraph[:11] and prevPara != ""):
                positives.append(prevPara)
            else:
                if len(prevPara) > 2:
                    negatives.append(prevPara)

            prevIsPositive = False

            if "(Laughter)" in paragraph[10:]:
                prevIsPositive = True

            prevPara = paragraph.replace("(Laughter)", "")

    if prevIsPositive:
        positives.append(prevPara)
    else:
        if len(prevPara) > 2:
            negatives.append(prevPara)

    rf.close
    return [positives, negatives]


def divideDataIntoGroups(metadata, firstLaughs, lengthNonLaugh, numTestFiles):
    cutOutMeta = int(math.floor(len(metadata)*CUT_OUTLIER_PERCENTAGE))
    trimmedFirstLaughs = firstLaughs[cutOutMeta:-cutOutMeta]

    cutOutNonLaughs = int(math.floor(len(lengthNonLaugh)*CUT_OUTLIER_PERCENTAGE*2))
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

        segments = trimFile(posFile)
        posData = posData + segments[0]
        negData = negData + segments[1]
        segments = trimFile(negFile)
        negData = negData + segments[1]

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
