import sys
import nltk, re, pprint
import os
import statistics
from nltk import word_tokenize

#contains the Meta Data for a TED Talk
class tedMetaData(object):
    """docstring for metaData"""

    def __init__(self, fname = "", name = "", aut = "", lc = 0, wc = 0, laughtAt = -1, newTags = []):
        self.name = name
        self.author = aut
        self.numLaughs = lc
        self.wordCount = wc
        self.firstLaughAt = laughtAt
        self.tags = newTags
        self.filename = fname
        self.paraCount += 1

    def setName(self, title):
        self.name = title

    def setAuthor(self, aut):
        self.author = aut

    def setFileName(self, fname):
        self.filename = fname

    def setLaughtCount(self, lc):
        self.numLaughs = lc

    def setWordCount(self, wc):
        self.wordCount = wc

    def setParaCount(self, pc):
        self.paraCount = pc

    def setFirstLaugh(self, laughtAt):
        self.firstLaughAt = laughtAt

    def addTag(self, newTag):
        self.tags.append(newTag)

    def addMultipleTags(self, newTags):
        self.tags = self.tags + newTags

    def toString(self):
        return (self.filename + "    " + self.name + "    " + self.author + "    " + str(self.tags) +
            "    " + "Word Count: " + str(self.wordCount)  + "    " +
            "Laugh Count: " + str(self.numLaughs) + "    " +
            "First Laugh At: " + str(self.firstLaughAt))


#parses a file and gets the meta data
def getMetaDataForFile(filename):
    rf = open(filename, 'r')
    md = tedMetaData(filename)

    laughCount = 0
    wordCount = 0
    firstLaughAt = -1
    paragraphCount = 0

    for paragraph in rf:
        keepSearching = True
        paragraphCount += 1

        #grab title
        for title in re.findall(r'^Title: (.+)', paragraph):
            md.setName(title)
            keepSearching = False

        #grab author
        for aut in re.findall(r'^Author: (.+)', paragraph):
            md.setAuthor(aut)
            keepSearching = False

        #grab tags
        for tagGroup in re.findall(r'^Tags: (.+)', paragraph):
            md.addMultipleTags(tagGroup.split(", "))
            keepSearching = False

        #count words and laughter, and when the first laughter is recorded
        if keepSearching:
            for word in nltk.word_tokenize(paragraph):
                wordCount += 1
                if 'Laughter' in word:
                    laughCount += 1
                    if firstLaughAt is -1:
                        firstLaughAt = wordCount

    rf.close
    md.firstLaughAt = firstLaughAt
    md.wordCount = wordCount
    md.numLaughs = laughCount
    md.paraCount = paragraphCount - laughCount

    return md

#Goes to path and reads the parsed TED Talks
#Then makes a collection of tedMetaData from the TED Talks in path
#stores metadata about the laughter in the TED Talks at laughDataLocation
def createTedMetaFile(path, laughDataLocation):

    #the file that stores the laugh meta data
    lf = open(laughDataLocation, "w")

    numFiles = 0

    firstLaughs = []
    firstLaughsPercent = []
    laughCounters = [0]
    avgLaughCount = 0
    highestLaughs = 0
    linesToWrite = []
    numWithoutLaughs = 0
    metadataCollection = []
    totalParas = 0
    totalWords = 0

    posLengths = []
    negLengths = []

    for filename in os.listdir(path):
        fileToCheck = path+filename
        md = getMetaDataForFile(fileToCheck)
        metadataCollection.append(md)

        if md.name != "":
            numFiles += 1
            linesToWrite.append(md.toString() + "\n")

            if md.firstLaughAt != -1:
                firstLaughs.append(md.firstLaughAt)
                firstLaughsPercent.append(md.firstLaughAt/md.wordCount)
                posLengths.append(md.wordCount)
            else:
                numWithoutLaughs += 1
                negLengths.append(md.wordCount)

            if md.numLaughs > highestLaughs:
                for i in range(md.numLaughs - highestLaughs):
                    laughCounters.append(0)

                highestLaughs = md.numLaughs

            laughCounters[md.numLaughs] += 1
            avgLaughCount += md.numLaughs
            totalParas += md.paraCount
            totalWords += md.wordCount

    linesToWrite.sort()
    firstLaughs.sort()
    firstLaughsPercent.sort()
    negLengths.sort()
    posLengths.sort()

    lf.writelines("Number of files with laughs: " + str(numFiles - numWithoutLaughs) + "\n")
    lf.writelines("Number of files without Laughs: " + str(numWithoutLaughs) + "\n\n")

    lf.writelines("Average Paragraphs: " + str(totalParas/numFiles) + "\n")
    lf.writelines("Average Words: " + str(totalWords/numFiles) + "\n\n")

    lf.writelines("Total Paragraphs: " + str(totalParas) + "\n")
    lf.writelines("Total Words: " + str(totalWords) + "\n\n")

    lf.writelines("Total Laugh Count: " + str(avgLaughCount) + "\n")
    lf.writelines("Average Laugh Count (only for files with laughter): " + str(avgLaughCount/(numFiles - numWithoutLaughs)) + "\n")
    lf.writelines("Average Laugh Location: " + str(statistics.mean(firstLaughs)) + "\n")
    lf.writelines("Average Laugh Location by Percent: " + str(statistics.mean(firstLaughsPercent)) + "\n\n")

    lf.writelines("standard deviation of first laugh location: " + str(statistics.stdev(firstLaughs)) + "\n")
    lf.writelines("standard deviation of first laugh percent : " + str(statistics.stdev(firstLaughsPercent)) + "\n\n")

    lf.writelines("Median Low Laugh Location: " + str(statistics.median_low(firstLaughs)) + "\n")
    lf.writelines("Median Low Laugh Location by Percent: " + str(statistics.median_low(firstLaughsPercent)) + "\n\n")

    lf.writelines("Laugh Counts: \n")
    for i in range(len(laughCounters)):
        lf.writelines("    " + str(i) + ": " + str(laughCounters[i]) + "\n")

    lf.writelines("\nLaugh Locations: \n")
    for fl in firstLaughs:
        lf.writelines("    " + str(fl))

    lf.writelines("\nLaugh Locations by percent down: \n")
    for fl in firstLaughsPercent:
        lf.writelines("    " + str(fl))

    lf.writelines("\nWord count for non laugh scripts\n")
    for line in negLengths:
        lf.writelines("    " + str(line))

    lf.close

    lengths = posLengths + negLengths
    lengths.sort()

    return (metadataCollection, lengths)

createTedMetaFile("parsed_websites/", "Ted_Laughs.txt")
