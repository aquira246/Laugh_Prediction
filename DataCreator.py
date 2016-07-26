import sys
import nltk, re, pprint
import math, random, statistics
import pickle
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

import TedMeta
import FeatureCollection
import FeatureExtractor
import loadingbar

CUT_OUTLIER_PERCENTAGE = .05
stop = stopwords.words('english')
stemmer = SnowballStemmer("english")
swears = ["ass", "fuck", "bitch", "shit", "butt", "dick", "clitoris", "cock",
          "vagina", "pussy", "boner", "blowjob", "blow job", "bastard", "cunt",
          "penis", "asshole", "shity", "shitty", "fucker", "motherfucker",
          "bullshit"]


def splitFile(md, splitBySentence):
    rf = open(md.filename, 'r')

    # skip first 6 lines since they aren't important
    talk = "\n".join(rf.readlines()[6:])

    # remove the Audio: Laughing and the applause
    talk = talk.replace("(Applause)", "")
    talk = talk.replace("(Audio: Laughing)", "")
    talk = talk.replace("-- (Laughter) --", " (Laughter) ")
    talk = talk.replace("-- (Laughter) ", " (Laughter) ")

    # remove hyphens for clarity
    talk = talk.replace("-", " ")

    chunks = []
    if splitBySentence:
        chunks = sent_tokenize(talk)       # the talk turned into sentences
    else:
        chunks = talk.split("\n")          # the talk is split by paragraph

    passedChunks = []                  # the pure (laughs removed) chunks passed
    passedWords = [["TS", "TS", "TS"]]  # prev chunks broken into stemmed/CC wds
    passedPOS = []                    # passed chunks broken into POS
    numChunks = len(chunks)             # the number of chunks in the talk
    chunksSinceLastLaugh = 0           # the number of chunks since last laugh
    laughCount = 0                    # the laughs counted so far
    positives = []                    # all of the positives
    negatives = []                    # all of the negatives
    previousSentiment = {"Polarity": 0}             # the previous chunk's sentiment

    for i in range(numChunks):
        # create a FeatureCollection for each chunk
        features = FeatureCollection.FeatureCollection(md.name)

        # if there is laughter in the chunk and it's not at the beginning OR
        # it is at the start of the next (if there is one) chunk
        if ("(Laughter)" in chunks[i][3:]) or (i != numChunks - 1 and ("(Laughter)" in chunks[i+1][:11])):
            features.positive = True
        else:
            features.positive = False

        # increase the distance since the last laugh
        chunksSinceLastLaugh += 1

        features.depth = i/numChunks  # Depth
        features.laughsUntilNow = laughCount  # Laugh Count Before This
        features.chunksSinceLaugh = chunksSinceLastLaugh  # Chunks since lastlaugh

        # remove the laughter from the chunk
        features.chunk = curChunk = chunks[i].replace("(Laughter)", "")

        # put this chunk at the end of the passed chunks list
        passedChunks.append(curChunk)

        # get sentiment
        features.sentimentFeats = FeatureExtractor.getSentiment(curChunk, previousSentiment)
        previousSentiment = features.sentimentFeats
        features.sentimentFeats["swearing"] = False

        words = word_tokenize(curChunk)

        # store the word vector
        if splitBySentence:
            features.wordVector = [[w.lower() for w in words if w.lower() not in stop and  w.isalpha()]]
        else:
            for s in sent_tokenize(curChunk):
                wordlist = [w.lower() for w in word_tokenize(s) if w.lower() not in stop and  w.isalpha()]
                features.wordVector.append(wordlist)

        # get Parts of speech features
        (_, pos) = FeatureExtractor.getPOS(words)
        passedPOS.append(pos)
        features.POS = pos

        # get length of chunk
        features.length = len(words)

        # set the last 3 words
        features.prev3Words = passedWords[-1][-3:]

        # case collapse and stem words and get the variance
        variousWords = {}
        for j in range(features.length):
            # move following line below if variance by stemmed words
            variousWords[words[j]] = True

            for swear in swears:
                if words[j].lower() == swear:
                    features.sentimentFeats["swearing"] = True
                    words[j] = "SWEARWORD"

            words[j] = stemmer.stem(words[j].lower())

        # calculate word variance
        if (features.length > 0):
            features.wordVariance = len(variousWords)/features.length
        else:
            features.wordVariance = 0

        # get words as features
        passedWords.append((words + ["EOS"]))
        features.words = words

        # if the chunk was positive thenwe need to:
        # reset the distance since last laugh and increment laugh count
        if features.positive:
            laughCount += 1
            chunksSinceLastLaugh = 0
            positives.append(features)
        else:
            negatives.append(features)

    rf.close

    return [positives, negatives]


def creatingData(metadata, lengths, useSentences):
    # cut out talks that are too long or short
    cutPoint = int(math.floor(len(metadata)*CUT_OUTLIER_PERCENTAGE))
    trimmedLengths = lengths[cutPoint:-cutPoint]

    if (len(metadata) < 40):
        trimmedLengths = lengths

    posData = []
    negData = []

    i = 0

    random.shuffle(trimmedLengths)
    for wordCount in trimmedLengths:
        for md in metadata:
            if md.wordCount == wordCount:
                matchedFile = md
                break

        metadata.remove(matchedFile)
        pieces = splitFile(matchedFile, useSentences)
        posData = posData + pieces[0]
        negData = negData + pieces[1]

        i += 1
        loadingbar.printPercentage(i/len(trimmedLengths) * 100, "Creating Data: ", False)

    print("Number of Positive Data: ", len(posData), "\n")
    print("Number of Negative Data: ", len(negData), "\n")

    return [posData, negData]


def createDataFrom(location, metadataFile, laughDataFile, createFiles, useSentences = True):
    if createFiles:
        print("Creating meta files\n")
        (metadata, lengths) = \
            TedMeta.createTedMetaFile(location, metadataFile, laughDataFile)
    else:
        print("Using previous meta files to get data\n")
        (metadata, lengths) = \
            TedMeta.useTedMetaFiles(location, metadataFile)

    print("Creating data\n")
    return creatingData(metadata, lengths, useSentences)


def pickleData(wf, data):
    pickle.dump(data, open(wf, "wb"))


def usePickledFile(rf):
    return pickle.load(open(rf, "rb"))
