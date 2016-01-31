import sys
import nltk, re, pprint
import math, random, statistics
import pickle
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import TedMeta
import FeatureCollection
import FeatureExtractor
import loadingbar

CUT_OUTLIER_PERCENTAGE = .05

SENTENCES_CARED_ABOUT = 2

def splitFile(md):
    rf = open(md.filename, 'r')

    # skip first 6 lines since they aren't important
    talk = " ".join(rf.readlines()[6:])

    # remove the Audio: Laughing and the applause
    talk = talk.replace("(Applause)", "")
    talk = talk.replace("(Audio: Laughing)", "")

    sents = sent_tokenize(talk)  # the talk turned into sentences
    passedSents = []             # the pure (laughs removed) sentences passed
    passedWords = []             # prev sents broken into stemmed/CC words
    passedPOS = []               # passed sents broken into POS
    passedWordNgrams = []        # passed sents broken into word ngrams
    passedCharNgrams = []        # passed sents broken into char ngrams
    numSents = len(sents)        # the number of sentences in the talk
    sentsSinceLastLaugh = 0      # the number of sentences since the last laugh
    laughCount = 0               # the laughs counted so far
    positives = []               # all of the positives
    negatives = []               # all of the negatives

    for i in range(numSents):
        # create a FeatureCollection for each sentence
        features = FeatureCollection.FeatureCollection(md.name)

        # if there is laughter in the sentence and it's not at the beginning OR
        # it is at the start of the next (if there is one) sentence
        if ("(Laughter)" in sents[i][3:]) or (i != numSents - 1 and ("(Laughter)" in sents[i+1][:11])):
            features.positive = True
        else:
            features.positive = False

        # increase the distance since the last laugh
        sentsSinceLastLaugh += 1

        features.depth = i/numSents  # Depth
        features.laughsUntilNow = laughCount  # Laugh Count Before This
        features.sentsSinceLaugh = sentsSinceLastLaugh  # Sents since lastlaugh

        # remove the laughter from the sentence
        curSent = sents[i].replace("(Laughter)", "")

        # put this sentence at the end of the passed sentences list
        passedSents.append(curSent)  # TODO might need to change it to a deep copy and not just a reference to curSent

        # TODO named entities and named Entity count

        # get char-ngrams
        charGrams = FeatureExtractor.textToCharGrams(curSent)
        passedCharNgrams.append(charGrams)
        allGrams = passedCharNgrams[-SENTENCES_CARED_ABOUT:]
        features.charNgrams = []

        for grams in allGrams:
            features.charNgrams = features.charNgrams + grams

        # get sentiment
        sentiment = FeatureExtractor.getSentiment(curSent)
        features.subjectivity = sentiment["Subjectivity"]
        features.polarity = sentiment["Polarity"]

        # get Parts of speech features
        pos = FeatureExtractor.getPOS(curSent)
        passedPOS.append(pos)

        # TODO, get POS for more than just the curSent
        features.POS = pos

        # tokenize string
        words = word_tokenize(curSent)

        # case collapse and stem words
        stemmer = SnowballStemmer("english")

        for i in range(len(words)):
            words[i] = words[i].lower()
            words[i] = stemmer.stem(words[i])

        # get words as features
        passedWords.append(words)

        allWords = passedWords[-SENTENCES_CARED_ABOUT:]
        features.words = []

        for w in allWords:
            features.words = features.words + w

        # get word ngrams
        wordGrams = FeatureExtractor.textToWordGrams(words)
        passedWordNgrams.append(wordGrams)
        allGrams = passedWordNgrams[-SENTENCES_CARED_ABOUT:]
        features.wordNgrams = []

        for grams in allGrams:
            features.wordNgrams = features.wordNgrams + grams

        # get length of sentence
        features.length = len(words)

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

    i = 0

    for wordCount in trimmedLengths:
        for md in metadata:
            if md.wordCount == wordCount:
                matchedFile = md
                break

        metadata.remove(matchedFile)
        pieces = splitFile(matchedFile)
        posData = posData + pieces[0]
        negData = negData + pieces[1]

        i += 1
        loadingbar.printPercentage(i/len(trimmedLengths) * 100, "Creating Data: ", False)

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


def pickleData(wf, data):
    pickle.dump(data, open(wf, "wb"))


def usePickledFile(rf):
    return pickle.load(open(rf, "rb"))
