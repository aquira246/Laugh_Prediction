import sys
import nltk, re, pprint
import math, random, statistics
import pickle
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

# from TreeParser import getSubtreeFeatures

import TedMeta
import FeatureCollection
import FeatureExtractor
import loadingbar

CUT_OUTLIER_PERCENTAGE = .05
stop = stopwords.words('english') + ["__Person__", "__Quantity__", "__Year__", "__Value__", "__Statistic__"]
stemmer = SnowballStemmer("english")
swears = ["ass", "fuck", "bitch", "shit", "butt", "dick", "clitoris", "cock",
          "vagina", "pussy", "boner", "blowjob", "blow job", "bastard", "cunt",
          "penis", "asshole", "shity", "shitty", "fucker", "motherfucker",
          "bullshit", "bull shit", "damn", "darn", "sex"]


def entity_recognition(text):
        ret = text
        ret = re.sub(r'Mr\. (\w|-)+', '__Person__', ret)
        ret = re.sub(r'Ms\. (\w|-)+', '__Person__', ret)
        ret = re.sub(r'Mrs\. (\w|-)+', '__Person__', ret)
        ret = re.sub(r'(\d)+%', '__Statistic__', ret)
        ret = re.sub(r'(\w+ to )?\w+(\s|-)percent( |\,)', '__Statistic__ ', ret)
        ret = re.sub(r'(\w+ to )?\w+(\s|-)percent\.', '__Statistic__.', ret)
        ret = re.sub(r'\d+ (out)? of \d+', '__Statistic__', ret)
        ret = re.sub(r'(\d|\,)+\,\d\d\d', '__Quantity__', ret)
        ret = re.sub(r'\d\d\d\d(s)?', '__Year__', ret)
        ret = re.sub(r'\'\d\d(s)?', '__Year__', ret)
        ret = re.sub(r' \d{1,3}', ' __Quantity__', ret)
        ret = re.sub(r'\$\d+(\,\d+)*(\.\d+)*', '__Value__', ret)

        return ret


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

    # grab the frequency distribution
    talkFrequency = nltk.FreqDist(word_tokenize(talk.lower()))
    hapaxes = talkFrequency.hapaxes()
    hapaxes = [w for w in hapaxes if w.isalpha()]

    chunks = []
    if splitBySentence:
        chunks = sent_tokenize(talk)       # the talk turned into sentences
    else:
        chunks = talk.split("\n")          # the talk is split by paragraph

    passedChunks = []                   # the pure (laughs removed) chunks passed
    passedWords = [["TS", "TS", "TS"]]  # prev chunks broken into stemmed/CC wds
    passedPOS = []                      # passed chunks broken into POS
    numChunks = len(chunks)             # the number of chunks in the talk
    chunksSinceLastLaugh = 0            # the number of chunks since last laugh
    laughCount = 0                      # the laughs counted so far
    positives = []                      # all of the positives
    negatives = []                      # all of the negatives
    previousSentiment = {"Polarity": 0}             # the previous chunk's sentiment

    for i in range(numChunks):
        # create a FeatureCollection for each chunk
        features = FeatureCollection.FeatureCollection(md.name)

        # if there is laughter in the chunk and it's not at the beginning OR
        # it is at the start of the next (if there is one) chunk
        if ("(Laughter)" in chunks[i][3:]) or (i != numChunks - 1 and ("(Laughter)" in chunks[i+1][:12])):
            features.positive = True
        else:
            features.positive = False

        # remove the laughter from the chunk
        curChunk = chunks[i].replace("(Laughter)", "")
        features.chunk = curChunk

        # end here if there is nothing in the chunk besides laughter
        if len(curChunk) > 1:
            # increase the distance since the last laugh
            chunksSinceLastLaugh += 1

            features.features["depth"] = i/numChunks  # Depth
            features.features["laughsUntilNow"] = laughCount  # Laugh Count Before This
            features.features["chunksSinceLaugh"] = chunksSinceLastLaugh  # Chunks since lastlaugh

            # put this chunk at the end of the passed chunks list
            passedChunks.append(curChunk)

            # get sentiment
            features.sentimentFeats = FeatureExtractor.getSentiment(curChunk, previousSentiment)
            previousSentiment = features.sentimentFeats
            features.sentimentFeats["swearing"] = False

            # analyze sentence structure
            # (maxD, maxST, avgDepth, avgSubTrees) = getSubtreeFeatures(curChunk)
            # features["max_depth"] = maxD
            # features["max_subtree"] = maxST
            # features["avg_depth"] = avgDepth
            # features["avg_subtree"] = avgSubTrees

            # get Parts of speech features
            words = word_tokenize(curChunk)
            (_, pos) = FeatureExtractor.getPOS(words)
            passedPOS.append(pos)
            features.POS = pos

            # get length of chunk
            features.features["length"] = len(words)

            # replace quantites and years and some person names
            curChunk = entity_recognition(curChunk)
            features.features["statistic_count"] = curChunk.count("__Statistic__")

            # tokenize the words
            words = word_tokenize(curChunk)
            talklen = len(words)

            # store the word vector
            if splitBySentence:
                features.wordVector = [[w.lower() for w in words if w.lower() not in stop and  w.isalpha()]]
            else:
                for s in sent_tokenize(curChunk):
                    wordlist = [w.lower() for w in word_tokenize(s) if w.lower() not in stop and  w.isalpha()]
                    features.wordVector.append(wordlist)

            # set the last 3 words
            features.prev3Words = passedWords[-1][-3:]

            # case collapse and stem words and get the variance
            variousWords = {}
            for j in range(talklen):
                # move following line below if variance by stemmed words
                variousWords[words[j]] = True

                if words[j].lower() in swears:
                    features.sentimentFeats["swearing"] = True
                    # words[j] = "SWEARWORD"     # seems to have lowered accuracy

                words[j] = stemmer.stem(words[j].lower())

            # calculate word variance
            if (talklen > 0):
                features.wordVariance = len(variousWords)/talklen
            else:
                features.wordVariance = 0

            # get words as features
            passedWords.append((words + ["EOS"]))
            features.words = word_tokenize(curChunk)

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


def createDataFrom(location, metadataFile, laughDataFile, createFiles, useSentences=True):
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
    pickle.dump(data, open(wf, "wb"), protocol=2)


def usePickledFile(rf):
    return pickle.load(open(rf, "rb"))
