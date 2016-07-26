import numpy
import FeatureCollection
import Word2VecHelper

import nltk
from nltk.tag import pos_tag, map_tag
from nltk.util import ngrams
from textblob import TextBlob


model = Word2VecHelper.LoadModel()

# First define a function that produces features from a given object
# This function takes in a paragraph. It then breaks it up and uses it
# for the feature sets
# The features extracted for the talk are:
# 1. every word in the text
# 2. ngram for words and characters
# 3. POS tag
# 3A. Personal Pronouns and Proper Nouns per Noun
# 3B. Noun, adjective, and verb percentage
# 4. Sentiment Analysis
# 5. Laugh Count Before This
# 6. Sentences since last laugh
# 7. Depth
# 8. Length of the sentence
# 9A. is question
# 9B. is exclamation point
# 10. has quotation
# 11. word variance
# 12. incongruity approximation with word vectors
# 13. swear words
# Note that featuresets are lists. That's what the classifier takes as input
def langFeatures(featsCollection, featuresToUse):
    D = {}  # dictionary of keys

    # 1. every word in the text
    if "words" in featuresToUse and featuresToUse["words"]:
        # featureset of just words
        for word in featsCollection.words:
            # the feature list is the words in the script
            D[word] = True

    # 2. ngram for words and characters
    if "ngrams" in featuresToUse and featuresToUse["ngrams"]:
        # create char ngrams
        # text = featsCollection.prev3Words[-1][-1] + ". "\
        #         + featsCollection.chunk

        # cg = textToCharGrams(text)

        # create word ngrams
        wordList = featsCollection.prev3Words+featsCollection.words
        wg = textToWordGrams(wordList)

        # combine the ngrams
        allgrams = wg #+ cg

        for gram in allgrams:
            D[gram] = True

    # 3. POS tag
    if "pos_nouns" in featuresToUse or "pos_perc" in featuresToUse:
        # POS tag based feature set
        # 3A. Personal Pronouns and Proper Nouns per Noun
        if "pos_nouns" in featuresToUse and featuresToUse["pos_nouns"]:
            D["Personal_Pronoun_Percentage"] = \
                featsCollection.POS["Personal_Pronoun_Percentage"]

            D["Proper_Noun_Percentage"] = \
                featsCollection.POS["Proper_Noun_Percentage"]


        # 3B. Noun, adjective, and verb percentage
        if "pos_perc" in featuresToUse and featuresToUse["pos_perc"]:
            D["noun_percentage"] = featsCollection.POS["noun_percentage"]
            D["adj_percentage"] = featsCollection.POS["adj_percentage"]
            D["verb_percentage"] = featsCollection.POS["verb_percentage"]

            D["nouns"] = featsCollection.POS["nouns"]
            D["adjectives"] = featsCollection.POS["adjectives"]
            D["verbs"] = featsCollection.POS["verbs"]

    # 4. Sentiment Analysis
    if "sentiment" in featuresToUse and featuresToUse["sentiment"]:
        D["Subjectivity"] = featsCollection.sentimentFeats["Subjectivity"]
        D["Polarity"] = featsCollection.sentimentFeats["Polarity"]

        D["Polarity_Diff"] = featsCollection.sentimentFeats["Polarity_Diff"]

        D["Subjectivity_Bin"] = featsCollection.sentimentFeats["Subjectivity_Bin"]
        D["Polarity_Bin"] = featsCollection.sentimentFeats["Polarity_Bin"]
        D["Diff_Bin"] = featsCollection.sentimentFeats["Diff_Bin"]

    # 5. Laugh Count Before This
    if "laugh_count" in featuresToUse and featuresToUse["laugh_count"]:
        D["Laugh Count"] = featsCollection.laughsUntilNow

    # 6. Sentences since last laugh
    if "last_laugh" in featuresToUse and featuresToUse["last_laugh"]:
        D["Last Laugh"] = featsCollection.chunksSinceLaugh

    # 7. Depth
    if "depth" in featuresToUse and featuresToUse["depth"]:
        D["Depth"] = featsCollection.depth

    # 8. Length of the sentence
    if "length" in featuresToUse and featuresToUse["length"]:
        D["Length"] = featsCollection.length

    # 9A. is question
    if "question" in featuresToUse and featuresToUse["question"]:
        if "?" in featsCollection.chunk and len(featsCollection.chunk) > 1:
            D["isQuestion"] = 1
        else:
            D["isQuestion"] = 0

    # 9B. is question
    if "exclamation" in featuresToUse and featuresToUse["exclamation"]:
        if "!" in featsCollection.chunk and len(featsCollection.chunk) > 1:
            D["isExclamation"] = 1
        else:
            D["isExclamation"] = 0

    # 10. has quotation
    if "quote" in featuresToUse and featuresToUse["quote"]:
        if "\"" in featsCollection.chunk:
            D["hasQuote"] = 1
        else:
            D["hasQuote"] = 0

    # 11. word variance
    if "variance" in featuresToUse and featuresToUse["variance"]:
        D["Word_Variance"] = featsCollection.wordVariance

    # 12. incongruity approxmiation with Word Vectors.
    if "incongruity" in featuresToUse and featuresToUse["incongruity"]:
        # 1 is added to all results to remove the negative numbers
        # if "last_laugh" in featuresToUse and featuresToUse["last_laugh"]:
        (highest, lowest, avg) = getIncongruityFull(featsCollection.wordVector)
        D["Repitition_Full"] = highest + 1
        D["Disconnection_Full"] = lowest + 1
        D["Average_Full"] = avg + 1
        # else:
        (highest, lowest, avg) = getIncongruityPairs(featsCollection.wordVector)
        D["Repitition_Pairs"] = highest + 1
        D["Disconnection_Pairs"] = lowest + 1
        D["Average_Pairs"] = avg + 1

    # 13. Swear words
    if "swearing" in featuresToUse and featuresToUse["swearing"]:
        D["Swearing"] = featsCollection.sentimentFeats["swearing"]

    return D


def featuresToString(featuresToUse):
    ret = ""

    if "words" in featuresToUse and featuresToUse["words"]:
        ret = ret + "Words, "

    if "ngrams" in featuresToUse and featuresToUse["ngrams"]:
        ret = ret + "Ngrams, "

    if "pos_nouns" in featuresToUse and featuresToUse["pos_nouns"]:
        ret = ret + "Personal Pronouns and PN/N, "

    if "pos_perc" in featuresToUse and featuresToUse["pos_perc"]:
        ret = ret + "POS percentages, "

    if "sentiment" in featuresToUse and featuresToUse["sentiment"]:
        ret = ret + "Sentiment analysis, "

    if "laugh_count" in featuresToUse and featuresToUse["laugh_count"]:
        ret = ret + "Laughs previous, "

    if "last_laugh" in featuresToUse and featuresToUse["last_laugh"]:
        ret = ret + "Sentences since last laugh, "

    if "depth" in featuresToUse and featuresToUse["depth"]:
        ret = ret + "Depth, "

    if "length" in featuresToUse and featuresToUse["length"]:
        ret = ret + "Sentence Length, "

    if "question" in featuresToUse and featuresToUse["question"]:
        ret = ret + "Is Question, "

    if "exclamation" in featuresToUse and featuresToUse["exclamation"]:
        ret = ret + "Is Exclamation, "

    if "quote" in featuresToUse and featuresToUse["quote"]:
        ret = ret + "Has Quote, "

    if "variance" in featuresToUse and featuresToUse["variance"]:
        ret = ret + "Word Variance, "

    if "incongruity" in featuresToUse and featuresToUse["incongruity"]:
        ret = ret + "Incongruity, "

    if "swearing" in featuresToUse and featuresToUse["swearing"]:
        ret = ret + "Swearing, "

    if "Dim Reduction" in featuresToUse and featuresToUse["Dim Reduction"]:
        ret = ret + "Dim Reduction, "


    ret = ret + "\n"

    return ret


def textToCharGrams(text):
    char_text = list(text)
    char_bigrams = ngrams(char_text, 2)
    char_trigrams = ngrams(char_text, 3)
    char_quadgrams = ngrams(char_text, 4)

    # combine the ngrams
    allgrams = []

    for gram in char_bigrams:
        allgrams.append("".join(gram))

    for gram in char_trigrams:
        allgrams.append("".join(gram))

    for gram in char_quadgrams:
        allgrams.append("".join(gram))

    return allgrams


def textToWordGrams(words):
    word_bigrams = ngrams(words, 2)
    word_trigrams = ngrams(words, 3)
    # word_quadgrams = ngrams(words, 4)

    # combine the ngrams
    allgrams = []

    for gram in word_bigrams:
        allgrams.append("".join(gram))

    for gram in word_trigrams:
        allgrams.append("".join(gram))

    # for gram in word_quadgrams:
    #     allgrams.append("".join(gram))

    return allgrams


def sentimentBin(sent):
    if sent < .1:
        return 0
    elif sent > 1.1:
        return 2
    else:
        return 1


# 1 is added to each value to remove the negatives
# the polarity and subjectivity are initially -1 < x < 1
# now they are 0 < x < 2
def getSentiment(text, previousSentiment):
    D = {}
    testimonial = TextBlob(text)
    polarity = testimonial.sentiment.polarity + 1
    subjectivity = testimonial.sentiment.subjectivity + 1
    D["Subjectivity"] = subjectivity
    D["Polarity"] = polarity

    diff = polarity - previousSentiment["Polarity"]

    # 2 is add to prevent negatives
    D["Polarity_Diff"] = diff + 2

    D["Subjectivity_Bin"] = sentimentBin(testimonial.sentiment.subjectivity)
    D["Polarity_Bin"] = sentimentBin(polarity)

    if diff < -.1:
        D["Diff_Bin"] = 0
    elif diff > .1:
        D["Diff_Bin"] = 2
    else:
        D["Diff_Bin"] = 1

    return D


def getPOS(text):
    # POS tag based feature set
    # get the parts of speech tags
    parts_of_speech = nltk.pos_tag(text)

    ret = {}
    verbCount = 0
    nounCount = 0
    properNounCount = 0
    adjCount = 0
    prpCount = 0
    word_list = []
    punctuation = [".", ",", "!", "?", ";", ":", "\'", "\""]

    for (word, pos) in parts_of_speech:
        if word not in punctuation:
            word_list.append(word)

            if 'NNP' in pos:
                properNounCount += 1
            elif 'PRP' in pos:
                prpCount += 1


            # TODO possibly get social relationships

            # simplify the POS tag
            tag = map_tag('en-ptb', 'universal', pos)
            # increment pos counters
            if "NOUN" in tag:
                nounCount += 1
            elif "ADJ" in tag:
                adjCount += 1
            elif "VERB" in tag:
                verbCount += 1


    wordCount = len(word_list)

    # record the percentages the pos
    np = 0
    ap = 0
    vp = 0

    if (wordCount > 0):
        np = nounCount/wordCount
        ap = adjCount/wordCount
        vp = verbCount/wordCount

    # check the documentation for binning explanation
    # bin the nouns and add them to dictionary
    ret["nouns"] = np
    ret["adjectives"] = ap
    ret["verbs"] = vp

    if np < .145:
        ret["noun_percentage"] = 0
    elif np < .255:
        ret["noun_percentage"] = 1
    else:
        ret["noun_percentage"] = 2

    # bin the adjectives and add them to dictionary
    if ap < .028:
        ret["adj_percentage"] = 0
    elif ap < .096:
        ret["adj_percentage"] = 1
    else:
        ret["adj_percentage"] = 2

    # bin the verbs and add them to dictionary
    if vp < .13:
        ret["verb_percentage"] = 0
    elif vp < .22:
        ret["verb_percentage"] = 1
    else:
        ret["verb_percentage"] = 2

    if (wordCount > 0):
        ret["Personal_Pronoun_Percentage"] = prpCount/wordCount
    else:
        ret["Personal_Pronoun_Percentage"] = 0

    if (nounCount > 0):
        ret["Proper_Noun_Percentage"] = properNounCount/nounCount
    else:
        ret["Proper_Noun_Percentage"] = 0

    ret["word_count"] = wordCount

    return (word_list, ret)


def getIncongruityFull(wordVectors):
    highest = -1
    lowest = 1
    total = 0
    count = 0

    for curVec in wordVectors:
        for j in range(len(curVec) - 1):
            for k in range(j + 1, len(curVec)):
                a = curVec[j]
                b = curVec[k]

                if a != b:
                    try:
                        rating = Word2VecHelper.Similarity(model, a, b)
                    except KeyError:
                        rating = 2

                    if rating < 1:
                        highest = max(rating, highest)
                        lowest = min(rating, lowest)
                        total += rating
                        count += 1

    avg = 0 if count == 0 else total/count
    return (highest, lowest, avg)


def getIncongruityPairs(wordVectors):
    highest = -1
    lowest = 1
    total = 0
    count = 0

    for curVec in wordVectors:
        for j in range(len(curVec) - 1):
            a = curVec[j]
            b = curVec[j + 1]

            if a != b:
                try:
                    rating = Word2VecHelper.Similarity(model, a, b)
                except KeyError:
                    rating = 2

                if rating < 1:
                    highest = max(rating, highest)
                    lowest = min(rating, lowest)
                    total += rating
                    count += 1
    avg = 0 if count == 0 else total/count
    return (highest, lowest, avg)
