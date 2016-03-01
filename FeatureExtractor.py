import nltk
import numpy
import FeatureCollection
from nltk.tag import pos_tag, map_tag
from nltk.util import ngrams
from textblob import TextBlob

TOTAL_FEATURES = 11


# First define a function that produces features from a given object
# This function takes in a paragraph. It then breaks it up and uses it
# for the feature sets
# The features extracted for the talk are:
# 1. every word in the text
# 2. ngram for words and characters
# 3. POS tag
# 3A. Personal Pronouns
# 3B. Noun, adjective, and verb percentage
# 4. Sentiment Analysis
# 5. Laugh Count Before This
# 6. Sentences since last laugh
# 7. Depth
# 8. Length of the sentence
# 9. is question
# 10. has quotation
# Note that featuresets are lists. That's what the classifier takes as input
def langFeatures(featsCollection, featuresToUse):

    if (len(featuresToUse) < TOTAL_FEATURES):
        for x in range(TOTAL_FEATURES - len(featuresToUse)):
            featuresToUse.append(False)

    D = {}  # dictionary of keys

    # 1. every word in the text
    if featuresToUse[0]:
        # featureset of just words
        for word in featsCollection.words:
            # the feature list is the words in the script
            D[word] = True

    # 2. ngram for words and characters
    if featuresToUse[1]:
        # create char ngrams
        # text = featsCollection.prev3Words[-1][-1] + ". "\
        #         + featsCollection.sentence

        # cg = textToCharGrams(text)

        # create word ngrams
        wordList = featsCollection.prev3Words+featsCollection.words
        wg = textToWordGrams(wordList)

        # combine the ngrams
        allgrams = wg #+ cg

        for gram in allgrams:
            D[gram] = True

    # 3. POS tag
    if featuresToUse[2] or featuresToUse[3]:
        # POS tag based feature set
        # 3A. Personal Pronouns
        if featuresToUse[2]:
            D["Personal_Pronoun_Percentage"] = \
                featsCollection.POS["Personal_Pronoun_Percentage"]

        # 3B. Noun, adjective, and verb percentage
        if featuresToUse[3]:
            D["noun_percentage"] = featsCollection.POS["noun_percentage"]
            D["adj_percentage"] = featsCollection.POS["adj_percentage"]
            D["verb_percentage"] = featsCollection.POS["verb_percentage"]

            D["nouns"] = featsCollection.POS["nouns"]
            D["adjectives"] = featsCollection.POS["adjectives"]
            D["verbs"] = featsCollection.POS["verbs"]

    # 4. Sentiment Analysis
    if featuresToUse[4]:
        D["Subjectivity"] = featsCollection.subjectivity
        D["Polarity"] = featsCollection.polarity

        D["Polarity_Diff"] = featsCollection.sentimentFeats["Polarity_Diff"]

        D["Subjectivity_Bin"] = featsCollection.sentimentFeats["Subjectivity_Bin"]
        D["Polarity_Bin"] = featsCollection.sentimentFeats["Polarity_Bin"]
        D["Diff_Bin"] = featsCollection.sentimentFeats["Diff_Bin"]

    # 5. Laugh Count Before This
    if featuresToUse[5]:
        D["Laugh Count"] = featsCollection.laughsUntilNow

    # 6. Sentences since last laugh
    if featuresToUse[6]:
        D["Last Laugh"] = featsCollection.sentsSinceLaugh

    # 7. Depth
    if featuresToUse[7]:
        D["Depth"] = featsCollection.depth

    # 8. Length of the sentence
    if featuresToUse[8]:
        D["Length"] = featsCollection.length

    # 9. is question
    if featuresToUse[9]:
        if "?" in featsCollection.sentence and len(featsCollection.sentence) > 1:
            D["isQuestion"] = 1
        else:
            D["isQuestion"] = 0

    # 10. has quotation
    if featuresToUse[10]:
        if "\"" in featsCollection.sentence:
            D["hasQuote"] = 1
        else:
            D["hasQuote"] = 0

    return D


def featuresToString(featuresToUse):
    if (len(featuresToUse) < TOTAL_FEATURES):
        for x in range(TOTAL_FEATURES - len(featuresToUse)):
            featuresToUse.append(False)

    ret = ""

    if featuresToUse[0]:
        ret = ret + "Words, "

    if featuresToUse[1]:
        ret = ret + "Ngrams, "

    if featuresToUse[2]:
        ret = ret + "Personal Pronouns, "

    if featuresToUse[3]:
        ret = ret + "POS percentages, "

    if featuresToUse[4]:
        ret = ret + "Sentiment analysis, "

    if featuresToUse[5]:
        ret = ret + "Laughs previous, "

    if featuresToUse[6]:
        ret = ret + "Sentences since last laugh, "

    if featuresToUse[7]:
        ret = ret + "Depth, "

    if featuresToUse[8]:
        ret = ret + "Sentence Length, "

    if featuresToUse[9]:
        ret = ret + "Is Question, "

    if featuresToUse[10]:
        ret = ret + "Has Quote "

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
    if sent < -.1:
        return -1
    elif sent > .1:
        return 1
    else:
        return 0


def getSentiment(text, previousSentiment):
    D = {}
    testimonial = TextBlob(text)
    polarity = testimonial.sentiment.polarity
    D["Subjectivity"] = testimonial.sentiment.subjectivity
    D["Polarity"] = polarity

    diff = polarity - previousSentiment
    D["Polarity_Diff"] = diff

    D["Subjectivity_Bin"] = sentimentBin(testimonial.sentiment.subjectivity)
    D["Polarity_Bin"] = sentimentBin(polarity)
    D["Diff_Bin"] = sentimentBin(diff)

    return D


def getPOS(text):
    # POS tag based feature set
    # get the parts of speech tags
    parts_of_speech = nltk.pos_tag(text)

    ret = {}
    verbCount = 0
    nounCount = 0
    adjCount = 0
    prpCount = 0
    word_list = []

    for (word, pos) in parts_of_speech:
        word_list.append(word)

        if 'PRP' in pos:
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

    ret["Personal_Pronoun_Percentage"] = prpCount/wordCount
    ret["word_count"] = wordCount

    return (word_list, ret)
