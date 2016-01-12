import nltk
import loadingbar
import numpy
import FeatureCollection
from nltk import word_tokenize
from nltk.tag import pos_tag, map_tag
from nltk.util import ngrams
from textblob import TextBlob

# First define a function that produces features from a given object
# This function takes in a paragraph. It then breaks it up and uses it
# for the feature sets
# The features extracted for the talk are:
# 1. every word in the text
# 2. ngram for words and characters
# 3. POS tag
# 4. Sentiment Analysis
# 5. Laugh Count Before This
# 6. Sentences since last laugh
# 7. Depth
#Note that featuresets are lists. That's what the classifier takes as input
def langFeatures(featsCollection):
    D = {} #dictionary of keys

    lastSents = featsCollection.sentences[-1]
    text = nltk.word_tokenize(lastSents)

    wordCount = len(text)
    verbCount = 0
    nounCount = 0
    adjCount = 0

    # 1. every word in the text
    if True:
        #featureset of just words
        for word in text:
            #the feature list is the words in the script
            D[word] = True

    # 2. ngram for words and characters
    if False:
        #create word ngrams
        word_bigrams = ngrams(text, 2)
        word_trigrams = ngrams(text, 3)
        word_quadgrams = ngrams(text, 4)
        #create character ngrams
        char_text = list(lastSents)
        char_bigrams = ngrams(char_text, 2)
        char_trigrams = ngrams(char_text, 3)
        char_quadgrams = ngrams(char_text, 4)

        #combine the ngrams
        D["word_bigrams"] =  word_bigrams
        D["word_trigrams"] = word_trigrams
        D["word_quadgrams"] = word_quadgrams
        D["char_bigrams"] = char_bigrams
        D["char_trigrams"] = char_trigrams
        D["char_quadgrams"] = char_quadgrams

    # 3. POS tag
    if True:
        #POS tag based feature set
        #get the parts of speech tags
        parts_of_speech = nltk.pos_tag(text)
        for (word, pos) in parts_of_speech:
            #simplify the POS tag
            tag = map_tag('en-ptb', 'universal', pos)
            #increment pos counters
            if "NOUN" in tag:
                nounCount += 1
            elif "ADJ" in tag:
                adjCount += 1
            elif "VERB" in tag:
                verbCount += 1

        if wordCount == 0:
            D["Empty"] = True
            return D

        #record the percentages the pos
        np = nounCount/wordCount
        ap = adjCount/wordCount
        vp = verbCount/wordCount

        #check the documentation for binning explanation
        #bin the nouns and add them to dictionary
        if np < .145:
            D["noun_percentage"] = 0
        elif np < .255:
            D["noun_percentage"] = 1
        else:
            D["noun_percentage"] = 2

        #bin the adjectives and add them to dictionary
        if ap < .028:
            D["adj_percentage"] = 0
        elif ap < .096:
            D["adj_percentage"] = 1
        else:
            D["adj_percentage"] = 2

        #bin the verbs and add them to dictionary
        if vp < .13:
            D["verb_percentage"] = 0
        elif vp < .22:
            D["verb_percentage"] = 1
        else:
            D["verb_percentage"] = 2

    # 4. Sentiment Analysis
    if False:
        testimonial = TextBlob(lastSents)
        D["Subjectivity"] = testimonial.sentiment.subjectivity
        D["Polarity"] = testimonial.sentiment.polarity

    # 5. Laugh Count Before This
    if False:
        D["Laugh Count"] = featsCollection.laughsUntilNow

    # 6. Sentences since last laugh
    if False:
        D["Last Laugh"] = featsCollection.sentsSinceLaugh

    # 7. Depth
    if False:
        D["Depth"] = featsCollection.depth

    return D
