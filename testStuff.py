import nltk

from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.stem.snowball import SnowballStemmer
import pickle

stop = stopwords.words('english')
stemmer = SnowballStemmer("english")

import FeatureExtractor
import DataCreator


def getData(usePickled=False, useSentences = True):
    positives = []
    negatives = []

    # determine if we are going to use pickled data or not
    if usePickled:
        print("Using pickled files\n")
        positives = DataCreator.usePickledFile("pickled_data/test_positives.p")
        negatives = DataCreator.usePickledFile("pickled_data/test_negatives.p")
    else:
        print("Creating data\n")
        (positives, negatives)=DataCreator.createDataFrom("parsed_websites/", "Ted_Meta_testing.txt", "Ted_Laughs.txt", False, useSentences)
        print("Pickling data\n")
        DataCreator.pickleData("pickled_data/test_positives.p", positives)
        DataCreator.pickleData("pickled_data/test_negatives.p", negatives)

    return(positives, negatives)



"""MAIN"""
if __name__ == '__main__':
    (positives, negatives) = getData(False, False)

    featureSetsToUse = {}
    featureSetsToUse["words"] = True          # every word in the text
    featureSetsToUse["ngrams"] = False         # ngram for words and characters
    featureSetsToUse["pos_nouns"] = False      # POS tag Personal Pronouns and Proper Nouns per Noun
    featureSetsToUse["pos_perc"] = False       # Noun+adjective+verb percentage
    featureSetsToUse["sentiment"] = False      # Sentiment Analysis
    featureSetsToUse["laugh_count"] = False    # Laugh Count Before This
    featureSetsToUse["last_laugh"] = False     # Chunks since last laugh
    featureSetsToUse["depth"] = False          # Depth
    featureSetsToUse["length"] = False         # length
    featureSetsToUse["question"] = False       # there is a question mark
    featureSetsToUse["exclamation"] = False    # there is a exclamation mark
    featureSetsToUse["quote"] = False          # isQuote
    featureSetsToUse["variance"] = False       # word variance
    featureSetsToUse["incongruity"] = False    # incongruity
    featureSetsToUse["swearing"] = True       # swearing
    featureSetsToUse["statistics"] = True     # counting statistics
    featureSetsToUse["frequency"] = True      # frequency statistics
    featureSetsToUse["hapax"] = True          # hapax count


    wf = open("deleteme.txt", 'w')

    wf.writelines("Sentence   |||   proper noun percentage, word variance, prp, wordCount\n")
    wf.writelines("========================================================================\n\n")

    for data in positives:
        wf.writelines(data.chunk + "\n")
        # wf.writelines(str(data.features))

        # wf.writelines("\n")
        feats = FeatureExtractor.langFeatures(data, featureSetsToUse)

        for i in feats.keys():
            wf.writelines(i + " ")
            wf.writelines(str(feats[i]))
            wf.writelines("   ")

        wf.writelines("\n")

    wf.writelines("========================================================================\n")

    for data in negatives:
        wf.writelines(data.chunk + "\n")
        # wf.writelines(str(data.features))
        # wf.writelines("\n")
        feats = FeatureExtractor.langFeatures(data, featureSetsToUse)

        for i in feats.keys():
            wf.writelines(i + " ")
            wf.writelines(str(feats[i]))
            wf.writelines("   ")

        wf.writelines("\n")

    wf.close
