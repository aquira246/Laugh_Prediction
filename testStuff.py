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
    (positives, negatives) = getData(True, False)

    featureSetsToUse = {}
    featureSetsToUse["words"] = True          # every word in the text
    featureSetsToUse["ngrams"] = True         # ngram for words and characters
    featureSetsToUse["pos_nouns"] = True      # POS tag Personal Pronouns and Proper Nouns per Noun
    featureSetsToUse["pos_perc"] = True       # Noun+adjective+verb percentage
    featureSetsToUse["sentiment"] = True      # Sentiment Analysis
    featureSetsToUse["laugh_count"] = True    # Laugh Count Before This
    featureSetsToUse["last_laugh"] = True     # Chunks since last laugh
    featureSetsToUse["depth"] = True          # Depth
    featureSetsToUse["length"] = True         # length
    featureSetsToUse["question"] = True       # there is a question mark
    featureSetsToUse["quote"] = True          # isQuote
    featureSetsToUse["variance"] = True       # word variance
    featureSetsToUse["incongruity"] = True    # incongruity

    wf = open("deleteme.txt", 'w')

    wf.writelines("Sentence   |||   proper noun percentage, word variance, prp, wordCount\n")
    wf.writelines("========================================================================\n\n")

    for data in positives:
        wf.writelines(data.chunk + "\n")
        for wv in data.wordVector:
            wf.writelines(", ".join(wv))
            wf.writelines("\n")

        wf.writelines("\n")
        feats = FeatureExtractor.langFeatures(data, featureSetsToUse)

        # for i in feats.keys():
        #     wf.writelines(i + " ")
        #     wf.writelines(str(feats[i]))
        #     wf.writelines("   ")

        # wf.writelines("\n")

    wf.writelines("========================================================================\n")

    for data in negatives:
        wf.writelines(data.chunk + "\n")
        for wv in data.wordVector:
            wf.writelines(", ".join(wv))
            wf.writelines("\n")
        wf.writelines("\n")
        feats = FeatureExtractor.langFeatures(data, featureSetsToUse)

        # for i in feats.keys():
        #     wf.writelines(i + " ")
        #     wf.writelines(str(feats[i]))
        #     wf.writelines("   ")

        # wf.writelines("\n")

    wf.close
