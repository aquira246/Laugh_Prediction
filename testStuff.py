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

    featuresToUse = [True, False, False, False, False, False, False, False, False, False, False, False]

    wf = open("deleteme.txt", 'w')

    wf.writelines("Sentence   |||   proper noun percentage, word variance, prp, wordCount\n")
    wf.writelines("========================================================================\n\n")

    for data in positives:
        wf.writelines(data.sentence + "\n")
        # feats = FeatureExtractor.langFeatures(data, featuresToUse)

        # for i in feats.keys():
        #     wf.writelines(i + " ")

        # wf.writelines("\n")

    wf.writelines("========================================================================\n")

    for data in negatives:
        wf.writelines(data.sentence + "\n")
        # feats = FeatureExtractor.langFeatures(data, featuresToUse)

        # for i in feats.keys():
        #     wf.writelines(i + " ")

        # wf.writelines("\n")

    wf.close
