import FeatureExtractor
import DataCreator


def getData(usePickled):
    positives = []
    negatives = []

    # determine if we are going to use pickled data or not
    if usePickled:
        print("Using pickled files\n")
        positives = DataCreator.usePickledFile("pickled_data/positives.p")
        negatives = DataCreator.usePickledFile("pickled_data/negatives.p")
    else:
        print("Creating data\n")
        (positives, negatives) = DataCreator.createDataFrom("parsed_websites/", "Ted_Meta.txt", "Ted_Laughs.txt", False)
        print("Pickling data\n")
        DataCreator.pickleData("pickled_data/positives.p", positives)
        DataCreator.pickleData("pickled_data/negatives.p", negatives)

    return(positives, negatives)



"""MAIN"""
if __name__ == '__main__':
    (positives, negatives) = getData(True)

    featuresToUse = [False, False, False, False, False, False, False, False, False, True, True]

    wf = open("deleteme.txt", 'w')
    wf.writelines("Question\n")

    for data in positives[0:25]:
        feats = FeatureExtractor.langFeatures(data, featuresToUse)

        if feats["isQuestion"] == 0:
            wf.writelines(data.sentence + "\n")

    wf.writelines("\n=============================================================================\nQuote:\n")

    for data in positives[0:25]:
        feats = FeatureExtractor.langFeatures(data, featuresToUse)

        if feats["hasQuote"] == 0:
            wf.writelines(data.sentence + "\n")

    wf.close
