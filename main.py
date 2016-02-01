import Classifiers
import DataCreator
import random

# TODO Handle command line arguments. 1 for the amount of test files, another for if we are making a new meta file or not, and
# what classifier to use

positives = []
negatives = []

# determine if we are going to use pickled data or not
if True:
    print("Creating data\n")
    (positives, negatives) = DataCreator.createDataFrom("parsed_websites/", "Ted_Meta.txt", "Ted_Laughs.txt", False)
    print("Pickling data\n")
    DataCreator.pickleData("pickled_data/positives.p", positives)
    DataCreator.pickleData("pickled_data/negatives.p", negatives)
else:
    print("Using pickled files\n")
    positives = DataCreator.usePickledFile("pickled_data/positives.p")
    negatives = DataCreator.usePickledFile("pickled_data/negatives.p")


# random.shuffle(positives)
# random.shuffle(negatives)
# positives = positives[:3500]
# negatives = negatives[:3500]

# print("Extracting Features\n")

# # [every word in the text,
# #  ngram for words and characters,
# #  POS tag Personal Pronouns,
# #  Noun+adjective+verb percentage,
# #  Sentiment Analysis,
# #  Laugh Count Before This,
# #  Sentences since last laugh,
# #  Depth,
# #  length]
# featureSetsToUse = [
#     # [True, False, False, False, False, False, False, False, False],
#     # [False, True, False, False, False, False, False, False, False],
#     # [False, False, True, False, False, False, False, False, False],
#     # [False, False, False, True, False, False, False, False, False],
#     # [False, False, False, False, True, False, False, False, False],
#     # [False, False, False, False, False, True, False, False, False],
#     # [False, False, False, False, False, False, True, False, False],
#     # [False, False, False, False, False, False, False, True, False],
#     # [False, False, False, False, False, False, False, False, True],
#     # [True, True, True, True, True, True, True, True, False],
#     [True, False, True, True, True, True, True, True, True],
#     # [True, False, True, True, True, True, True, False, False],
#     # [False, False, True, True, True, True, True, True, False],
#     [True, False, True, True, True, False, False, False, True],
#     # [False, False, False, False, False, True, True, True, False],
# ]

# # featureSetsToUse = [
# #     # [False, False, False, False, False, True, True, True, False],
# #     # [True, False, True, True, True, True, True, True, False],
# #     [True, False, False, False, False, False, False, False]
# # ]

# # [Naive Bayes, Decision Tree, Max Entropy, Support Vector machine]
# classifiersToUse = [True, False, False, False, True]

# # positives, negatives, featuresToUse, whereToPrint, verbose, classifiersToUse
# for feats in featureSetsToUse:
#     Classifiers.runClassifiers(positives, negatives, feats, "output.txt", False, classifiersToUse)
