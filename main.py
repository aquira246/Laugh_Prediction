import Classifiers
import DataCreator
import random


positives = []
negatives = []

# determine if we are going to use pickled data or not
if False:
    print("Creating data\n")
    (positives, negatives) = DataCreator.createDataFrom("parsed_websites/", "Ted_Meta.txt", "Ted_Laughs.txt", False)
    print("Pickling data\n")
    DataCreator.pickleData("pickled_data/positives.p", positives)
    DataCreator.pickleData("pickled_data/negatives.p", negatives)
else:
    print("Using pickled files\n")
    positives = DataCreator.usePickledFile("pickled_data/positives.p")
    negatives = DataCreator.usePickledFile("pickled_data/negatives.p")


random.shuffle(positives)
random.shuffle(negatives)
positives = positives[:3580]
negatives = negatives[:3580]

print("Extracting Features\n")

# [every word in the text,
#  ngram for words and characters,
#  POS tag Personal Pronouns,
#  Noun+adjective+verb percentage,
#  Sentiment Analysis,
#  Laugh Count Before This,
#  Sentences since last laugh,
#  Depth,
#  length]
featureSetsToUse = [
    # [True, False, False, False, False, False, False, False, False],
    # [True, True, False, False, False, False, False, False, False],
    # [False, False, True, False, False, False, False, False, False],
    # [False, False, False, True, False, False, False, False, False],
    # [False, False, False, False, True, False, False, False, False],
    # [False, False, False, False, False, True, False, False, False],
    # [False, False, False, False, False, False, True, False, False],
    # [False, False, False, False, False, False, False, True, False],
    # [False, False, False, False, False, False, False, False, True],
    # [True, False, True, True, False, True, True, True, True],
    [True, True, True, True, True, True, True, True, True],
    # [True, False, True, True, True, True, True, True, True],
    # [True, False, True, True, True, True, True, False, False],
    # [False, False, True, True, True, True, True, True, False],
    # [True, False, True, True, True, False, False, False, True],
    # [False, False, False, False, False, True, True, True, False],
    [True, True, True, True, True, False, False, False, True],
]

# featureSetsToUse = [
#     # [False, False, False, False, False, True, True, True, False],
#     # [True, False, True, True, True, True, True, True, False],
#     [True, False, False, False, False, False, False, False]
# ]

# [Naive Bayes, Decision Tree, Max Entropy, Support Vector machine, adaboost, random forest]
classifiersToUse = [False, False, False, False, True, False]

# positives, negatives, featuresToUse, whereToPrint, verbose, classifiersToUse
for feats in featureSetsToUse:
    Classifiers.runClassifiers(positives, negatives, feats, "output.txt", True, classifiersToUse)
