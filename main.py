import Classifiers
import DataCreator

#TODO Handle command line arguments. 1 for the amount of test files, another for if we are making a new meta file or not, and
#what classifier to use
(positives, negatives) = DataCreator.createDataFrom("parsed_websites/", "Ted_Meta.txt", "Ted_Laughs.txt", False)
print("Extracting Features\n")

# [every word in the text,
#  ngram for words and characters,
#  POS tag Personal Pronouns,
#  Noun+adjective+verb percentage,
#  Sentiment Analysis,
#  Laugh Count Before This,
#  Sentences since last laugh,
#  Depth]
# featureSetsToUse = [
#     [True, False, False, False, False, False, False, False],
#     [False, True, False, False, False, False, False, False],
#     [False, False, True, False, False, False, False, False],
#     [False, False, False, True, False, False, False, False],
#     [False, False, False, False, True, False, False, False],
#     [False, False, False, False, False, True, False, False],
#     [False, False, False, False, False, False, True, False],
#     [False, False, False, False, False, False, False, True],
# ]

# featureSetsToUse = [
#     [True, True, True, True, True, True, True, True],
#     [True, False, True, True, True, True, True, True],
#     [True, False, True, True, True, True, True, False],
#     [False, False, True, True, True, True, True, True],
#     [False, False, False, False, False, True, True, True],
# ]

featureSetsToUse = [
    [False, False, False, False, False, True, True, True],
    [True, False, True, True, True, True, True, True],
]

for feats in featureSetsToUse:
    # positives, negatives, featuresToUse, whereToPrint, verbose, bayes, tree, entropy
    if feats[1]:
        Classifiers.runClassifiers(positives, negatives, feats, "output.txt", False, True, False, False)
    else:
        Classifiers.runClassifiers(positives, negatives, feats, "output.txt", False, True, True, False)
