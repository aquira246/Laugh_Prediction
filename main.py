import Classifiers
import DataCreator
import random
import multiprocessing


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


def worker(positives, negatives, classifiersToUse, feats, outFile, i):
    """thread worker function"""
    # positives, negatives, featuresToUse, whereToPrint, verbose, classifiersToUse
    results = Classifiers.runClassifiers(positives, negatives, feats, "output.txt", False, classifiersToUse)
    print("done ", i)

    for r in results:
        s = str(r[1]) + "   ", str(r[2]) + "   ",\
            str(r[3]) + "   ", str(r[4]) + "   ",\
            str(r[5]) + "   ", str(r[6]) + "   ",\
            str(r[7]) + "\n"
        wf = open(outFile, 'a')
        wf.writelines(s)
        wf.close()

    return


"""MAIN"""
if __name__ == '__main__':
    (positives, negatives) = getData(True)

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
    # featureSetsToUse = [
    #     # [True, False, False, False, False, False, False, False, False],
    #     # [True, True, False, False, False, False, False, False, False],
    #     # [False, False, True, False, False, False, False, False, False],
    #     # [False, False, False, True, False, False, False, False, False],
    #     # [False, False, False, False, True, False, False, False, False],
    #     # [False, False, False, False, False, True, False, False, False],
    #     # [False, False, False, False, False, False, True, False, False],
    #     # [False, False, False, False, False, False, False, True, False],
    #     # [False, False, False, False, False, False, False, False, True],
    #     # [True, False, True, True, False, True, True, True, True],
    #     [True, True, True, True, True, True, True, True, True],
    #     # [True, False, True, True, True, True, True, True, True],
    #     # [True, False, True, True, True, True, True, False, False],
    #     # [False, False, True, True, True, True, True, True, False],
    #     # [True, False, True, True, True, False, False, False, True],
    #     # [False, False, False, False, False, True, True, True, False],
    #     # [True, True, True, True, True, False, False, False, True],
    # ]

    featureSetsToUse = [
        [True, True, True, True, True, False, False, True, True]
    ]

    # [Naive Bayes, Decision Tree, Max Entropy, Support Vector machine, adaboost, random forest]
    classifiersToUse = [False, False, False, False, True, False]

    jobs = []

    # clear file
    wf = open("blah.txt", 'w')
    wf.writelines("  accuracy    pos precision    pos recall    pos f1    neg precision    neg recall    neg f1")
    wf.close

    for feats in featureSetsToUse:
        for j in range(5):
            for i in range(5):
                random.shuffle(positives)
                random.shuffle(negatives)
                positives = positives[:3580]
                negatives = negatives[:3580]
                p = multiprocessing.Process(target=worker, args=(positives, negatives, classifiersToUse, feats, "blah.txt", j*5 + i,))
                jobs.append(p)
                p.start()

            for p in jobs:
                p.join()

rf = open("blah.txt", 'r')

accuracy = 0
posPrecision = 0
posRecall = 0
posf1 = 0
negPrecision = 0
negRecall = 0
negf1 = 0
i = 0
for line in rf.readlines()[1:]:
    info = line.strip().split("   ")
    i += 1
    accuracy += float(info[0])
    posPrecision += float(info[1])
    posRecall += float(info[2])
    posf1 += float(info[3])
    negPrecision += float(info[4])
    negRecall += float(info[5])
    negf1 += float(info[6])

accuracy = accuracy/i
posPrecision = posPrecision/i
posRecall = posRecall/i
posf1 = posf1/i
negPrecision = negPrecision/i
negRecall = negRecall/i
negf1 = negf1/i

print("Accuracy: ", accuracy)
print("Positive Precision: ", posPrecision)
print("Positive Recall: ", posRecall)
print("Positive f1: ", posf1)
print("Negative Precision: ", negPrecision)
print("Negative Recall: ", negRecall)
print("Negative f1: ", negf1)

rf.close
