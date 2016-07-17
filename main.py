import Classifiers
import DataCreator
import random
import multiprocessing


def getData(usePickled = True, useSentences = True):
    positives = []
    negatives = []

    # determine if we are going to use pickled data or not
    if usePickled:
        print("Using pickled files\n")
        if useSentences:
            positives = DataCreator.usePickledFile("pickled_data/sent_positives.p")
            negatives = DataCreator.usePickledFile("pickled_data/sent_negatives.p")
        else:
            positives = DataCreator.usePickledFile("pickled_data/para_positives.p")
            negatives = DataCreator.usePickledFile("pickled_data/para_negatives.p")
    else:
        print("Creating data\n")
        (positives, negatives) = DataCreator.createDataFrom("parsed_websites/", "Ted_Meta.txt", "Ted_Laughs.txt", False, useSentences)
        print("Pickling data\n")

        if useSentences:
            DataCreator.pickleData("pickled_data/sent_positives.p", positives)
            DataCreator.pickleData("pickled_data/sent_negatives.p", negatives)
        else:
            DataCreator.pickleData("pickled_data/para_positives.p", positives)
            DataCreator.pickleData("pickled_data/para_negatives.p", negatives)

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
    (positives, negatives) = getData(True, False)

    print("Extracting Features\n")
    # [every word in the text,
    #  ngram for words and characters,
    #  POS tag Personal Pronouns and Proper Nouns per Noun,
    #  Noun+adjective+verb percentage,
    #  Sentiment Analysis,
    #  Laugh Count Before This,
    #  Sentences since last laugh,
    #  Depth,
    #  length,
    #  isQuestion,
    #  isQuote,
    #  word variance]
    featureSetsToUse = [
        [True, False, True, True, True, True, True, True, True, True, True, True],
    ]

    # [Naive Bayes, Decision Tree, Max Entropy, Support Vector machine, adaboost, random forest]
    classifiersToUse = [False, False, False, False, True, False]

    jobs = []

    # clear file
    wf = open("blah.txt", 'w')
    wf.writelines("  accuracy    pos precision    pos recall    pos f1    neg precision    neg recall    neg f1")
    wf.close

    maxlen = max(len(positives), len(negatives))


    for feats in featureSetsToUse:
        for j in range(5):
            for i in range(1):
                random.shuffle(positives)
                random.shuffle(negatives)
                positives = positives[:maxlen]
                negatives = negatives[:maxlen]
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
