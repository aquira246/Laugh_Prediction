import Classifiers
import DataCreator
import random
import multiprocessing

from tabulate import tabulate


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


def main():
    (positives, negatives) = getData(True, useSentences=True)

    print("Extracting Features\n")
    featureSetsToUse = {}
    featureSetsToUse["words"] = True          # every word in the text
    featureSetsToUse["ngrams"] = True         # ngram for words and characters
    featureSetsToUse["pos_nouns"] = False      # POS tag Personal Pronouns and Proper Nouns per Noun
    featureSetsToUse["pos_perc"] = True       # Noun+adjective+verb percentage
    featureSetsToUse["sentiment"] = True      # Sentiment Analysis
    featureSetsToUse["laugh_count"] = False    # Laugh Count Before This
    featureSetsToUse["last_laugh"] = False     # Chunks since last laugh
    featureSetsToUse["depth"] = True          # Depth
    featureSetsToUse["length"] = True         # length
    featureSetsToUse["question"] = True       # there is a question mark
    featureSetsToUse["exclamation"] = True    # there is a exclamation mark
    featureSetsToUse["quote"] = True          # isQuote
    featureSetsToUse["variance"] = True       # word variance
    featureSetsToUse["incongruity"] = True    # incongruity
    featureSetsToUse["swearing"] = False       # swearing
    featureSetsToUse["Dim Reduction"] = False

    classifiersToUse = [True,  # Naive Bayes
                        False,  # Decision Tree
                        False,  # Max Entropy
                        False,  # Support Vector machine
                        False,  # adaboost
                        False,  # random forest
                        False]  # SGD? NEVER USE WITH NGRAMS! Crashes machine

    jobs = []

    # clear file
    wf = open("blah.txt", 'w')
    wf.writelines("  accuracy    pos precision    pos recall    pos f1    neg precision    neg recall    neg f1")
    wf.close

    dataCut = min(len(positives), len(negatives))


    for j in range(9):
        for i in range(3):
            random.shuffle(positives)
            random.shuffle(negatives)
            positives = positives[:dataCut]
            negatives = negatives[:dataCut]
            p = multiprocessing.Process(target=worker, \
                args=(positives, negatives, classifiersToUse, featureSetsToUse, "blah.txt", j*5 + i,))
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()

    rf = open("blah.txt", 'r')
    out = open("output.txt", 'a')

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

    if i > 0:
        accuracy = accuracy/i
        posPrecision = posPrecision/i
        posRecall = posRecall/i
        posf1 = posf1/i
        negPrecision = negPrecision/i
        negRecall = negRecall/i
        negf1 = negf1/i

        table = [["Fill in", accuracy, posPrecision, posRecall, posf1, negPrecision, negRecall, negf1]]
        headers=["Classifier", "accuracy", "pos precision", "pos recall", "pos f1", "neg precision", "neg recall", "neg f1"]
        out.write(tabulate(table, headers))
    else:
        print("i is 0")

    rf.close
    out.close

"""MAIN"""
if __name__ == '__main__':
    main()
