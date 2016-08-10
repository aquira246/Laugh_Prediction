import Classifiers
import DataCreator
import random
import multiprocessing
from multiprocessing import Manager

from tabulate import tabulate

NUM_ITERATIONS = 25
NUM_COPROCESSES = 1

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


def worker(positives, negatives, classifiersToUse, feats, outFile, i, return_dict):
    """thread worker function"""
    # positives, negatives, featuresToUse, whereToPrint, verbose, classifiersToUse
    results = Classifiers.runClassifiers(positives, negatives, feats, "output.txt", False, classifiersToUse)
    print("done ", i)

    for r in results:
        return_dict["accuracy"] += r[1]
        return_dict["pos_precision"] += r[2]
        return_dict["pos_recall"] += r[3]
        return_dict["pos_f1"] += r[4]
        return_dict["neg_precision"] += r[5]
        return_dict["neg_recall"] += r[6]
        return_dict["neg_f1"] += r[7]

    return


def main():
    (positives, negatives) = getData(usePickled=True, useSentences=False)

    print("Extracting Features\n")
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
    featureSetsToUse["exclamation"] = True    # there is a exclamation mark
    featureSetsToUse["quote"] = True          # isQuote
    featureSetsToUse["variance"] = True       # word variance
    featureSetsToUse["incongruity"] = True    # incongruity
    featureSetsToUse["swearing"] = True       # swearing
    featureSetsToUse["statistics"] = True     # counting statistics
    featureSetsToUse["frequency"] = True      # frequency statistics
    featureSetsToUse["hapax"] = True          # hapax count

    featureSetsToUse["complexity"] = False     # complexity
    featureSetsToUse["max_ent"] = False        # max ent support
    featureSetsToUse["Dim Reduction"] = False

    classifiersToUse = [True,  # Naive Bayes
                        False,  # Decision Tree
                        False,  # Max Entropy
                        True,  # Support Vector machine
                        True,  # adaboost
                        False,  # random forest
                        False]  # COMBO

    jobs = []
    dataCut = min(len(positives), len(negatives))

    n = len(classifiersToUse)
    for clf in range(n):
        if classifiersToUse[clf]:
            clfrs = [False] * n
            clfrs[clf] = classifiersToUse[clf]

            manager = Manager()
            return_dict = manager.dict()
            return_dict["accuracy"] = 0
            return_dict["pos_precision"] = 0
            return_dict["pos_recall"] = 0
            return_dict["pos_f1"] = 0
            return_dict["neg_precision"] = 0
            return_dict["neg_recall"] = 0
            return_dict["neg_f1"] = 0

            for j in range(NUM_ITERATIONS):
                for i in range(NUM_COPROCESSES):
                    random.shuffle(positives)
                    random.shuffle(negatives)
                    positives = positives[:dataCut]
                    negatives = negatives[:dataCut]
                    p = multiprocessing.Process(target=worker,
                        args=(positives, negatives, clfrs, featureSetsToUse,
                        "blah.txt", j*NUM_COPROCESSES + i, return_dict))
                    jobs.append(p)
                    p.start()

                for p in jobs:
                    p.join()

            testing_length = NUM_ITERATIONS*NUM_COPROCESSES
            if testing_length > 0:
                accuracy = return_dict["accuracy"] / (testing_length)
                posPrecision = return_dict["pos_precision"] / (testing_length)
                posRecall = return_dict["pos_recall"] / (testing_length)
                posf1 = return_dict["pos_f1"] / (testing_length)
                negPrecision = return_dict["neg_precision"] / (testing_length)
                negRecall = return_dict["neg_recall"] / (testing_length)
                negf1 = return_dict["neg_f1"] / (testing_length)

                table = [["Fill in", accuracy, posPrecision, posRecall, posf1, negPrecision, negRecall, negf1]]
                headers=["Classifier", "accuracy", "pos precision", "pos recall", "pos f1", "neg precision", "neg recall", "neg f1"]
                out = open("output.txt", 'a')
                out.write(tabulate(table, headers))
                out.close
            else:
                print("i is 0")


"""MAIN"""
if __name__ == '__main__':
    main()
