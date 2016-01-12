import Classifiers
import DataCreator

#TODO Handle command line arguments. 1 for the amount of test files, another for if we are making a new meta file or not, and
#what classifier to use
(positives, negatives) = DataCreator.createDataFrom("parsed_websites/", "Ted_Meta.txt", "Ted_Laughs.txt", False)
print("Extracting Features\n")
Classifiers.runClassifiers(positives, negatives, False, True, True, False) #positives, negatives, verbose, bayes, tree, entropy
