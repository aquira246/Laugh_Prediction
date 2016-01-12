import Classifiers
import DataCreator

#TODO Handle command line arguments. 1 for the amount of test files, another for if we are making a new meta file or not, and
#what classifier to use
(positives, negatives) = DataCreator.createDataFrom("parsed_websites/", "Ted_Meta_testing.txt", "Ted_Laughs.txt", False)

pf = open("positives.txt", "w")
nf = open("negatives.txt", "w")

for feats in positives:
    pf.writelines(feats.infoToString())
    pf.writelines(feats.sentences[-1])
    pf.writelines("\n\n")

for feats in negatives:
    nf.writelines(feats.infoToString())
    nf.writelines(feats.sentences[-1])
    nf.writelines("\n\n")

pf.close
nf.close
