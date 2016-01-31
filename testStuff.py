import FeatureCollection
import DataCreator
import Classifiers


# (ps, ns) = DataCreator.createDataFrom("parsed_websites/", "Ted_Meta_testing.txt", "Ted_Laughs.txt", False)
# DataCreator.pickleData("pickled_data/test_positives.p", ps)
# DataCreator.pickleData("pickled_data/test_negatives.p", ns)

positives = DataCreator.usePickledFile("pickled_data/test_positives.p")
negatives = DataCreator.usePickledFile("pickled_data/test_negatives.p")

print("Positives: ", len(positives), "\n")
print("Negatives: ", len(negatives), "\n")

featureSetsToUse = [True, False, False, False, False, False, False, False, False]
Classifiers.runClassifiers(positives, negatives, featureSetsToUse, "", False, [True, False, False, True])

pf = open("positives.txt", 'w')
nf = open("negatives.txt", 'w')

for p in positives:
    pf.write(p.infoToString())
    # pf.write(p.sentimentToString())
    pf.write(p.posToString())
    # pf.write(p.stringWords() + "\n\n")
    # pf.write(p.charGramsToString() + "\n\n")
    # pf.write(p.wordGramsToString() + "\n\n")
    pf.write("\n\n")

for n in negatives:
    nf.write(n.infoToString())
    # nf.write(n.sentimentToString())
    nf.write(n.posToString())
    # nf.write(n.stringWords() + "\n\n")
    # nf.write(n.charGramsToString() + "\n\n")
    # nf.write(n.wordGramsToString() + "\n\n")
    nf.write("\n\n")
