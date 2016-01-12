from nltk import sent_tokenize
from nltk import word_tokenize
import FeatureCollection
import copy

def splitFile(filename):
    rf = open(filename, 'r')

    #skip first 6 lines since they aren't important
    talk = " ".join(rf.readlines()[6:])

    #remove the Audio: Laughing and the applause
    talk = talk.replace("(Applause)", "")
    talk = talk.replace("(Audio: Laughing)", "")

    sents = sent_tokenize(talk) # the talk turned into sentences
    passedSents = []            # all of the sentences passed
    numSents = len(sents)       # the number of sentences in the talk
    sentsSinceLastLaugh = 0     # the number of sentences since the last laugh
    laughCount = 0              # the laughs counted so far
    positives = []              # all of the positives
    negatives = []              # all of the negatives

    for i in range(numSents):
        # create a FeatureCollection for each sentence
        features = FeatureCollection.FeatureCollection("TODO_GETNAME")

        # if there is laughter in the sentence and it is not at the beginning OR
        # it is at the start of the next (if there is one) sentence
        if ("(Laughter)" in sents[i][3:]) or (i != numSents - 1 and ("(Laughter)" in sents[i+1][:11])):
            features.positive = True
        else:
            features.positive = False

        # remove the laughter from the sentence
        curSent = sents[i].replace("(Laughter)", "")

        #increase the distance since the last laugh
        sentsSinceLastLaugh += 1

        # put this sentence at the end of the passed sentences list
        passedSents.append(curSent) #TODO might need to change it to a deep copy and not just a reference to curSent

        features.length = len(word_tokenize(curSent))
        features.depth = i/numSents
        features.laughsUntilNow = laughCount
        features.sentsSinceLaugh = sentsSinceLastLaugh
        features.sentences = copy.copy(passedSents) # need a shallow copy so that we can modify passedSents

        # if the sentence was positive, we need to reset the distance since last laugh and increment laugh count
        if features.positive:
            laughCount += 1
            sentsSinceLastLaugh = 0
            positives.append(features)
            negatives.append(features)  # remove this. This is only for testing this
        else:
            negatives.append(features)

    rf.close

    return [positives, negatives]



[ps, ns] = splitFile("testfile.txt")
wf = open("checkanswer.txt", 'w')

# for feats in ps:
#     wf.writelines(feats.infoToString())
#     wf.writelines(feats.sentences[-1])
#     wf.writelines("\n\n")

for feats in ns:
    wf.writelines(feats.infoToString())
    wf.writelines(feats.sentences[-1])
    wf.writelines("\n\n")

wf.close

