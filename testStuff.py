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
