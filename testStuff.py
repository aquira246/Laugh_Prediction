def text2int (textnum, numwords={}):
    if not numwords:
        units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        numwords["and"] = (1, 0)
        for idx, word in enumerate(units):  numwords[word] = (1, idx)
        for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)

    ordinal_words = {'first':1, 'second':2, 'third':3, 'fifth':5, 'eighth':8, 'ninth':9, 'twelfth':12}
    ordinal_endings = [('ieth', 'y'), ('th', '')]

    textnum = textnum.replace('-', ' ')

    current = result = 0
    curstring = ""
    onnumber = False
    for word in textnum.split():
        if word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
            onnumber = True
        else:
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = "%s%s" % (word[:-len(ending)], replacement)

            if word not in numwords:
                if onnumber:
                    curstring += repr(result + current) + " "
                curstring += word + " "
                result = current = 0
                onnumber = False
            else:
                scale, increment = numwords[word]

                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True

    if onnumber:
        curstring += repr(result + current)

    return curstring

print(text2int("ten I want fifty five hot dogs for two hundred dollars."))
print(text2int("Does 300 cheese fifteen tacos seven thousand. nineteen thousand"))
print(text2int("Lol, nope"))


# import nltk

# from nltk.corpus import stopwords
# from nltk.corpus import brown
# from nltk.stem.snowball import SnowballStemmer
# import pickle

# stop = stopwords.words('english')
# stemmer = SnowballStemmer("english")

# import FeatureExtractor
# import DataCreator


# def getData(usePickled=False, useSentences = True):
#     positives = []
#     negatives = []

#     # determine if we are going to use pickled data or not
#     if usePickled:
#         print("Using pickled files\n")
#         positives = DataCreator.usePickledFile("pickled_data/test_positives.p")
#         negatives = DataCreator.usePickledFile("pickled_data/test_negatives.p")
#     else:
#         print("Creating data\n")
#         (positives, negatives)=DataCreator.createDataFrom("parsed_websites/", "Ted_Meta_testing.txt", "Ted_Laughs.txt", False, useSentences)
#         print("Pickling data\n")
#         DataCreator.pickleData("pickled_data/test_positives.p", positives)
#         DataCreator.pickleData("pickled_data/test_negatives.p", negatives)

#     return(positives, negatives)



# """MAIN"""
# if __name__ == '__main__':
#     (positives, negatives) = getData(False, False)

#     featureSetsToUse = {}
#     featureSetsToUse["words"] = False          # every word in the text
#     featureSetsToUse["ngrams"] = False         # ngram for words and characters
#     featureSetsToUse["pos_nouns"] = False      # POS tag Personal Pronouns and Proper Nouns per Noun
#     featureSetsToUse["pos_perc"] = False       # Noun+adjective+verb percentage
#     featureSetsToUse["sentiment"] = False      # Sentiment Analysis
#     featureSetsToUse["laugh_count"] = False    # Laugh Count Before This
#     featureSetsToUse["last_laugh"] = False     # Chunks since last laugh
#     featureSetsToUse["depth"] = False          # Depth
#     featureSetsToUse["length"] = False         # length
#     featureSetsToUse["question"] = False       # there is a question mark
#     featureSetsToUse["quote"] = False          # isQuote
#     featureSetsToUse["variance"] = False       # word variance
#     featureSetsToUse["incongruity"] = True    # incongruity
#     featureSetsToUse["swearing"] = True       # complexity


#     wf = open("deleteme.txt", 'w')

#     wf.writelines("Sentence   |||   proper noun percentage, word variance, prp, wordCount\n")
#     wf.writelines("========================================================================\n\n")

#     for data in positives:
#         wf.writelines(data.chunk + "\n")
#         # wf.writelines(str(data.features))

#         # wf.writelines("\n")
#         feats = FeatureExtractor.langFeatures(data, featureSetsToUse)

#         for i in feats.keys():
#             wf.writelines(i + " ")
#             wf.writelines(str(feats[i]))
#             wf.writelines("   ")

#         wf.writelines("\n")

#     wf.writelines("========================================================================\n")

#     for data in negatives:
#         wf.writelines(data.chunk + "\n")
#         # wf.writelines(str(data.features))
#         # wf.writelines("\n")
#         feats = FeatureExtractor.langFeatures(data, featureSetsToUse)

#         for i in feats.keys():
#             wf.writelines(i + " ")
#             wf.writelines(str(feats[i]))
#             wf.writelines("   ")

#         wf.writelines("\n")

#     wf.close
