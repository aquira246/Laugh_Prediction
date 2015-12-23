import sys
import nltk, re, pprint
import math, random, statistics
from nltk import word_tokenize
import TedMeta

def trimFile(metadata):
    rf = open(metadata.filename, 'r')

    prevPara = ""
    prevIsPositive = False
    positives = []
    negatives = []


    for paragraph in rf:
        if re.match(r'^Title: .+', paragraph) is not None:
            pass
        elif re.match(r'^Author: .+', paragraph) is not None:
            pass
        elif re.match(r'^Tags: .+', paragraph) is not None:
            pass
        else:
            paragraph = paragraph.replace("(Applause)", "")
            paragraph = paragraph.replace("(Audio: Laughing)", "")

            if prevIsPositive or ("(Laughter)" in paragraph[:11] and prevPara != ""):
                positives.append(prevPara)
            else:
                if len(prevPara) > 2:
                    negatives.append(prevPara)

            prevIsPositive = False

            if "(Laughter)" in paragraph[10:]:
                prevIsPositive = True

            prevPara = paragraph.replace("(Laughter)", "")

    if prevIsPositive:
        positives.append(prevPara)
    else:
        if len(prevPara) > 2:
            negatives.append(prevPara)

    rf.close
    return [positives, negatives]

#main
ret = []

pf = open("positives.txt", 'w')
nf = open("negatives.txt", 'w')

meta = TedMeta.tedMetaData("parsed_websites/Aliens,_love_--_where_are_they?_.txt", "Aliens, love -- where are they? ")
ret = trimFile(meta)

pf.writelines("---------------------" + meta.name + "---------------------\n")
nf.writelines("---------------------" + meta.name + "---------------------\n")
for para in ret[0]:
    pf.writelines(para + "\n")
for para in ret[1]:
    nf.writelines(para + "\n")

meta = TedMeta.tedMetaData("parsed_websites/Why_we_laugh_.txt", "Why we laugh ")
ret = trimFile(meta)

pf.writelines("---------------------" + meta.name + "---------------------\n")
nf.writelines("---------------------" + meta.name + "---------------------\n")
for para in ret[0]:
    pf.writelines(para + "\n")
for para in ret[1]:
    nf.writelines(para + "\n")

meta = TedMeta.tedMetaData("parsed_websites/Comedy_is_translation_.txt", "Comedy is translation ")
ret = trimFile(meta)

pf.writelines("---------------------" + meta.name + "---------------------\n")
nf.writelines("---------------------" + meta.name + "---------------------\n")
for para in ret[0]:
    pf.writelines(para + "\n")
for para in ret[1]:
    nf.writelines(para + "\n")
