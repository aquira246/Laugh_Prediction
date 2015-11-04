import TedMeta

mf = open("Ted_Meta.txt", "r")
lineCount = 0

for line in mf:
    lineCount += 1
    if lineCount > 2:
        TedMeta.getMetaDataFromString(line)

TedMeta.getMetaDataFromString("A cyber-magic card trick like no other     Marco Tempest    []    Word Count: 979    Laugh Count: 3    First Laugh At: 446")
TedMeta.getMetaDataFromString("5 ways to kill your dreams     Bel Pesce    ['Inspiring', 'Beautiful']    Word Count: 1185    Laugh Count: 4    First Laugh At: 60")
