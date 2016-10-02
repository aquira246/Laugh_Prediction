import sys
import nltk, re, pprint
import os
from nltk import word_tokenize
from urllib import request


def parseTedTalkTranscript(readFileName, writePath):
    rf = open(readFileName, 'r')
    ws = ""
    title = ""
    ratings = ["Informative", "Inspiring", "Courageous", "Jaw-dropping",
        "Longwinded", "Unconvincing", "Ingenious", "Persuasive", "Beautiful",
        "Obnoxious", "Fascinating", "OK", "Funny", "Confusing"]

    validScript = True

    for line in rf:
        keepTesting = True

        # get the title and author
        if keepTesting:
            for toWrite in re.findall(r'<title>(.+)</title>', line):
                if "Page Not Found" in toWrite:
                    validScript = False

                for titleSegment in toWrite.split("|"):
                    titleSegment = titleSegment.replace("&#39;", "\'")
                    titleSegment = titleSegment.replace("&quot;", "\"")
                    titleSegment = titleSegment.replace("&amp;", "&")
                    if "TED" not in titleSegment:
                        authorAndTitle = titleSegment.split(": ")
                        if len(authorAndTitle) > 1:
                            title = authorAndTitle[1]
                            title = title.replace(" ", "_")
                            ws += "Title: " + authorAndTitle[1] + "\n"
                            ws += "Author: " + authorAndTitle[0] + "\n\n"
                            keepTesting = False

        # find all dialogue that's on a line that says "<span class..."
        if keepTesting:
            for toWrite in re.findall(r'<span class=\'talk-transcript__fragment\' .+>(.+)', line):
                toWrite = toWrite.replace("</span>", "")
                toWrite = toWrite.replace("&quot;", "\"")
                ws += toWrite + " "
                keepTesting = False

        # find all dialogue that gets cut off
        if keepTesting:
            for toWrite in re.findall(r'^([^<].+)</span>$', line):
                toWrite = toWrite.replace("&quot;", "\"")
                ws += toWrite + " "
                keepTesting = False

        # find the paragraph endings
        if keepTesting:
            for toWrite in re.findall(r"^<p class='talk-transcript__para'>$", line):
                ws += "\n"
                keepTesting = False
        if keepTesting:
            for toWrite in re.findall(r"^<p class='talk-transcript__para__text'>$", line):
                ws += "\n"
                keepTesting = False

        # mark the tags
        if keepTesting:
            for toWrite in re.findall(r'^(\w+, \w+)$', line):
                tags = toWrite.split(", ")
                if len(tags) > 1:
                    if tags[0] in ratings and tags[1] in ratings:
                        ws += "Tags: " + tags[0] + ", " + tags[1] + "\n\n"
                        keepTesting = False

    rf.close
    if validScript:
        writeFileName = "websites/parsedFile.txt"
        wf = open(writeFileName, 'w')
        wf.writelines(ws)
        wf.close


# can't seem to grab the videos using the sort :(
def grabSearchPages(rangeStart, rangeEnd):
    wf = open("search_pages/pageGrabber.sh", 'w')
    wf.writelines("#!/bin/bash\n\n")

    for num in range(rangeStart, rangeEnd):
        wf.writelines("wget https://www.ted.com/talks?page=" + str(num) + "&sort=newest\n")

    wf.close


def grabTalksFromSearchPages(searchPath, writePath):
    wf = open(writePath, 'w')
    wf.writelines("#!/bin/bash\n\n")

    # previous needed to get rid of duplicates that are next to each other
    previous = ""

    for filename in os.listdir(searchPath):
        if ".sh" not in filename:
            print(filename)
            rf = open(searchPath + filename, 'r')

            for line in rf:
                for toWrite in re.findall(r'^<a class=\'\' href=\'/(talks/.+)\'>$', line):
                    if previous != toWrite:
                        previous = toWrite
                        wf.writelines("wget "+ "https://www.ted.com/" + toWrite + "/transcript?language=en\n")

            rf.close

    wf.close


# main
fileToParse = "websites/transcript?language=en"
writePathForParsedFiles = "parsed_websites/"

grabSearchPages(40, 63)
# run the bash script in the search_pages folder
grabTalksFromSearchPages("search_pages/", "websites/TedTalkGrabber2.sh")
# run the bash script

for filename in os.listdir("websites/"):
    if ".sh" not in filename:
        fileToParse = "websites/" + filename
        parseTedTalkTranscript(fileToParse, writePathForParsedFiles)
