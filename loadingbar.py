import time
import sys

toolbar_width = 40


def getAnimationChar():
    getAnimationChar.ctr = (getAnimationChar.ctr + 1) % 4
    if getAnimationChar.ctr == 0:
        return "\\"
    elif getAnimationChar.ctr == 1:
        return "|"
    elif getAnimationChar.ctr == 2:
        return "/"
    else:
        return "-"

getAnimationChar.ctr = 0


# little function to show percentage
def printPercentage(perc, intro_str="", useBar=True):
    if useBar:
        toWrite = intro_str + "|<"
        for i in range(int(perc)//4):
            toWrite = toWrite + "="

        toWrite = toWrite + getAnimationChar() + ">"

        for i in range(int(perc)//4 + 1, 100//4):
            toWrite = toWrite + " "

        toWrite = toWrite + "|\r"
        sys.stdout.write(toWrite)
        sys.stdout.flush()
    else:
        toWrite = intro_str + "%d%% \r"
        sys.stdout.write(toWrite % perc)
        sys.stdout.flush()
