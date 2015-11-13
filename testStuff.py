import time
import sys


for k in range(20):
    i = k*5
    time.sleep(1)
    # toWrite = "\r" + "hi" + "%d%%"
    # sys.stdout.write(toWrite % i)
    # sys.stdout.flush()
    toWrite = "\r|<"

    for j in range(i//5):
        toWrite = toWrite + "="

    toWrite = toWrite + ">"

    for j in range(i//5 + 2, 100//5):
        toWrite = toWrite + " "

    toWrite = toWrite + "|"
    sys.stdout.write(toWrite)
    sys.stdout.flush()
