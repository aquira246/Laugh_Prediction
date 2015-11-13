import time
import sys

for i in range(100):
    time.sleep(1)
    toWrite = "\r" + "hi" + "%d%%"
    sys.stdout.write(toWrite % i)
    sys.stdout.flush()
