class FeatureCollection(object):
    """docstring for featureCollection"""

    name = ""
    author = ""
    length = 0
    depth = 0
    laughsUntilNow = 0
    sentsSinceLaugh = 0
    positive = False
    sentences = []

    def __init__(self, talkName="", talkAuthor=""):
        self.name = talkName
        self.author = talkAuthor
        self.length = 0
        self.depth = 0
        self.laughsUntilNow = 0
        self.sentsSinceLaugh = 0
        self.positive = False
        self.sentences = []

    def infoToString(self):
        ret = "Name: " + self.name + " " \
            + "Author: " + self.author + " " \
            + "Length: " + str(self.length) + " " \
            + "Depth: " + str(self.depth) + " " \
            + "Laughs Until Now: " + str(self.laughsUntilNow) + " " \
            + "Since Last Laugh: " + str(self.sentsSinceLaugh) + " " \
            + "positive: " + str(self.positive) + " \n"

        return ret
