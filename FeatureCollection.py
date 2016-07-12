class FeatureCollection(object):
    """docstring for featureCollection"""

    def __init__(self, talkName="", talkAuthor=""):
        self.name = talkName
        self.author = talkAuthor
        self.length = 0
        self.depth = 0
        self.laughsUntilNow = 0
        self.sentsSinceLaugh = 0
        self.positive = False
        # self.namedEntities = []
        # self.namedEntityCount = 0
        self.POS = {}
        self.sentimentFeats = {}
        self.subjectivity = 0
        self.polarity = 0
        self.words = []
        self.sentence = ""
        self.prev3Words = []
        self.wordVariance = 0

    def infoToString(self):
        ret = "Name: " + self.name + " " \
            + "Author: " + self.author + " " \
            + "Length: " + str(self.length) + " " \
            + "Depth: " + str(self.depth) + " " \
            + "Laughs Until Now: " + str(self.laughsUntilNow) + " " \
            + "Since Last Laugh: " + str(self.sentsSinceLaugh) + " " \
            + "positive: " + str(self.positive) + " \n"

        return ret

    def stringWords(self):
        ret = " ".join(self.words)
        return ret

    def posToString(self):
        ret = "Personal Pronouns: " + str(self.POS["Personal_Pronoun_Percentage"]) + " "\
            + "noun_percentage: " + str(self.POS["noun_percentage"]) + " "\
            + "adj_percentage: " + str(self.POS["adj_percentage"]) + " "\
            + "verb_percentage: " + str(self.POS["verb_percentage"]) + "\n"

        return ret

    def sentimentToString(self):
        return "Polarity: " + str(self.polarity) + " " +\
                "Polarity diff: " + str(self.sentimentFeats["Polarity_Diff"]) + " " +\
                "  Subjectivity: " + str(self.subjectivity) + "\n"
