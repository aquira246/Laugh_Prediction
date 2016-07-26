import os.path
import logging

import gensim
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.corpora import WikiCorpus

DEFAULT_DICT_SIZE = 100000


def LoadModel(MakeNew=False, useWiki=False):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    typeOfModel = "wiki" if useWiki else "text8"

    if not MakeNew:
        if os.path.isfile("word2vec/" + typeOfModel + ".model"):
            print("Using " + typeOfModel + ".model file")
            model = Word2Vec.load("word2vec/" + typeOfModel + ".model")
            return model

        if os.path.isfile("word2vec/" + typeOfModel + ".bin"):
            print("Using " + typeOfModel + ".bin file")
            model = Word2Vec.load_word2vec_format('word2vec/" + typeOfModel + ".bin', binary=True)  # C binary format
            return model

    print("Generating new model. This may take some time")
    sentences = gensim.models.word2vec.Text8Corpus('word2vec/text8')
    model = Word2Vec(sentences, size=200, workers=4)
    print("Saving model as text8.model")
    model.save('word2vec/text8.model')
    return model

def Similarity(model, a, b):
    return model.similarity(a, b)


"""MAIN"""
if __name__ == '__main__':
    model = LoadModel(MakeNew=True, useWiki=True)
    print(model.most_similar(positive=['woman', 'king'], negative=['man']))
    print()
    print(model.similarity('woman', 'man'))
    print()
    print(model.doesnt_match("breakfast cereal dinner lunch".split()))
