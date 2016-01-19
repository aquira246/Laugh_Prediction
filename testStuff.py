import FeatureExtractor
import FeatureCollection

features = FeatureCollection.FeatureCollection("test")
features.sentences = ["Hello world, I am paul. This is why I'm paul."]
features.numSents = len(features.sentences)

results = FeatureExtractor.langFeatures(features, [False, True, False])

for x in results.keys():
    print(x, "\n")
