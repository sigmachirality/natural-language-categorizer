import gensim
import numpy

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
vector = model['internship']
print(vector)