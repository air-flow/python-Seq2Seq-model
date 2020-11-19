import numpy as np
import MeCab
from operator import itemgetter
import pprint
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from customize_data import SampleData, ReadTestQuestionnaireData,TextRandomLack
# #わかち書き関数
def wakachi(text):
    m = MeCab.Tagger("-Owakati")
    return m.parse(text).split()

#文書ベクトル化関数
def vecs_array(documents):
    docs = np.array(documents)
    vectorizer = TfidfVectorizer(analyzer=wakachi,binary=True,use_idf=False)
    vecs = vectorizer.fit_transform(docs)
    return vecs.toarray()

# Cos類似度判定
def CosSimilarity():
    sample = SampleData()
    for i in sample:
        docs = [i[0][0],TextRandomLack(i[0][0])]
        cs_array = np.round(cosine_similarity(vecs_array(docs), vecs_array(docs)),3)
        print(docs,cs_array[0][1])

if __name__ == "__main__":
    CosSimilarity()