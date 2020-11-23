import numpy as np
import MeCab
from operator import itemgetter
import pprint
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from customize_data import SampleData, ReadTestQuestionnaireData,TextRandomLack,SetInputOutput,GetFile
from main import SpeechOneText,DataConverter

# def GetFile (path):
#     with open(path,mode='r',encoding="utf-8") as f:
#         input_data = f.readlines()
#     return input_data
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
def CosSimilarity(question,answer):
    """
    question: エンコード文
    answer:　デコード文
    """
    for i in sample:
        docs = [i[0][0],TextRandomLack(i[0][0])]
        cs_array = np.round(cosine_similarity(vecs_array(docs), vecs_array(docs)),3)
        print(docs,cs_array[0][1])

def CountWord():
    input_data = GetFile(".\\mine\\siritori\\input\\input.txt")
    output_data = GetFile(".\\mine\\siritori\\output\\output.txt")
    sum_text = list(map(lambda x: x.replace( '\n' , '' ),input_data)) + list(map(lambda x: x.replace( '\n' , '' ),output_data))
    data = SetInputOutput(input_data,output_data)
    BATCH_COL_SIZE = 15
    data_converter = DataConverter(batch_col_size=BATCH_COL_SIZE)
    data_converter.load(data)
    vocab_size = len(data_converter.vocab)
    items = list(data_converter.vocab.keys())
    temp = []
    for i in sum_text:
        if i not in items:
            temp.append(i)
    print(list(set(temp)))

if __name__ == "__main__":
    # CosSimilarity()
    data = SampleData()
