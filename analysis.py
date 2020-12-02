import numpy as np
import MeCab
from operator import itemgetter
import pprint
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from customize_data import SampleData, ReadTestQuestionnaireData,TextRandomLack,SetInputOutput,GetFile
import collections
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
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
def CosSimilarity(data):
    result = []
    for i in data:
        docs = [i[0][0],TextRandomLack(i[0][0])]
        cs_array = np.round(cosine_similarity(vecs_array(docs), vecs_array(docs)),3)
        # print(docs,cs_array[0][1])
        result.append([docs,cs_array[0][1]])
    return result

# main → analysis import 重複するのでコメントアウト
# def CountWord():
#     input_data = GetFile(".\\mine\\siritori\\input\\input.txt")
#     output_data = GetFile(".\\mine\\siritori\\output\\output.txt")
#     sum_text = list(map(lambda x: x.replace( '\n' , '' ),input_data)) + list(map(lambda x: x.replace( '\n' , '' ),output_data))
#     data = SetInputOutput(input_data,output_data)
#     BATCH_COL_SIZE = 15
#     data_converter = DataConverter(batch_col_size=BATCH_COL_SIZE)
#     data_converter.load(data)
#     vocab_size = len(data_converter.vocab)
#     items = list(data_converter.vocab.keys())
#     temp = []
#     for i in sum_text:
#         if i not in items:
#             temp.append(i)
#     print(list(set(temp)))

def PercentageOfCorrectAnswers(text_answer,ai_answer):
    a =np.array(text_answer)
    b =np.array(ai_answer)
    bool_count = dict(collections.Counter(a==b))
    temp = bool_count[True]/(bool_count[True]+bool_count[False]) * 100
    print("正答率 : "+str(Decimal(str(temp)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))+"%")

def funcname(parameter_list):
    """
学習ファイル名
Personality
保存ファイル
Personality.sample1.network
学習状態
単語数: 1061
time :  0:00:00.073748
time :  0:00:00.095691
Timing :  単語ID変換  time : 0:00:00.046915
Timing :  読み込み  time : 0:00:00.074520
epoch:  10      total loss:     1760.929386138916       time:   0:13:46.749892
epoch:  20      total loss:     426.48425364494324      time:   0:13:42.134865
epoch:  30      total loss:     38.94310259819031       time:   0:13:43.389317
epoch:  40      total loss:     10.703677378594875      time:   0:13:45.170566
epoch:  50      total loss:     5.275798492133617       time:   0:13:57.714911
学習データチェック
教師データ
Timing :  単語ID変換  time : 0:00:02.478010
Timing :  読み込み  time : 0:00:02.970723
正答率 : 97%
Timing :  END  time : 0:01:30.225784
欠如データ
Timing :  単語ID変換  time : 0:00:02.478010
Timing :  読み込み  time : 0:00:02.970723
正答率 : 26%
Timing :  END  time : 0:01:30.225784
    """
    pass

if __name__ == "__main__":
    # data = SampleData()
    # input_data = GetFile(".\\mine\\base_sample\\input\\input.txt")
    # output_data = GetFile(".\\mine\\base_sample\\output\\output.txt")
    # data = SetInputOutput(input_data,output_data)
    # # pprint.pprint(data)
    # CosSimilarity(data)
    pass