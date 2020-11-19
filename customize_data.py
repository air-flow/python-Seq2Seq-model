import numpy as np
import MeCab
from operator import itemgetter
import pprint
import random
# from analysis import CosSimilarity
# 教師データ
def ReadTestData():
    input_data = []
    output_data = []
    with open(".\\mine\\data\\input\\input.txt",mode='r',encoding="utf-8") as f:
        input_data = f.readlines()
    with open(".\\mine\\data\\output\\output.txt",mode='r',encoding="utf-8") as f:
        output_data = f.readlines()
    result = []
    max_value = 1000 # 抽出する行数指定値
    for i in range(max_value):
        # result.append([input_data[i].rsplit("\n"),output_data[i].rsplit("\n")])
        result.append([[input_data[i].replace( '\n' , '' )],[output_data[i].replace( '\n' , '' )]])
    return result

def ReadTestQuestionnaireData(number=2):
    input_data = []
    output_data = []
    with open(".\\mine\\Question\\input 入力\\inputText.txt",mode='r',encoding="utf-8") as f:
        input_data = f.readlines()
    with open(".\\mine\\Question\\output 出力\\output{}.txt".format(number),mode='r',encoding="utf-8") as f:
        output_data = f.readlines()
    result = []
    for i in range(len(input_data)):
        # result.append([input_data[i].rsplit("\n"),output_data[i].rsplit("\n")])
        temp_index = output_data[i].index(".")+1
        temp_str = output_data[i][temp_index:]
        result.append([[input_data[i].replace( '\n' , '' )],[temp_str.replace( '\n' , '' )]])
    return result

def SampleData():
    data = [
        [["初めまして。"], ["初めまして。よろしくお願いします。"]],
        [["どこから来たんですか？"], ["日本から来ました。"]],
        [["日本のどこに住んでるんですか？"], ["東京に住んでいます。"]],
        [["仕事は何してますか？"], ["私は会社員です。"]],
        [["お会いできて嬉しかったです。"], ["私もです！"]],
        [["おはよう。"], ["おはようございます。"]],
        [["いつも何時に起きますか？"], ["6時に起きます。"]],
        [["朝食は何を食べますか？"], ["たいていトーストと卵を食べます。"]],
        [["朝食は毎日食べますか？"], ["たまに朝食を抜くことがあります。"]],
        [["野菜をたくさん取っていますか？"], ["毎日野菜を取るようにしています。"]],
        [["週末は何をしていますか？"], ["友達と会っていることが多いです。"]],
        [["どこに行くのが好き？"], ["私たちは渋谷に行くのが好きです。"]]
    ]
    return data

# 文章をわかち書きした単語をランダムで選択し再構築する
# もとの学習文章からどれだけ単語が欠如しても対応できるのかのテスト用
def TextRandomLack(text):
    m = MeCab.Tagger()
    result = []
    for m in m.parse(text).split("\n"):
        temp = m.split("\t")[0].lower()
        if len(temp) != 0 and temp != "eos":
            result.append(temp)
    # print(text,result)
    if len(result) > 3:# 結果行が最低限の単語数に満たない場合を弾く
        data_size = len(result)
        sample_size = 3
        if data_size - 3 > 0:# 抽出する単語数を指定するときのマイナス判定
            sample_size = random.randint(3, data_size-1)
        shuffled_idx = np.random.choice(np.arange(data_size), sample_size, replace=False)
        shuffled_idx.sort()#ランダムリストは順序が保証されていないため昇順に変換
        result = "".join(list(itemgetter(*shuffled_idx)(result)))
    else:
        result = "".join(result)
    return result

# テスト用ランダム生成
def TestRandom():
    result = list(range(0,1,1))
    data_size = len(result)
    # range(random.randint(3, data_size)
    temp = []
    r = {}
    for i in result:
        if i > 2:
            r[i] = 0
    # print(r)
    for i in range(100):
        sample_size = random.randint(3, data_size-1)
        shuffled_idx = np.random.choice(np.arange(data_size), sample_size, replace=False)
        shuffled_idx.sort()
        # print(shuffled_idx)
        temp.append([len(shuffled_idx),shuffled_idx])
        r[len(shuffled_idx)]+=1


if __name__ == "__main__":
    pprint.pprint(SampleData())
    # pprint.pprint(ReadTestData())
    # print(len(ReadTestData()))
    # pprint.pprint(ReadTestQuestionnaireData(1))
    # print(len(ReadTestQuestionnaireData(1)))
    # pprint.pprint(temp)
    # for i in SampleData():
        # pprint.pprint(TextRandomLack(i[0][0]))