import datetime
import numpy as np
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import MeCab
import pprint
# pip install numpy scipy scikit-learn chainer
# GPUのセット
FLAG_GPU = False # GPUを使用するかどうか
if FLAG_GPU: # numpyかcuda.cupyか
    xp = cuda.cupy
    cuda.get_device(0).use()
else:
    xp = np

# データ変換クラスの定義
class DataConverter:
    def __init__(self, batch_col_size):
        # クラスの初期化
        # :param batch_col_size: 学習時のミニバッチ単語数サイズ
        self.mecab = MeCab.Tagger() # 形態素解析器
        self.vocab = {"<eos>":0, "<unknown>": 1} # 単語辞書
        self.batch_col_size = batch_col_size

    def load(self, data):
        # 学習時に、教師データを読み込んでミニバッチサイズに対応したNumpy配列に変換する
        # :param data: 対話データ
        # 単語辞書の登録
        self.vocab = {"<eos>":0, "<unknown>": 1} # 単語辞書を初期化
        for d in data:
            sentences = [d[0][0], d[1][0]] # 入力文、返答文
            for sentence in sentences:
                sentence_words = self.sentence2words(sentence) # 文章を単語に分解する
                for word in sentence_words:
                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab)
        # 教師データのID化と整理
        queries, responses = [], []
        for d in data:
            query, response = d[0][0], d[1][0] #  エンコード文、デコード文
            queries.append(self.sentence2ids(sentence=query, train=True, sentence_type="query"))
            responses.append(self.sentence2ids(sentence=response, train=True, sentence_type="response"))
        self.train_queries = xp.vstack(queries)
        self.train_responses = xp.vstack(responses)

    def sentence2words(self, sentence):
        # 文章を単語の配列にして返却する
        # :param sentence: 文章文字列
        sentence_words = []
        for m in self.mecab.parse(sentence).split("\n"): # 形態素解析で単語に分解する
            w = m.split("\t")[0].lower() # 単語
            if len(w) == 0 or w == "eos": # 不正文字、EOSは省略
                continue
            sentence_words.append(w)
        sentence_words.append("<eos>") # 最後にvocabに登録している<eos>を代入する
        return sentence_words

    def sentence2ids(self, sentence, train=True, sentence_type="query"):
        # 文章を単語IDのNumpy配列に変換して返却する
        # :param sentence: 文章文字列
        # :param train: 学習用かどうか
        # :sentence_type: 学習用でミニバッチ対応のためのサイズ補填方向をクエリー・レスポンスで変更するため"query"or"response"を指定　
        # :return: 単語IDのNumpy配列
        ids = [] # 単語IDに変換して格納する配列
        sentence_words = self.sentence2words(sentence) # 文章を単語に分解する
        for word in sentence_words:
            if word in self.vocab: # 単語辞書に存在する単語ならば、IDに変換する
                ids.append(self.vocab[word])
            else: # 単語辞書に存在しない単語ならば、<unknown>に変換する
                ids.append(self.vocab["<unknown>"])
        # 学習時は、ミニバッチ対応のため、単語数サイズを調整してNumpy変換する
        if train:
            if sentence_type == "query": # クエリーの場合は前方にミニバッチ単語数サイズになるまで-1を補填する
                while len(ids) > self.batch_col_size: # ミニバッチ単語サイズよりも大きければ、ミニバッチ単語サイズになるまで先頭から削る
                    ids.pop(0)
                ids = xp.array([-1]*(self.batch_col_size-len(ids))+ids, dtype="int32")
            elif sentence_type == "response": # レスポンスの場合は後方にミニバッチ単語数サイズになるまで-1を補填する
                while len(ids) > self.batch_col_size: # ミニバッチ単語サイズよりも大きければ、ミニバッチ単語サイズになるまで末尾から削る
                    ids.pop()
                ids = xp.array(ids+[-1]*(self.batch_col_size-len(ids)), dtype="int32")
        else: # 予測時は、そのままNumpy変換する
            ids = xp.array([ids], dtype="int32")
        return ids

    def ids2words(self, ids):
        # 予測時に、単語IDのNumpy配列を単語に変換して返却する
        # :param ids: 単語IDのNumpy配列
        # :return: 単語の配列
        words = [] # 単語を格納する配列
        for i in ids: # 順番に単語IDを単語辞書から参照して単語に変換する
            words.append(list(self.vocab.keys())[list(self.vocab.values()).index(i)])
        return words

# モデルクラスの定義

# LSTMエンコーダークラス
class LSTMEncoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        # Encoderのインスタンス化
        # :param vocab_size: 使われる単語の種類数
        # :param embed_size: 単語をベクトル表現した際のサイズ
        # :param hidden_size: 隠れ層のサイズ
        super(LSTMEncoder, self).__init__(
            xe = L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh = L.Linear(embed_size, 4 * hidden_size),
            hh = L.Linear(hidden_size, 4 * hidden_size)
        )

    def __call__(self, x, c, h):
        # Encoderの計算
        # :param x: one-hotな単語
        # :param c: 内部メモリ
        # :param h: 隠れ層
        # :return: 次の内部メモリ、次の隠れ層
        e = F.tanh(self.xe(x))
        return F.lstm(c, self.eh(e) + self.hh(h))

# Attention Model + LSTMデコーダークラス
class AttLSTMDecoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        # Attention ModelのためのDecoderのインスタンス化
        # :param vocab_size: 語彙数
        # :param embed_size: 単語ベクトルのサイズ
        # :param hidden_size: 隠れ層のサイズ
        super(AttLSTMDecoder, self).__init__(
            ye = L.EmbedID(vocab_size, embed_size, ignore_label=-1), # 単語を単語ベクトルに変換する層
            eh = L.Linear(embed_size, 4 * hidden_size), # 単語ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
            hh = L.Linear(hidden_size, 4 * hidden_size), # Decoderの中間ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
            fh = L.Linear(hidden_size, 4 * hidden_size), # 順向きEncoderの中間ベクトルの加重平均を隠れ層の4倍のサイズのベクトルに変換する層
            bh = L.Linear(hidden_size, 4 * hidden_size), # 順向きEncoderの中間ベクトルの加重平均を隠れ層の4倍のサイズのベクトルに変換する層
            he = L.Linear(hidden_size, embed_size), # 隠れ層サイズのベクトルを単語ベクトルのサイズに変換する層
            ey = L.Linear(embed_size, vocab_size) # 単語ベクトルを語彙数サイズのベクトルに変換する層
        )

    def __call__(self, y, c, h, f, b):
        # Decoderの計算
        # :param y: Decoderに入力する単語
        # :param c: 内部メモリ
        # :param h: Decoderの中間ベクトル
        # :param f: Attention Modelで計算された順向きEncoderの加重平均
        # :param b: Attention Modelで計算された逆向きEncoderの加重平均
        # :return: 語彙数サイズのベクトル、更新された内部メモリ、更新された中間ベクトル
        e = F.tanh(self.ye(y)) # 単語を単語ベクトルに変換
        c, h = F.lstm(c, self.eh(e) + self.hh(h) + self.fh(f) + self.bh(b)) # 単語ベクトル、Decoderの中間ベクトル、順向きEncoderのAttention、逆向きEncoderのAttentionを使ってLSTM
        t = self.ey(F.tanh(self.he(h))) # LSTMから出力された中間ベクトルを語彙数サイズのベクトルに変換する
        return t, c, h

# Attentionモデルクラス
class Attention(Chain):
    def __init__(self, hidden_size):
        # Attentionのインスタンス化
        # :param hidden_size: 隠れ層のサイズ
        super(Attention, self).__init__(
            fh = L.Linear(hidden_size, hidden_size), # 順向きのEncoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
            bh = L.Linear(hidden_size, hidden_size), # 逆向きのEncoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
            hh = L.Linear(hidden_size, hidden_size), # Decoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
            hw = L.Linear(hidden_size, 1), # 隠れ層サイズのベクトルをスカラーに変換するための線形結合層
        )
        self.hidden_size = hidden_size # 隠れ層のサイズを記憶

    def __call__(self, fs, bs, h):
        # Attentionの計算
        # :param fs: 順向きのEncoderの中間ベクトルが記録されたリスト
        # :param bs: 逆向きのEncoderの中間ベクトルが記録されたリスト
        # :param h: Decoderで出力された中間ベクトル
        # :return: 順向きのEncoderの中間ベクトルの加重平均と逆向きのEncoderの中間ベクトルの加重平均
        batch_size = h.data.shape[0] # ミニバッチのサイズを記憶
        ws = [] # ウェイトを記録するためのリストの初期化
        sum_w = Variable(xp.zeros((batch_size, 1), dtype='float32')) # ウェイトの合計値を計算するための値を初期化
        # Encoderの中間ベクトルとDecoderの中間ベクトルを使ってウェイトの計算
        for f, b in zip(fs, bs):
            w = F.tanh(self.fh(f)+self.bh(b)+self.hh(h)) # 順向きEncoderの中間ベクトル、逆向きEncoderの中間ベクトル、Decoderの中間ベクトルを使ってウェイトの計算
            w = F.exp(self.hw(w)) # softmax関数を使って正規化する
            ws.append(w) # 計算したウェイトを記録
            sum_w += w
        # 出力する加重平均ベクトルの初期化
        att_f = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
        att_b = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
        for f, b, w in zip(fs, bs, ws):
            w /= sum_w # ウェイトの和が1になるように正規化
            # ウェイト * Encoderの中間ベクトルを出力するベクトルに足していく
            att_f += F.reshape(F.batch_matmul(f, w), (batch_size, self.hidden_size))
            att_b += F.reshape(F.batch_matmul(b, w), (batch_size, self.hidden_size))
        return att_f, att_b

# Attention Sequence to Sequence Modelクラス
class AttSeq2Seq(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_col_size):
        # Attention + Seq2Seqのインスタンス化
        # :param vocab_size: 語彙数のサイズ
        # :param embed_size: 単語ベクトルのサイズ
        # :param hidden_size: 隠れ層のサイズ
        super(AttSeq2Seq, self).__init__(
            f_encoder = LSTMEncoder(vocab_size, embed_size, hidden_size), # 順向きのEncoder
            b_encoder = LSTMEncoder(vocab_size, embed_size, hidden_size), # 逆向きのEncoder
            attention = Attention(hidden_size), # Attention Model
            decoder = AttLSTMDecoder(vocab_size, embed_size, hidden_size) # Decoder
        )
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.decode_max_size = batch_col_size # デコードはEOSが出力されれば終了する、出力されない場合の最大出力語彙数
        # 順向きのEncoderの中間ベクトル、逆向きのEncoderの中間ベクトルを保存するためのリストを初期化
        self.fs = []
        self.bs = []

    def encode(self, words, batch_size):
        # Encoderの計算
        # :param words: 入力で使用する単語記録されたリスト
        # :param batch_size: ミニバッチのサイズ
        # :return:
        # 内部メモリ、中間ベクトルの初期化
        c = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
        h = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
        # 順向きのEncoderの計算
        for w in words:
            c, h = self.f_encoder(w, c, h)
            self.fs.append(h) # 計算された中間ベクトルを記録
        # 内部メモリ、中間ベクトルの初期化
        c = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
        h = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))        
        # 逆向きのEncoderの計算
        for w in reversed(words):
            c, h = self.b_encoder(w, c, h)
            self.bs.insert(0, h) # 計算された中間ベクトルを記録
        # 内部メモリ、中間ベクトルの初期化
        self.c = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))
        self.h = Variable(xp.zeros((batch_size, self.hidden_size), dtype='float32'))        

    def decode(self, w):
        # Decoderの計算
        # :param w: Decoderで入力する単語
        # :return: 予測単語
        att_f, att_b = self.attention(self.fs, self.bs, self.h)
        t, self.c, self.h = self.decoder(w, self.c, self.h, att_f, att_b)
        return t

    def reset(self):
        # インスタンス変数を初期化する
        # Encoderの中間ベクトルを記録するリストの初期化
        self.fs = []
        self.bs = []
        # 勾配の初期化
        self.zerograds()

    def __call__(self, enc_words, dec_words=None, train=True):
        # 順伝播の計算を行う関数
        # :param enc_words: 発話文の単語を記録したリスト
        # :param dec_words: 応答文の単語を記録したリスト
        # :param train: 学習か予測か
        # :return: 計算した損失の合計 or 予測したデコード文字列
        enc_words = enc_words.T
        if train:
            dec_words = dec_words.T
        batch_size = len(enc_words[0]) # バッチサイズを記録
        self.reset() # model内に保存されている勾配をリセット
        enc_words = [Variable(xp.array(row, dtype='int32')) for row in enc_words] # 発話リスト内の単語をVariable型に変更
        self.encode(enc_words, batch_size) # エンコードの計算
        t = Variable(xp.array([0 for _ in range(batch_size)], dtype='int32')) # <eos>をデコーダーに読み込ませる
        loss = Variable(xp.zeros((), dtype='float32')) # 損失の初期化
        ys = [] # デコーダーが生成する単語を記録するリスト
        # デコーダーの計算
        if train: # 学習の場合は損失を計算する
            for w in dec_words:
                y = self.decode(t) # 1単語ずつをデコードする
                t = Variable(xp.array(w, dtype='int32')) # 正解単語をVariable型に変換
                loss += F.softmax_cross_entropy(y, t) # 正解単語と予測単語を照らし合わせて損失を計算
            return loss
        else: # 予測の場合はデコード文字列を生成する
            for i in range(self.decode_max_size):
                y = self.decode(t)
                y = xp.argmax(y.data) # 確率で出力されたままなので、確率が高い予測単語を取得する
                ys.append(y)
                t = Variable(xp.array([y], dtype='int32'))
                if y == 0: # EOSを出力したならばデコードを終了する
                    break
            return ys

# 学習

# 教師データ
def ReadTestData():
    input_data = []
    output_data = []
    with open(".\\mine\\data\\input\\input.txt",mode='r',encoding="utf-8") as f:
        input_data = f.readlines()
    with open(".\\mine\\data\\output\\output.txt",mode='r',encoding="utf-8") as f:
        output_data = f.readlines()
    result = []
    for i in range(1000):
        # result.append([input_data[i].rsplit("\n"),output_data[i].rsplit("\n")])
        result.append([[input_data[i].replace( '\n' , '' )],[output_data[i].replace( '\n' , '' )]])
    return result

def ReadTestQuestionnaireData():
    input_data = []
    output_data = []
    with open(".\\mine\\Question\\input 入力\\inputText.txt",mode='r',encoding="utf-8") as f:
        input_data = f.readlines()
    with open(".\\mine\\Question\\output 出力\\output2.txt",mode='r',encoding="utf-8") as f:
        output_data = f.readlines()
    result = []
    for i in range(len(input_data)):
        # result.append([input_data[i].rsplit("\n"),output_data[i].rsplit("\n")])
        temp_index = output_data[i].index(".")+1
        temp_str = output_data[i][temp_index:]
        result.append([[input_data[i].replace( '\n' , '' )],[temp_str.replace( '\n' , '' )]])
    return result

# 学習開始
def StudyStart(output_path):
    print("Train")
    st = datetime.datetime.now()
    for epoch in range(EPOCH_NUM):
        # ミニバッチ学習
        perm = np.random.permutation(N) # ランダムな整数列リストを取得
        total_loss = 0
        for i in range(0, N, BATCH_SIZE):
            enc_words = data_converter.train_queries[perm[i:i+BATCH_SIZE]]
            dec_words = data_converter.train_responses[perm[i:i+BATCH_SIZE]]
            model.reset()
            loss = model(enc_words=enc_words, dec_words=dec_words, train=True)
            loss.backward()
            loss.unchain_backward()
            total_loss += loss.data
            opt.update()
        serializers.save_npz(output_path, model)
        if (epoch+1)%10 == 0:
            ed = datetime.datetime.now()
            print("epoch:\t{}\ttotal loss:\t{}\ttime:\t{}".format(epoch+1, total_loss, ed-st))
            st = datetime.datetime.now()

def SpeechStart():
    # while True:
    #     print("Predict input start")
    #     text = input()
    #     result = predict(model, text)
    #     print("".join(result[:-1]))
    print(predict(model, "名前は何ですか") == predict(model, "名前は何ですか"[:3]))
    print(predict(model, "名前は何ですか"[:3]))
    print(predict(model, "好きな食べ物は何ですか") == predict(model, "好きな食べ物は何ですか"[:3]))
    print(predict(model, "好きな食べ物は何ですか"[:3]))
    print(predict(model, "どこに住んでいますか") == predict(model, "どこに住んでいますか"[:3]))
    print(predict(model, "どこに住んでいますか"[:3]))
    print(predict(model, "趣味は何ですか") == predict(model, "趣味は何ですか"[:3]))
    print(predict(model, "趣味は何ですか"[:3]))
    print(predict(model, "将来の夢は何ですか") == predict(model, "将来の夢は何ですか"[:3]))
    print(predict(model, "将来の夢は何ですか"[:3]))
    print(predict(model, "好きなドリンクは何ですか") == predict(model, "好きなドリンクは何ですか"[:3]))
    print(predict(model, "好きなドリンクは何ですか"[:3]))
    print(predict(model, "憧れている人は誰ですか") == predict(model, "憧れている人は誰ですか"[:3]))
    print(predict(model, "憧れている人は誰ですか"[:3]))
    print(predict(model, "最近どこに行きましたか") == predict(model, "最近どこに行きましたか"[:3]))
    print(predict(model, "最近どこに行きましたか"[:3]))
    print(predict(model, "現在働いているところはどこですか") == predict(model, "現在働いているところはどこですか"[:3]))
    print(predict(model, "現在働いているところはどこですか"[:3]))
    print(predict(model, "理想の恋人のタイプは何") == predict(model, "理想の恋人のタイプは何"[:3]))
    print(predict(model, "理想の恋人のタイプは何"[:3]))

def funcname(parameter_list):
    """
    docstring
    """
    pass

def predict(model, query):
    enc_query = data_converter.sentence2ids(query, train=False)
    dec_response = model(enc_words=enc_query, train=False)
    response = data_converter.ids2words(dec_response)
    # print(query, "=>", "".join(response[:-1]))
    return response

if __name__ == "__main__":
    # data = ReadTestData()
    data = ReadTestQuestionnaireData()
    # data = [
    #     [["初めまして。"], ["初めまして。よろしくお願いします。"]],
    #     [["どこから来たんですか？"], ["日本から来ました。"]],
    #     [["日本のどこに住んでるんですか？"], ["東京に住んでいます。"]],
    #     [["仕事は何してますか？"], ["私は会社員です。"]],
    #     [["お会いできて嬉しかったです。"], ["私もです！"]],
    #     [["おはよう。"], ["おはようございます。"]],
    #     [["いつも何時に起きますか？"], ["6時に起きます。"]],
    #     [["朝食は何を食べますか？"], ["たいていトーストと卵を食べます。"]],
    #     [["朝食は毎日食べますか？"], ["たまに朝食を抜くことがあります。"]],
    #     [["野菜をたくさん取っていますか？"], ["毎日野菜を取るようにしています。"]],
    #     [["週末は何をしていますか？"], ["友達と会っていることが多いです。"]],
    #     [["どこに行くのが好き？"], ["私たちは渋谷に行くのが好きです。"]]
    # ]
    # pprint.pprint(data)
    # pprint.pprint(ReadTestData())
    EMBED_SIZE = 100
    HIDDEN_SIZE = 100
    BATCH_SIZE = 6 # ミニバッチ学習のバッチサイズ数
    BATCH_COL_SIZE = 15
    EPOCH_NUM = 50 # エポック数
    N = len(data) # 教師データの数

    # 教師データの読み込み
    data_converter = DataConverter(batch_col_size=BATCH_COL_SIZE) # データコンバーター
    data_converter.load(data) # 教師データ読み込み
    vocab_size = len(data_converter.vocab) # 単語数
# 4　6
    # モデルの宣言
    model = AttSeq2Seq(vocab_size=vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, batch_col_size=BATCH_COL_SIZE)
    # ネットワークファイルの読み込み
    network = ".\\mine\\data\\network\\questionnaire\\second_test_q2.network"
    serializers.load_npz(network, model)
    opt = optimizers.Adam()
    opt.setup(model)
    opt.add_hook(optimizer.GradientClipping(5))
    if FLAG_GPU:
        model.to_gpu(0)
    model.reset()
    # pprint.pprint(data)
    # StudyStart(".\\mine\\data\\network\\questionnaire\\second_test_q2.network")
    SpeechStart()