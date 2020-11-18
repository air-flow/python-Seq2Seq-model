# python-seq2seq-model

air-flow/python-seq2seq-copus で作成した corpus をもとに、Seq2Seq モデル作成を行っていく。

# DEMO

```python
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
```

input : 初めまして。
output : 初めまして。よろしくお願いします。

# Features

input, output ファイルにそれぞれわけて作成した corpus を、model に変換。
簡易的なものを使用したいので、12 行程度のプログラムファイル中に記述している変数データを使用する。

# Requirement

- Python 3.7.3
- MeCab mecab of 0.996
- chainer 7.7.0

## application

none

## Python Install package

- gensim 3.8.3
- mecab 0.996.2
- mecab-python-windows 0.996.3
- chainer 7.7.0

# Usage

```bash
git clone https://github.com/air-flow/python-Seq2Seq-model.git
python main.py
```

# Note

このプログラムの諸注意事項
学習データは、各自であつめてください。

## MeCab

MeCab の辞書に、NEologd を使用してます。

## Works Cited

[Attention Seq2Seq](http://www.ie110704.net/2017/08/21/attention-seq2seq%E3%81%A7%E5%AF%BE%E8%A9%B1%E3%83%A2%E3%83%87%E3%83%AB%E3%82%92%E5%AE%9F%E8%A3%85%E3%81%97%E3%81%A6%E3%81%BF%E3%81%9F/)
