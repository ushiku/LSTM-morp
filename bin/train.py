#! /usr/bin/env python
# -*- coding: utf-8 -*- 
from __future__ import print_function
import argparse
import math
import sys
import time
import re
import pickle
import  random
import collections

import numpy as np
from numpy.random import *
import six
from scipy.sparse import csr_matrix
from progressbar import ProgressBar, Percentage, Bar
import progressbar
import time

import chainer
from chainer import optimizers
from chainer import serializers
import chainer.functions as F
import chainer.links as L
from chainer import cuda

CORPUS_PATH = '../corpus/sample.txt'  # 学習するcorpusの場所
TEST_PATH = '../corpus/validation.txt'  # validationの場所
MODEL_SAVE_PATH = '../auto/model.dump'  # 学習したモデルの保存先
DICT_SAVE_PATH = '../auto/dict_list.dump'  # dictの保存先 

def make_index(file_path):
    '''
    ファイルを開いてIndexを生成する.  key=character, value=feature_number
    '''
    index = {'UNK':1, 'EOF':0}  # 全単語を含んだIndex
    feature_number = 2  # 素性番号
    with open(file_path, 'r') as file:  # この書き方推奨
        sentences = file.readlines()
    for sentence in sentences:
        words = sentence.split()  # word は、「醤油」を指す
        for word in words:
            chars = list(word)
            for char in chars:  # char は、 「醤」を指す
                if not char in index:  # 今回のchar が初出の時
                    index[char] = feature_number
                    feature_number += 1
    return index


def make_train(file_path):
    '''
    訓練データを作成する。
    inputは「私,は,元,気,で,す,EOF,EOF,EOF,EOF,EOF」の, outputは「0,0,0,0,0,0,1,1,0,1,0」
    まず、文字数はno-outputで、EOFが入りしだい、先頭の文字から始まり、文字の後ろにboundaryが存在するかどうかの2値分類を行い続ける
    '''
    input_data, output_data = [], []  # 一つの要素はinput, outputの一行単位
    with open(file_path, 'r') as file:  # 学習はsentence 単位
        sentences = file.readlines()
    for sentence in sentences:
        input_line, output_line = np.array([]), np.array([])
        words = sentence.strip().split()  # ['私', 'は', '元気', 'です']
        chars = ''.join(words)  # ['私', 'は', '元', '気', 'で', 'す']
        chars_length = len(chars)  # 文字数: 6
        for char in chars:
            if char in index:
                input_line = np.append(input_line, index[char])  #  ['私','は','元','気','で','す'](ただし、素性番号)
            else:
                input_line = np.append(input_line, index['UNK'])  # 未知文字
        output_line = np.append(output_line, np.zeros(chars_length))  # [0,0,0,0,0,0]
        input_line = np.append(input_line, np.zeros(chars_length - 1))  # 「私,は,元,気,で,す,EOF,EOF,EOF,EOF,EOF」
        for word in words:
            word_length = len(word)
            word_boundary_flag = np.zeros(word_length)  # 文字長のフラグ
            word_boundary_flag[-1] = 1  # 末尾はboundary)
            output_line = np.append(output_line, word_boundary_flag)  # 「0,0,0,0,0,0,1,1,0,1,0,1」(末尾に1が付いていることに注意)
        output_line = output_line[:-1]  # 末尾削除
        input_line = xp.array(input_line, dtype=np.int32)
        output_line = xp.array(output_line, dtype=np.int32)
        input_data.append(input_line) 
        output_data.append(output_line)
    input_data.sort(key=len)  # ndarrayの長さ順に並び替える # ここもしかして、対応取れてない?  # Chainerの高速バッチ学習の為
    input_data.reverse()
    output_data.sort(key=len)
    output_data.reverse()
    input_data = F.transpose_sequence(input_data)
    output_data = F.transpose_sequence(output_data)
    return input_data, output_data


# LSTMのネットワーク定義
class LSTM(chainer.Chain):
    def __init__(self, p, n_units, train=True):
        super(LSTM, self).__init__(
            embed=L.EmbedID(p, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.Linear(n_units, 2),
        )

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(h0)
        y = self.l2(h1)
        return y

    def reset_state(self):
        self.l1.reset_state()

# パラメータ設定
n_units = 512    # 隠れ層のユニット数

# 引数の処理
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

index = make_index(CORPUS_PATH)  # indexを作成
index_inv = {v:k for k, v in index.items()}  # indexを逆転

# モデルの準備
lstm = LSTM(len(index), n_units)
model = L.Classifier(lstm)
model.compute_accuracy = False

for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.2, 0.2, data.shape)

xp = cuda.cupy if args.gpu >= 0 else np
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# 訓練データの準備
input_data, output_data = make_train(CORPUS_PATH)  # 学習データを生成

# テストデータの準備
test_input_data, test_output_data = make_train(TEST_PATH)  # Validationデータを生成

# optimizerの設定
optimizer = optimizers.Adam()
optimizer.setup(model)

roop = 3000  # ループの回数
# 設定を表示する
print('------------------------------')
print('indexのサイズ:', len(index))
print('学習の文の数:', len(input_data[0].data))
print('ユニット数', n_units)
print('ループの回数', roop)
print('------------------------------')

# 訓練を行うループ
print('学習スタート')
display = 10  # 何回ごとに表示するか
total_loss = 0  # 誤差関数の値を入れる変数
k = 0  # counter
for seq in range(0, roop):
    lstm.reset_state()  # 前の系列の影響がなくなるようにリセット  # 1系列は、input_line, output_line
    p = ProgressBar(widgets=[Percentage(), Bar(), progressbar.Timer()], maxval=len(input_data)*roop).start()
    for input, output in zip(input_data, output_data):
    # lossの表示
        loss = model(input, output)
        total_loss += loss.data
        k += 1
        p.update(k)  # progress_bar
        model.zerograds()
        loss.backward()
        optimizer.update()
    if seq%display==display-1:
        print("sequence:{}, loss:{}".format(seq, total_loss))
        total_loss = 0

        #sample表示
        print('-------------------')
        print('サンプル表示')
        input_list = []
        output_list = []
        answer_list = []
        for input, output in zip(test_input_data, test_output_data):
            output_list.append(lstm(input))  # 出力の最大の単語を追加する
            answer_list.append(output)  # 正解データを出力する
        output_list = F.transpose_sequence(output_list)
        answer_list = F.transpose_sequence(answer_list)
        arg_output_list = []
        arg_output_line = np.array([])
        for output in output_list:
            for output_line in output:
                arg_output_line = np.append(arg_output_line, int(np.argmax(output_line.data)))
            arg_output_list.append(arg_output_line)
            arg_output_line = np.array([])
        for output, answer in zip(arg_output_list, answer_list):
            print('出力:', output)
            print('正解:', answer.data)
        print('--------------------')


f = open(MODEL_SAVE_PATH, 'wb')  # モデルをsave
pickle.dump(model, f)
f.close()

f = open(DICT_SAVE_PATH, 'wb')
pickle.dump(index, f)
f.close()

