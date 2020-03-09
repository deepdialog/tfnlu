#!/usr/bin/env python
# coding: utf-8

import pickle

import tensorflow as tf

from tfnlu.parser import Parser


def main():
    x = ['hello', 'world']

    y = [
        tf.random.uniform((2, 7, 7), maxval=1, dtype=tf.int32),
        tf.random.uniform((2, 7, 7), maxval=1, dtype=tf.int32)
    ]

    par = Parser('../bert-embs/output/bert_wwm_zh_seq', 50, 50, 5, 5)

    par.fit(x, y)

    par.predict(x)[0].shape

    with open('/tmp/par_test', 'wb') as fp:
        pickle.dump(par, fp)

    with open('/tmp/par_test', 'rb') as fp:
        par2 = pickle.load(fp)

    par2.predict(x)[0].shape


if __name__ == "__main__":
    main()
