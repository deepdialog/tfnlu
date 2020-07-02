#!/usr/bin/env python
# coding: utf-8

import pickle
from tfnlu.classification import Classification


def main():
    multi = 2
    x = [
        ['我', '要', '去', '北', '京'],
        ['我', '要', '去', '巴', '黎'],
        ['今', '天', '天', '气', '不', '错'],
        ['明', '天', '天', '气', '不', '知', '道', '怎', '么', '样']
    ] * multi
    y = ['A', 'A', 'B', 'B'] * multi

    print('samples:', len(x))

    tag = Classification(
        encoder_path='./encoders/zh-roberta-wwm-L1',
        encoder_trainable=True)

    tag.fit(x, y, batch_size=2, epochs=2)

    print(tag.predict(x[:4]))

    with open('/tmp/model_s', 'wb') as fp:
        pickle.dump(tag, fp)

    with open('/tmp/model_s', 'rb') as fp:
        tag2 = pickle.load(fp)

    print(tag2.predict(x[:4]))

    tag2.fit(x, y, batch_size=4, epochs=2)

    with open('/tmp/model_s2', 'wb') as fp:
        pickle.dump(tag2, fp)

    with open('/tmp/model_s2', 'rb') as fp:
        tag3 = pickle.load(fp)

    print(tag3.predict(x[:4]))


if __name__ == "__main__":
    main()
