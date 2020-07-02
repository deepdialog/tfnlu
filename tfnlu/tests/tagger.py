#!/usr/bin/env python
# coding: utf-8

import pickle
from tfnlu.tagger import Tagger


def main():
    multi = 4
    x = [
        list(x) for x in [
            '我要去北京', '我要去巴黎', '今天天气不错', '明天天气不知道怎么样'
        ]
    ] * multi
    y = [['O', 'O', 'O', 'Bcity', 'Icity'], ['O', 'O', 'O', 'Bcity', 'Icity'],
         ['Bdate', 'Iadate', 'O', 'O', 'O', 'O'],
         ['Bdate', 'Idate', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']] * multi

    print('samples:', len(x))

    tag = Tagger(
        encoder_path='./encoders/zh-bert-wwm-L1/',
        encoder_trainable=True)

    tag.fit(x, y, validation_data=(x, y), batch_size=2, epochs=2)

    print(tag.predict(x[:4]))

    with open('/tmp/model_s', 'wb') as fp:
        pickle.dump(tag, fp)

    with open('/tmp/model_s', 'rb') as fp:
        tag2 = pickle.load(fp)

    print(tag2.predict(x[:4]))

    with open('/tmp/model_s2', 'wb') as fp:
        pickle.dump(tag2, fp)

    with open('/tmp/model_s2', 'rb') as fp:
        tag3 = pickle.load(fp)

    print(tag3.predict(x[:4]))


if __name__ == "__main__":
    main()
