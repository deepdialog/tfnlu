#!/usr/bin/env python
# coding: utf-8

import pickle
from tfnlu.classification import Classification


def main():
    multi = 2
    x = ['我要去北京', '我要去巴黎', '今天天气不错', '明天天气不知道怎么样'] * multi
    y = ['A', 'A', 'B', 'B'] * multi

    print('samples:', len(x))

    tag = Classification(encoder_path='./encoders/bert_wwm_zh/')

    tag.fit(x, y, batch_size=2, epochs=2)

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
