#!/usr/bin/env python
# coding: utf-8

import json
# import pickle

from tfnlu.parser import Parser


def main():
    x = [
        ['你', '好'],
        ['我', '不', '好'],
        ['你', '好'],
        ['你', '不', '好']
    ]

    y0 = [
        [
            ['', 'G'],
            ['', '']
        ],
        [
            ['', 'B', 'G'],
            ['', '', ''],
            ['', '', ''],
        ],
        [
            ['', 'G'],
            ['', '']
        ],
        [
            ['', 'B', 'G'],
            ['', '', ''],
            ['', '', ''],
        ]
    ]
    y1 = [
        ['', 'S'],
        ['', '', 'S'],
        ['', 'S'],
        ['', '', 'S'],
    ]

    par = Parser(
        './encoders/zh-roberta-wwm-L1',
        30, 30, 30, encoder_trainable=True)

    par.fit(x, y0, y1, batch_size=2, epochs=20)

    ret = par.predict(x)
    print(json.dumps(ret[0], indent=4))
    print(json.dumps(ret[1], indent=4))

    # with open('/tmp/par_test', 'wb') as fp:
    #     pickle.dump(par, fp)

    # with open('/tmp/par_test', 'rb') as fp:
    #     par2 = pickle.load(fp)

    # par2.fit(x, y0, y1, batch_size=2, epochs=5)

    # ret2 = par2.predict(x)
    # print(json.dumps(ret2[0], indent=4))
    # print(json.dumps(ret2[1], indent=4))


if __name__ == "__main__":
    main()
