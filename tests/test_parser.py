#!/usr/bin/env python
# coding: utf-8

import os
import uuid
import pickle
import tempfile
from tfnlu import Parser


class TestParser(object):

    def test_parser(self):
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
            None, 5, 5, 5, n_layers=1
        )

        par.fit(x, y0, y1, batch_size=2, epochs=2)

        ret = par.predict(x)
        assert len(ret) == 2

        model = par
        path = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        with open(path, 'wb') as fp:
            pickle.dump(model, fp)
        with open(path, 'rb') as fp:
            model = pickle.load(fp)
        assert model.predict(x) == ret
