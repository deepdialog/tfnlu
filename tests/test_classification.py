#!/usr/bin/env python
# coding: utf-8

import os
import uuid
import pickle
import tempfile
from tfnlu import Classification


class TestClassification(object):

    def test_classification(self):
        multi = 2
        x = [
            ['我', '要', '去', '北', '京'],
            ['我', '要', '去', '巴', '黎'],
            ['今', '天', '天', '气', '不', '错'],
            ['明', '天', '天', '气', '不', '知', '道', '怎', '么', '样']
        ] * multi
        y = ['A', 'A', 'B', 'B'] * multi

        mod = Classification()

        mod.fit(x, y, batch_size=2, epochs=2)
        mod.fit(x, y, batch_size=2, epochs=2)

        pred = mod.predict(x[:4])
        assert len(pred) == 4

        model = mod
        path = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        with open(path, 'wb') as fp:
            pickle.dump(model, fp)
        with open(path, 'rb') as fp:
            model = pickle.load(fp)
        assert model.predict(x[:4]) == pred
