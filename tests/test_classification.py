#!/usr/bin/env python
# coding: utf-8

# import pickle
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

        tag = Classification()

        tag.fit(x, y, batch_size=2, epochs=2)
        tag.fit(x, y, batch_size=2, epochs=2)

        pred = tag.predict(x[:4])
        assert len(pred) == 4
