#!/usr/bin/env python
# coding: utf-8

from tfnlu import Tagger


class TestTagger(object):

    def test_tagger(self):
        multi = 4
        x = [list(x)
             for x in ['我要去北京', '我要去巴黎', '今天天气不错', '明天天气不知道怎么样']] * multi
        y = [['O', 'O', 'O', 'Bcity', 'Icity'],
             ['O', 'O', 'O', 'Bcity', 'Icity'],
             ['Bdate', 'Idate', 'O', 'O', 'O', 'O'],
             ['Bdate', 'Idate', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
             ] * multi

        tag = Tagger()

        tag.fit(x, y, validation_data=(x, y), batch_size=4, epochs=2)

        pred = tag.predict(x[:4])

        for i in range(4):
            assert len(pred[i]) == len(x[i])
