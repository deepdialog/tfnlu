#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import sys
from tqdm import tqdm
import tensorflow as tf
import pickle
from tfnlu.parser import Parser


if len(sys.argv) != 2:
    print('train_parser.py <skip_number>')


root = '/media/qhduan/DATA10T/dp_data/'
paths = sorted([
    os.path.join(root, x)
    for x in os.listdir(root) if 'dp_out_' in x])


def get_data(skip=None):
    for i, path in enumerate(paths):
        x, y0, y1 = [], [], []
        if skip is not None and i < skip:
            yield x, y0, y1
            continue
        with open(path) as fp:
            for line in tqdm(fp):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                x.append(
                    ['[CLS]'] + list(obj.get('text')) + ['[SEP]']
                )
                y0.append(obj.get('mat_rel_type'))
                y1.append(
                    ['[CLS]'] + obj.get('bio').split() + ['[SEP]']
                )
        yield x, y0, y1
        x, y0, y1 = None, None, None
        x, y0, y1 = [], [], []


print(len(paths))

skip = int(sys.argv[1])
par = Parser(encoder_path='./encoders/zh-roberta-wwm-L12/')

if skip is not None:
    print('loading trained model')
    with open(f'./par_{skip}.pkl', 'rb') as fp:
        par2 = pickle.load(fp)
    for i, (x, y0, y1) in enumerate(get_data()):
        par.fit(x, y0, y1, build_only=True)
        break
    par.fit(x[:100], y0[:100], y1[:100])
    par.model.set_weights(par2.model.get_weights())
    print('loaded trained model')

for i, (x, y0, y1) in enumerate(get_data(skip)):
    print(f'{i + 1} / {len(paths)}')

    if i < skip:
        continue

    par.fit(
        x, y0, y1, epochs=1, batch_size=32,
        optimizer=tf.keras.optimizers.Adam(1e-6),
        optimizer_encoder=tf.keras.optimizers.Adam(5e-7)
    )
    x, y0, y1 = None, None, None

    if (i + 1) % 5 == 0:
        print(f'save to par_{i + 1}.pkl')
        with open(f'./par_{i + 1}.pkl', 'wb') as fp:
            pickle.dump(par, fp)
        exit(0)
    else:
        print('not save')


with open('./par_last_20200720.pkl', 'wb') as fp:
    pickle.dump(par, fp)
