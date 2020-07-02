from collections import Counter


def get_tags(y):
    tokens = Counter()
    for sent in y:
        tokens.update(sent)
    word_index = {'': 0, '[CLS]': 1, '[SEP]': 2}
    for k in tokens.keys():
        if k not in word_index:
            word_index[k] = len(word_index)
    index_word = {v: k for k, v in word_index.items()}
    return word_index, index_word
