from collections import Counter


def get_tags(y):
    targets = Counter()
    targets.update(y)
    word_index = {}
    for k in targets.keys():
        if k not in word_index:
            word_index[k] = len(word_index)
    index_word = {v: k for k, v in word_index.items()}
    return word_index, index_word
