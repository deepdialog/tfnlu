from collections import Counter


def get_tags(y):
    tokens = Counter()
    for sent in y:
        tokens.update(sent)
    word_index = {'': 0}
    for i, k in enumerate(tokens.keys()):
        word_index[k] = i + 1
    index_word = {v: k for k, v in word_index.items()}
    return word_index, index_word
