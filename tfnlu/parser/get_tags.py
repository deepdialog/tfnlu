from collections import Counter
from tqdm import tqdm


def get_tags(y):
    tokens = Counter()
    for mat in tqdm(y):
        for line in mat:
            tokens.update(line)
    word_index = {'': 0}
    for k in tokens.keys():
        if k not in word_index:
            word_index[k] = len(word_index)
    index_word = {v: k for k, v in word_index.items()}
    return word_index, index_word
