
def extract_entities(seq: list, x=None) -> list:
    """Extract entities from a sequences

    ---
    input: ['B', 'I', 'I', 'O', 'B', 'I']
    output: [(0, 3, ''), (4, 6, '')]
    ---
    input: ['B-loc', 'I-loc', 'I-loc', 'O', 'B-per', 'I-per']
    output: [(0, 3, '-loc'), (4, 6, '-per')]
    ---
    input:
        seq=['B-loc', 'I-loc', 'I-loc', 'O', 'B-per', 'I-per']
        x='我爱你欧巴桑'
    output:
        [(0, 3, '-loc', '我爱你'), (4, 6, '-per', '巴桑')]
    """
    ret = []
    start_ind, start_type = -1, None
    for i, tag in enumerate(seq):
        if tag.startswith('S'):
            if x is not None:
                ret.append((i, i + 1, tag[1:], x[i:i + 1]))
            else:
                ret.append((i, i + 1, tag[1:]))
            start_ind, start_type = -1, None
        if tag.startswith('B') or tag.startswith('O'):
            if start_ind >= 0:
                if x is not None:
                    ret.append((start_ind, i, start_type, x[start_ind:i]))
                else:
                    ret.append((start_ind, i, start_type))
                start_ind, start_type = -1, None
        if tag.startswith('B'):
            start_ind = i
            start_type = tag[1:]
    if start_ind >= 0:
        if x is not None:
            ret.append((start_ind, len(seq), start_type, x[start_ind:]))
        else:
            ret.append((start_ind, len(seq), start_type))
        start_ind, start_type = -1, None
    return ret
