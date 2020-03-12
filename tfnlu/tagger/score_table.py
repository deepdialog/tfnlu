import pandas as pd

from .get_sets import get_sets


def score_table(preds, y):
    """Calculate NER F1
    Based CONLL 2003 standard
    """
    apset, arset = get_sets(preds, y)
    types = [x[3] for x in apset] + [x[3] for x in arset]
    types = sorted(set(types))
    rows = []
    for etype in types:
        pset = set([x for x in apset if x[3] == etype])
        rset = set([x for x in arset if x[3] == etype])
        inter = pset.intersection(rset)
        precision = len(inter) / len(pset) if pset else 1
        recall = len(inter) / len(rset) if rset else 1
        f1score = 0
        if precision + recall > 0:
            f1score = 2 * ((precision * recall) / (precision + recall))
        rows.append((etype, precision, recall, f1score))
    pset = set(apset)
    rset = set(arset)
    inter = pset.intersection(rset)
    precision = len(inter) / len(pset) if pset else 1
    recall = len(inter) / len(rset) if rset else 1
    f1score = 0
    if precision + recall > 0:
        f1score = 2 * ((precision * recall) / (precision + recall))
    rows.append(('TOTAL', precision, recall, f1score))
    df = pd.DataFrame(rows,
                      columns=['name', 'precision', 'recall', 'f1score'])
    df.index = df['name']
    df = df.drop('name', axis=1)
    return df
