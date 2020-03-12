from .extract_entities import extract_entities


def get_sets(preds, y):
    pbar = enumerate(zip(preds, y))
    apset = []
    arset = []
    for i, (pred, y_true) in pbar:
        pset = extract_entities(pred)
        rset = extract_entities(y_true)
        for item in pset:
            apset.append(tuple([i] + list(item)))
        for item in rset:
            arset.append(tuple([i] + list(item)))
    return apset, arset
