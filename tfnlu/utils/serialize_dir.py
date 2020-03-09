import os


def serialize_dir(root, relative='.'):
    files = os.listdir(os.path.join(root, relative))
    ret = []
    for f in files:
        path = os.path.join(root, relative, f)
        if os.path.isdir(path):
            ret += serialize_dir(root, os.path.join(relative, f))
        elif os.path.isfile(path):
            with open(path, 'rb') as fp:
                ret.append((relative, f, fp.read()))
    return ret


def deserialize_dir(root, data):
    for dirname, filename, binary in data:
        dirpath = os.path.join(root, dirname)
        os.makedirs(dirpath, exist_ok=True)
        filepath = os.path.join(dirpath, filename)
        with open(filepath, 'wb') as fp:
            fp.write(binary)
