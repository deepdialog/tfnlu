import tensorflow as tf


class ToPointerTags(tf.keras.layers.Layer):
    def __init__(self, index_word, **kwargs):
        super(ToPointerTags, self).__init__(**kwargs)
        b_keys = sorted([
            x for x in index_word.values()
            if x.startswith('B')
        ])
        i_keys = sorted([
            x for x in index_word.values()
            if x.startswith('I')
        ])
        assert len(b_keys) == len(i_keys)
        self.b_keys = b_keys
        self.i_keys = i_keys

        new_word_index = {'': 0}
        for b in b_keys:
            new_word_index[b] = len(new_word_index)
        for i in i_keys:
            new_word_index[i] = len(new_word_index)
        new_index_word = {
            k: v
            for v, k in new_word_index.items()
        }

        keys = tf.constant(list(new_index_word.keys()), dtype=tf.int32)
        values = tf.constant(list(new_index_word.values()), dtype=tf.string)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            tf.constant(''))  # default value

    def call(self, inputs):
        """
        inputs: [tokens * 2, batch_size, max_len]
        """
        x = inputs
        iret = tf.zeros(tf.shape(x)[1:], dtype=tf.int32)
        for i in range(len(self.b_keys) * 2):
            iret += x[i, :] * (i + 1)
        ret = self.table.lookup(iret)
        return ret
