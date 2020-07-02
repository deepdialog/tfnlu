
import pickle
import tensorflow as tf


class CheckValidation(tf.keras.callbacks.Callback):

    def __init__(self,
                 tagger,
                 batch_size,
                 validation_data=None,
                 save_best=None,
                 **kwargs):
        super(CheckValidation, self).__init__(**kwargs)
        self.tagger = tagger
        self.batch_size = batch_size
        self.validation_data = validation_data
        self.save_best = save_best
        self.best_f1 = None

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            return
        tf.print()
        validation_result = self.tagger.evaluate_table(
            x=self.validation_data[0],
            y=self.validation_data[1],
            batch_size=self.batch_size
        )
        f1 = validation_result.iloc[-1]['f1score']
        tf.print(validation_result)

        if self.save_best is None:
            return

        if self.best_f1 is None:
            self.best_f1 = f1
            tf.print(f'best score: {self.best_f1:.4f}')
            with open(self.save_best, 'wb') as fp:
                pickle.dump(self.tagger, fp)
        elif f1 > self.best_f1:
            tf.print(
                f'best score: {f1:.4f} > '
                f'before {self.best_f1:.4f} '
                'saved')
            self.best_f1 = f1
            with open(self.save_best, 'wb') as fp:
                pickle.dump(self.tagger, fp)
        elif f1 <= self.best_f1:
            tf.print(
                f'best score: {f1:.4f} <= '
                f'before {self.best_f1:.4f} '
                'do nothing')
