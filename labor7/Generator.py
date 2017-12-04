
'''
https://github.com/moannuo/goodtoknow/blob/master/ML/Kaggle/KeypointsFacialKeras/generator.py

'''


import numpy as np


class Generator(object):
    """Several useful methods in order to expand our dataset at train time and
    help generalize our model.
    Sources: http://cs231n.stanford.edu/reports2016/010_Report.pdf
    """

    def __init__(self,
                 X_train,
                 Y_train,
                 batchsize=32,
                 flip_ratio=0.5,
                 flip_indices=[(0, 2), (1, 3), (4, 8), (5, 9), (6, 10), (7, 11), (12, 16), (13, 17), (14, 18), (15, 19), (22, 24), (23, 25)]
                 ):
        """
        Arguments
        ---------
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.size_train = X_train.shape[0]
        self.batchsize = batchsize
        self.flip_ratio = flip_ratio
        self.flip_indices = flip_indices

    def _random_indices(self, ratio):
        """Generate random unique indices according to ratio"""
        size = int(self.actual_batchsize * ratio)
        return np.random.choice(self.actual_batchsize, size, replace=False)

    def flip(self):
        """Flip image batch"""
        indices = self._random_indices(self.flip_ratio)
        self.inputs[indices] = self.inputs[indices, :, ::-1, :] # Flip cols
        self.targets[indices, ::2] = self.targets[indices, ::2] * -1
        for a, b in self.flip_indices:
            self.targets[indices, a], self.targets[indices, b] = self.targets[indices, b], self.targets[indices, a]

    def generate(self, batchsize=32):
        """Generator"""
        while True:
            cuts = [(b, min(b + self.batchsize, self.size_train)) for b in range(0, self.size_train, self.batchsize)]
            for start, end in cuts:
                self.inputs = self.X_train[start:end].copy()
                self.targets = self.Y_train[start:end].copy()
                self.actual_batchsize = self.inputs.shape[0]  # Need this to avoid indices out of bounds
                self.flip()
                yield (self.inputs, self.targets)
