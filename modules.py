import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats as spstat
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin

from typing import Optional


class DataLoader(object):

    def __init__(self, path_to_data: str, **_):
        super(DataLoader, self).__init__()

        self.path_to_data = path_to_data
        self.data = pickle.load(open(path_to_data, 'rb'), encoding='latin1').values.T

    def __getitem__(self, index):
        if not (0 <= index < 1):
            raise IndexError

        item = {
            'desc': '',
            'X': self.data,
            'y': np.zeros(self.data.shape[0])
        }

        return {'filename': os.path.basename(self.path_to_data), 'item': item}

    def __len__(self):
        return 1


class CorrelationModel(object):
    """
    Method for the identification and ranking of different patterns of distribution (ex. the bimodal one).
    Convolves correlation pattern with input data.
    Returns similarity rank: the higher the response, the more similar the input to pattern.
    """
    def __init__(self, n_bins: int = 10):
        super(CorrelationModel, self).__init__()

        x = np.linspace(0.0, 1.0, n_bins)
        self.pattern = spstat.norm.pdf(x, 0.0, 0.1) * 0.125 + spstat.norm.pdf(x, 0.8, 0.2) * 1
        self.pattern /= self.pattern.max()

    def fit(self, _, __):
        return self

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])

        for idx in range(X.shape[0]):
            rank = (X[idx] * self.pattern).sum()
            y_pred[idx] = rank

        return y_pred


class Scaling(TransformerMixin):

    def __init__(self, new_min: float = 0.0, new_max: float = 1.0, axis: Optional[int] = None):
        super(Scaling, self).__init__()

        self.new_min = new_min
        self.new_max = new_max
        self.axis = axis

    def fit(self, _: np.ndarray, __: Optional[np.ndarray]) -> 'Scaling':
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        old_min = np.expand_dims(X.min(axis=self.axis), axis=self.axis)
        old_max = np.expand_dims(X.max(axis=self.axis), axis=self.axis)

        old_range = (old_max - old_min)
        new_range = (self.new_max - self.new_min)

        X_transformed = (((X - old_min) * new_range) / old_range) + self.new_min

        return X_transformed


class DummyMetric(object):

    def __call__(self, _, __) -> float:
        return 0.0


class HistBuilder(TransformerMixin):

    def __init__(self, n_bins: int = 100):
        super(HistBuilder, self).__init__()

        self.n_bins = n_bins

    def fit(self, _: np.ndarray, __: Optional[np.ndarray]) -> 'HistBuilder':
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        edges = np.linspace(0, 1.0, self.n_bins + 1)
        hists = np.array([np.histogram(row, edges, density=True)[0] for row in X])

        return hists


class PredictionHists(object):

    def __init__(self,
                 filename: str,
                 use_raw: bool = False,
                 n_bins: Optional[int] = 15,
                 n_bm: int = 50,
                 path_to_orig: Optional[str] = None):

        super(PredictionHists, self).__init__()

        if not use_raw and n_bins is None:
            raise ValueError

        self.filename = filename
        self.use_raw = use_raw
        self.n_bins = n_bins
        self.n_bm = n_bm

        if path_to_orig is not None:
            self.X_orig: pd.DataFrame = pickle.load(open(path_to_orig, 'rb'), encoding='latin1').T
        else:
            self.X_orig = None

    def save(self, report):
        for entry in report:
            X = entry['X_test']
            y = entry['y_pred']

            if not self.use_raw:
                X = HistBuilder(n_bins=self.n_bins).transform(X)

            indices = y.argsort()
            X = X[indices]
            X_bm = X[::-1][:self.n_bm]

            filename = '{}_bm{}'.format(self.filename, self.n_bm)
            if self.X_orig is not None:
                X_orig_sorted = self.X_orig[::-1].iloc[indices].T
                X_orig_sorted.T.to_pickle(
                    os.path.join(
                        '/home/slipnitskaya/PycharmProjects/piRNA-analysis/data',
                        '{}.pkl'.format(filename)
                    ),
                    protocol=2
                )
            else:
                X_orig_sorted = None

            plt.title('Pattern')
            plt.bar(range(X_bm.shape[1]), CorrelationModel(n_bins=X_bm.shape[1]).pattern)
            plt.savefig(os.path.join(
                '/home/slipnitskaya/PycharmProjects/piRNA-analysis/images',
                '{}_hist_pattern.png'.format(filename)),
                bbox_inches='tight'
            )
            plt.close()
            # plt.show()

            num_items_to_plot = self.n_bm
            cols = 5
            rows = num_items_to_plot // cols
            rows += num_items_to_plot % cols
            pos = range(1, num_items_to_plot + 1)

            fig = plt.figure(1, figsize=(10, 20))
            suptitle = fig.suptitle('Top-{} selected TEs'.format(num_items_to_plot), y=1.02)

            for i in range(num_items_to_plot):
                ax = fig.add_subplot(rows, cols, pos[i])

                if X_orig_sorted is not None:
                    title = '{}'.format(X_orig_sorted.columns[i])
                    ax.set_title(title)

                    X_curr = X_orig_sorted.iloc[:, i].values
                    _, edges = np.histogram(X_curr, bins=X.shape[1], density=True)
                    width = (edges[-1] - edges[0]) / (X_bm.shape[1] - 1)

                    ax.bar(edges[:-1], X_bm[i], width=width)
                    ax.bar(edges[:-1], X_bm[i] * CorrelationModel(n_bins=X_bm.shape[1]).pattern, width=width)

                else:
                    ax.bar(range(X_bm.shape[1]), X_bm[i])
                    ax.bar(range(X_bm.shape[1]), X_bm[i] * CorrelationModel(n_bins=X_bm.shape[1]).pattern)

            fig.subplots_adjust(wspace=0.5, hspace=0.5)
            fig.tight_layout(pad=1.0)
            fig.savefig(os.path.join(
                '/home/slipnitskaya/PycharmProjects/piRNA-analysis/images',
                '{}_hist.png'.format(filename)),
                bbox_inches='tight',
                bbox_extra_artists=[suptitle])
            plt.close(fig)
            # plt.show()
