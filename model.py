import pickle
from sklearn.base import ClassifierMixin
import numpy as np
class IdentityPassthrough(ClassifierMixin):
    def __init__(self):
        self.classes_ = np.array([-1, 1])
    def fit(self, X, y):
        return self
    def predict(self, X):
        return X
    def predict_proba(self, X):
        return X
    def get_params(self, deep=True):
        return {}
    def classes_(self):
        return self.classes_
    def set_params(self, **kargs):
        return self

class columnDropperTransformer():
    def __init__(self, columnIdxs):
        self.columnIdxs = columnIdxs

    def transform(self, X, y=None):
        return np.delete(X, self.columnIdxs, axis=1)

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)
    
