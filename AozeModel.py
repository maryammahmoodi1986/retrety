


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class DozeModel:
  def __init__(self, X, y, n=None, tolerance=1e-10, max_iter=50, use_pca=False, use_scaler=False):
    # تبدیل DataFrame به numpy array در صورت نیاز
    if isinstance(X, pd.DataFrame):
      X = X.values
    if isinstance(y, pd.Series):
      y = y.values

    self.X = X
    self.y = y
    self.n = np.ones(len(y)) if n is None else n  # اگر n داده نشود، مقدار پیشفرض 1 است
    self.tolerance = tolerance
    self.max_iter = max_iter
    self.use_pca = use_pca
    self.use_scaler = use_scaler

    # اعمال StandardScaler در صورت نیاز
    if self.use_scaler:
      self.scaler = StandardScaler()
      self.X = self.scaler.fit_transform(self.X)

    # اعمال PCA در صورت نیاز
    if self.use_pca:
      self.pca = PCA()
      self.X = self.pca.fit_transform(self.X)

    self.b = np.zeros((self.X.shape[1], max_iter))
    self.z = np.zeros((len(y), max_iter))
    self.yhat = np.zeros((len(y), max_iter))
    self.syhat = np.zeros((self.X.shape[1], max_iter))
    self.kai2 = np.zeros(max_iter)
    self.e = np.zeros((self.X.shape[1], max_iter))
    self.sy = np.zeros(self.X.shape[1])
    self.p = np.zeros((len(y), max_iter))
    self.p[:, 0] = (y + 0.5) / (self.n + 0.5)
    self.yhat[:, 0] = y

  def fit(self):
    for i in range(self.max_iter - 1):
      self.p[:, i] = np.clip(self.p[:, i], self.tolerance, 1 - self.tolerance)
      self.z[:, i] = np.log(self.p[:, i] / (1 - self.p[:, i])) + (self.y - self.n * self.p[:, i]) / (self.n * self.p[:, i] * (1 - self.p[:, i]))
      W = np.diag(self.n * self.p[:, i] * (1 - self.p[:, i]))
      self.b[:, i + 1] = np.linalg.solve(self.X.T @ W @ self.X, self.X.T @ W @ self.z[:, i])
      xb = np.clip(self.X @ self.b[:, i + 1], -500, 500)
      self.p[:, i + 1] = np.exp(xb) / (1 + np.exp(xb))
      self.yhat[:, i + 1] = self.n * self.p[:, i + 1]
      self.syhat[:, i + 1] = self.yhat[:, i + 1] @ self.X
      self.sy = self.y @ self.X
      self.e[:, i + 1] = self.syhat[:, i + 1] - self.sy
      self.kai2[i + 1] = np.sum(((self.y - self.yhat[:, i + 1]) ** 2) / self.yhat[:, i + 1])
    return {
    'coefficients': self.b,
    'predicted_probabilities': self.p,
    'yhat': self.yhat,
    'syhat': self.syhat,
    'errors': self.e,
    'kai2': self.kai2
    }