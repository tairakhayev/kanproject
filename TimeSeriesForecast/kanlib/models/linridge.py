import numpy as np
from sklearn.linear_model import Ridge

class LinRidgeBaseline:
    def __init__(self, alpha=1.0, target_mode="uni", C=9, L=500, H=125):
        self.alpha = alpha; self.target_mode = target_mode
        self.C=C; self.L=L; self.H=H
        self.model = Ridge(alpha=alpha)

    def fit(self, X, Y):
        Xf = X.reshape(len(X), self.C*self.L)
        if self.target_mode=="uni": y = Y
        else: y = Y.reshape(len(Y), self.C*self.H)
        self.model.fit(Xf, y)
        return self

    def predict(self, X):
        Xf = X.reshape(len(X), self.C*self.L)
        yhat = self.model.predict(Xf)
        if self.target_mode=="multi":
            yhat = yhat.reshape(len(X), self.C, self.H)
        return yhat.astype(np.float32)
