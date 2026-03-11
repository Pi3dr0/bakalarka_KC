import torch
import torch.nn as nn
import torch.optim as optim


import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Lasso

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier


from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


##==============================================================##
## Logisticka Regressia
##==============================================================##
def lr_model(
        max_iter: int,
        random_state: int,
        class_weight: str,
        C: float,
        penalty: str | None,
        solver
):
    return LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        class_weight=class_weight,
        C=C,
        penalty=penalty,    #type: ignore
        solver=solver
    )
##==============================================================##


##==============================================================##
## MLP
##==============================================================##
class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int):
        super().__init__()
        self.layer = nn.Sequential(
            #----------------------------- vrstva 1
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            #----------------------------- vrstva 2 H
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            #----------------------------- vrstva 3 H
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            #----------------------------- vrstva 4
            nn.Linear(16, output_size)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
            return self.layer(x)
    
##~~~~~~~~~~~~~~~~~~~~~~~~~Wrapper~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
class Wrapper_MLP(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 output_dim: int,
                 lr: float,
                 epochs: int):
        self.output_dim = output_dim
        self.lr = lr
        self.epochs = epochs
    
    def fit(self, X, y):
        X = np.array(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)

        input_dim = X.shape[1]

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        ##-----------------
        self.model_ = MLP(input_dim, self.output_dim)
        self.criterion_ = nn.BCEWithLogitsLoss()
        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        #self.optimizer_ = torch.optim.SGD(self.model_.parameters(), lr=self.lr)
        ##-----------------

        self.model_.train()
        for _ in range(self.epochs):
            self.optimizer_.zero_grad()
            outputs = self.model_(X)
            loss = self.criterion_(outputs, y)
            loss.backward()
            self.optimizer_.step()

        return self
    
    def predict(self, X):
        X = np.array(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = torch.tensor(X, dtype=torch.float32)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X)
            probs = torch.sigmoid(logits)

        return (probs > 0.5).int().numpy().ravel()

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = torch.tensor(X, dtype=torch.float32)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X)
            probs = torch.sigmoid(logits)

        probs = probs.numpy()
        return np.hstack([1 - probs, probs])

##==============================================================##
## KNN
##==============================================================##
def knn(n_neighbors,
        weights: str | None,
        algorithm: str,
        leaf_size: int,
        p: float):
    return KNeighborsClassifier(n_neighbors=n_neighbors,
                                weights=weights,        #type: ignore
                                algorithm=algorithm,    #type: ignore
                                leaf_size=leaf_size,
                                p=p)                    #type: ignore

##==============================================================##
## Random Forest
##==============================================================##
def random_forest(n_estimators: int,
                  criterion: str,
                  max_depth: int,
                  class_weight: str | None):
    return RandomForestClassifier(n_estimators=n_estimators,
                                  criterion=criterion,  #type: ignore
                                  max_depth=max_depth,
                                  class_weight=class_weight) #type: ignore
