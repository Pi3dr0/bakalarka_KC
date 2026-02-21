import torch
import torch.nn as nn
import torch.optim as optim


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Lasso


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
        penalty: str
):
    return LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        class_weight=class_weight,
        C=C,
        penalty=penalty
    )
##==============================================================##


##==============================================================##
## MLP
##==============================================================##
class mlp(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int):
        super(mlp, self).__init__()
        self.layer = nn.Sequential(
            #----------------------------- vrstva 1
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),      #! sem asi leaky ReLU
            #----------------------------- vrstva 2 H
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            #----------------------------- vrstva 3 H
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
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


def mlp_model(input_dim: int,
              output_dim: int,
              lr: float):
    mlp_model = mlp(input_size=input_dim,
                    output_size=output_dim)
    
    criterion = nn.BCEWithLogitsLoss()  #? Ake ine funkcie viem pouzit? 
    optimization = optim.Adam(mlp_model.parameters(), lr=lr)

##==============================================================##
## SGD
##==============================================================##
def sgd_model(
          max_iter: int,
          random_state: int,
          learning_rate: str,
          early_stopping: bool,
          class_weight,
          l1_ratio: float
          ):
     return SGDClassifier(max_iter=max_iter,
                          random_state=random_state,
                          learning_rate=learning_rate,
                          early_stopping=early_stopping,
                          class_weight=class_weight,
                          l1_ratio=l1_ratio
                          )

##==============================================================##
## Lasso
##==============================================================##
def lasso_model(max_iter: int,
                random_state: int
                ):
     return Lasso(
          max_iter=max_iter,
          random_state=random_state
     )
