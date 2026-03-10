import sys
import json
from pathlib import Path
#sys.path.append(str(Path("../").resolve()))


import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sb


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay


from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline


from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE


from src.prerocessing import get_scalar
from src.prerocessing import scale_df
from src.models import lr_model
from src.models import Wrapper_MLP
from src.models import sgd_model
from src.models import lasso_model


from src.feature_selection import *


## ----- Funkcie -----

def get_oof_score(pipeline, X, y, cv):
    predict_proba_result = cross_val_predict(
        estimator=pipeline,
        X=X,
        y=y,
        cv=cv,
        method="predict_proba"
    )
    return predict_proba_result[:, 1]



## ----- Main -----
if __name__ == "__main__":
    ### ==== Nacitanie datastu ====
    conf_path: str = "conf/conf.json"
    df_path: str = "data/clean_ds.xlsx"
    df = pd.read_excel(df_path)


    ### ==== Nacitanie configuracneho suboru ====
    with open(conf_path, 'r') as file:
        data = json.load(file)

    config: dict = data["config"][0]
    n_netwrok: dict = data["n_network"][0]
    linear: dict = data["linear"][0]




    ## Rozdelenie datasetu
    X = df.drop('referred_CXL',
            axis=1
            )
    y = df.referred_CXL

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"]
        )

    ## Data Scaleing
    # --- scale ---
    scaler = get_scalar(config["scaling"])
    #X_train_scaled, X_test_scaled = scale_df(X_train, X_test, scalar)


    ## premenne
    max_iter: int = config["epoch"]
    random_state: int = config["random_state"]
    model_type: str = config["model"]
    n_features_to_selection: int = config["n_features_to_selection"]
    n_splits: int = config["n_splits"]

    input_dim: int = 57 #20 #58
    output_dim: int = n_netwrok["output_dim"]
    lr: float = n_netwrok["lr"]

    # Linear model configuration
    early_stopping: bool = bool(linear["early_stopping"])
    learning_rate: str = linear["learning_rate"]
    l1_ratio: float = linear["l1_ratio"]
    class_weights = linear["class_weight"]
    c: float = linear["c"]
    penalty: str = linear["penalty"]
    solver: str = linear["solver"]

    model_configs: dict = {
        "conf1":(1, "l2", {0: 1.5, 1:1.0}),
        "conf2":(0.1, "l2", {0: 1.5, 1:1.0}),
        "conf3":(10, "l2", {0: 1.5, 1:1.0}),
        "conf4":(1, "l2", "balanced"),
        "conf5":(0.1, "l2", "balanced"),
        "conf6":(10, "l2", "balanced"),
        "conf7":(1, "l1", {0: 1.5, 1:1.0}),
        "conf8":(0.1, "l1", {0: 1.5, 1:1.0}),
        "conf9":(10, "l1", {0: 1.5, 1:1.0}),
        "conf10":(1, "l1", "balanced"),
        "conf11":(10, "l1", "balanced")
    }
    roc_results = {}
    pr_results = {}

    #class_weights: dict = {0: 1.5,
    #                       1: 1.0}

    for name, (C, penalty, class_weights) in model_configs.items():

        solver = "liblinear" if penalty == "l1" else "lbfgs"

        ## preklad / vyber modelu
        models_dict: dict = {
            "lr": lr_model(max_iter=max_iter,
                        random_state=random_state,
                        class_weight=class_weights,
                        C=c,
                        penalty=penalty,
                        solver=solver),
            "mlp": Wrapper_MLP(output_dim=output_dim,
                            lr=lr,
                            epochs=max_iter),
            "sgd": sgd_model(max_iter=max_iter,
                            random_state=random_state,
                            learning_rate=learning_rate,
                            early_stopping=early_stopping,
                            class_weight=class_weights,
                            l1_ratio=l1_ratio),
            "lasso": lasso_model(max_iter=max_iter,
                                random_state=random_state)
        }

        # aplikacia modelu
        model = models_dict[model_type]

        """f2_score = make_scorer(fbeta_score, beta=2)

        eval_metrics: dict = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "recall_weighted": "recall_weighted",
            "roc_auc": "roc_auc",
            "f2_score": f2_score
        }"""


        ## Pipeline
        if model_type == "mlp":
            pipeline = Pipeline([
                ("scaler", scaler),
                ("classifier", model)
            ])
        else:
            pipeline = Pipeline([
                ("scaler", scaler),
                ("feature_selection", RFE(
                    estimator=model,
                    n_features_to_select=n_features_to_selection

                )),
                ("classifier", model)
            ])



        cv = StratifiedKFold(n_splits=n_splits,
                            shuffle=True,
                            random_state=random_state
                            )
        predict_results = cross_validate(pipeline,
                                X,
                                y,
                                cv=cv,
                                #scoring=eval_metrics
                                )
        
        y_scores = get_oof_score(pipeline=pipeline,
                                X=X,
                                y=y,
                                cv=cv)
        fpr, tpr, _ = roc_curve(y, y_scores)
        precision, recall, _ = precision_recall_curve(y, y_scores)

        roc_results[name] = (fpr, tpr)
        pr_results[name] = (precision, recall)
    


    """fpr, tpr, roc_threshold = roc_curve(y, y_scores)
    precision, recall, pr_threshold = precision_recall_curve(y, y_scores)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.show()
    PrecisionRecallDisplay(precision=precision,
                           recall=recall).plot()
    plt.show()"""

    for name, (fpr, tpr) in roc_results.items():
        plt.plot(fpr, tpr, label=name)
    plt.xlabel("False Positve Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    for name, (precision, recall) in pr_results.items():
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Revall Curve")
    plt.legend()
    plt.show()

    #print(predict_results)

    ## Evaluacia
    """acc = predict_results["test_accuracy"].mean() * 100
    prec = predict_results["test_precision"].mean() * 100
    rec = predict_results["test_recall"].mean() * 100
    rec_weig = predict_results["test_recall_weighted"].mean() * 100
    roc_auc = predict_results["test_roc_auc"].mean() * 100
    f2_sc = predict_results["test_f2_score"].mean() * 100

    print(f"Accuracy: {acc:.2f}%")
    print(f"Precision: {prec:.2f}%")
    print(f"Recall: {rec:.2f}%")
    print(f"Recall Weighted: {rec_weig:.2f}%")
    print(f"ROC AUC: {roc_auc:.2f}%")
    print(f"f2 score: {f2_sc:.2f}%")"""