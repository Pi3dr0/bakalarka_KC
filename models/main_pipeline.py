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
from src.models import knn
from src.models import random_forest


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
    ### ======================
    ### ==== Load dataset ====
    ### ======================
    df_path: str = "data/clean_ds.xlsx"
    df = pd.read_excel(df_path)


    ### ====================================
    ### ==== Load data from config file ====
    ### ====================================
    conf_path: str = "conf/conf.json"
    with open(conf_path, 'r') as file:
        data = json.load(file)

    config: dict = data["config"][0]
    model_configs: dict = data["n_tests"][0]

    ## Config <-----
    max_iter: int = config["epoch"]
    random_state: int = config["random_state"]
    model_type: str = config["model"]
    n_features_to_selection: int = config["n_features_to_selection"]
    n_splits: int = config["n_splits"]
    class_weights = config["class_weight"]
    n_tests: str = config["n_tests"]

    ## n_tests <-----
    model_configs_fixed: dict = {}
    for test_name, params in model_configs.items():
        fixed_params: list = []
        for p in params:
            if isinstance(p, dict):
                p_fixed = {int(k): v for k, v in p.items()}
            else:
                p_fixed = p
            p_fixed = p_fixed if p_fixed != "None" else None
            fixed_params.append(p_fixed)
        model_configs_fixed[test_name] = fixed_params
    model_configs = model_configs_fixed
    roc_results = {}
    pr_results = {}


    ### ==========================
    ### ==== Train Test split ====
    ### ==========================
    X = df.drop('referred_CXL',
                axis=1)
    y = df.referred_CXL

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"]
        )


    ### ===========================================
    ### ==== Application of the scaling method ====
    ### ===========================================
    scaler = get_scalar(config["scaling"])


    ### 
    ### 
    ###
    for test_name, params in model_configs.items():

        model_type = params[0]
        print(params)
        print(params[2])
        
        if model_type == "lr":
            model = lr_model(max_iter=params[1],
                             random_state=random_state,
                             class_weight=params[2],
                             C=params[3],
                             penalty=params[4],
                             solver=params[5])
        elif model_type == "mlp":
            model = Wrapper_MLP(output_dim=1,
                                lr=params[1],
                                epochs=params[2])
        elif model_type == "knn":
            model = knn(n_neighbors=params[1],
                        weights=params[2],
                        algorithm=params[3],
                        leaf_size=params[4],
                        p=params[5])
        elif model_type == "rf":
            model = random_forest(n_estimators=params[1],
                                  criterion=params[2],
                                  max_depth=params[3],
                                  class_weight=params[4])
        else:
            raise ValueError(f"Unknown model_type: {model_type}")


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
        if model_type in ["mlp", "knn"]:
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

        roc_results[test_name] = (fpr, tpr)
        pr_results[test_name] = (precision, recall)
    


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