import sys
import json
import joblib
import os
from pathlib import Path


import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss


from sklearn.calibration import CalibratedClassifierCV


from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline


from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE


from prerocessing import get_scalar
from prerocessing import scale_df
from models import lr_model         # type: ignore
from models import Wrapper_MLP      # type: ignore
from models import knn              # type: ignore
from models import random_forest    # type: ignore


from feature_selection import *


## ----- Funkcie -----
def save_test_results(test_name: str,
                      model,
                      y_true,
                      y_proba,
                      base_path: str):
    """
    Funkcia sluzi na ulozenie modelu, y_true a y_proba 
    do adresara results
    """
    path: str = os.path.join(base_path, test_name)
    os.makedirs(path, exist_ok=True)

    # ulozenie modelu
    joblib.dump(model, os.path.join(path, "model.pkl"))
    
    # ulozenie predikcii
    np.save(os.path.join(path, "y_true.npy"), y_true)
    np.save(os.path.join(path, "y_proba.npy"), y_proba)



def get_oof_score(pipeline, X, y, cv):
    """
    funkcia vracia Out-Of-Fold score
    """
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
    random_state: int = config["random_state"]
    n_features_to_selection: int = config["n_features_to_selection"]
    n_splits: int = config["n_splits"]
    threshold: float = config["threshold"]
    calibration: str = config["calibration"]
    calibration_type: str = config["calibration_type"]


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


    ### =======================
    ### ==== Training loop ====
    ### =======================
    for test_name, params in model_configs.items():

        model_type = params[0]
        
        if model_type == "lr":
            base_model = lr_model(max_iter=params[1],
                             random_state=random_state,
                             class_weight=params[2],
                             C=params[3],
                             penalty=params[4],
                             solver=params[5])
        elif model_type == "mlp":
            base_model = Wrapper_MLP(output_dim=1,
                                lr=params[1],
                                epochs=params[2])
        elif model_type == "knn":
            base_model = knn(n_neighbors=params[1],
                        weights=params[2],
                        algorithm=params[3],
                        leaf_size=params[4],
                        p=params[5])
        elif model_type == "rf":
            base_model = random_forest(n_estimators=params[1],
                                  criterion=params[2],
                                  max_depth=params[3],
                                  class_weight=params[4],
                                  min_samples_split=params[5],
                                  min_samples_leaf=params[6],
                                  max_features=params[7],
                                  random_state=random_state)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")


        # ===============================
        # === Aplikovanie kalibracie ====
        # ===============================
        if calibration == "True":
            model = CalibratedClassifierCV(
                base_model,
                method=calibration_type, # type: ignore
                cv=n_splits
            )
            print(f"calibration: {calibration}")
        else:
            model = base_model
            print("calibration: None")



        ## Pipeline
        if model_type in ["mlp", "knn", "rf", "lr"]:
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
                            random_state=random_state)
        predict_results = cross_validate(pipeline,
                                X,
                                y,
                                cv=cv)


        y_scores = get_oof_score(pipeline=pipeline,
                                X=X,
                                y=y,
                                cv=cv)
        
        # thresholds
        y_pred = (y_scores >= threshold).astype(int)

        acc = accuracy_score(y, y_pred)
        precision_sc = precision_score(y, y_pred)
        recall_sc = recall_score(y, y_pred)
        f2 = fbeta_score(y, y_pred, beta=2)

        print("Threshold:", threshold)
        print("Accuracy:", acc)
        print("Precision:", precision_sc)
        print("Recall:", recall_sc)
        print("F2:", f2)

        # ==========================
        # ==== Ukladanie modelo ====
        # ==========================
        save_test_results(test_name=test_name,
                          model=pipeline,
                          y_true=y,
                          y_proba=y_scores,
                          base_path=f"results/{model_type}")
    


    """fpr, tpr, roc_threshold = roc_curve(y, y_scores)
    precision, recall, pr_threshold = precision_recall_curve(y, y_scores)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.show()
    PrecisionRecallDisplay(precision=precision,
                           recall=recall).plot()
    plt.show()"""

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