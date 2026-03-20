import numpy as np
import os
import joblib


def get_model_type(test_name:str) -> str: # type: ignore
    """
    Funkcia sluzi na zisteie typu modelu z mena testu 
    a nasledne vratenie nazvu subora, v ktorom mozeme 
    hladat ulozeny test
    """
    if test_name.startswith("lr", 0, 2):
        return "lr"
    elif test_name.startswith("knn", 0, 3):
        return "knn"
    elif test_name.startswith("rf", 0, 2):
        return "rf"


def load_test(path: str) -> dict:
    """
    Funkcia sluzi na nacitanie ulozeneho modelu, y_true a y_proba
    """
    return {
        "model": joblib.load(os.path.join(path, "model.pkl")),
        "y_true": np.load(os.path.join(path, "y_true.npy")),
        "y_proba": np.load(os.path.join(path, "y_proba.npy"))
    }