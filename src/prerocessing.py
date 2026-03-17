# Standardization
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

# Normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer


def get_scalar(scalar_type: str):
    """
    Docstring for get_scalar
    
    :param scalar_type: Description
    :type scalar_type: str
    """

    if scalar_type == "minmax":
        return MinMaxScaler()
    elif scalar_type == "standard":
        return StandardScaler()
    elif scalar_type == "robust":
        return RobustScaler()
    elif scalar_type == "maxabs":
        return MaxAbsScaler()
    elif scalar_type == "normal":
        return Normalizer()
    else:
        print(f"Nespravne zadany typ scalaru: {scalar_type}")
        print("Prednastaveny typ scalaru: StandardScalar()")
        return StandardScaler()


def scale_df(X_train, X_test, scalar):
    """
    Docstring for scale_df
    
    :param X_train: Description
    :param X_test: Description
    :param scalar: Description
    """

    X_test_scaled = scalar.fit_transform(X_train)
    X_train_scaled = scalar.fit_transform(X_test)

    return X_test_scaled, X_train_scaled