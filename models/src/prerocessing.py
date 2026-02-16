from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


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