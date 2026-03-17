# bakalarka_KC

## skratky

**lr** = Logistic Regression

**knn** = K-nearest neighbours

**rf** = Random Forest

**mlp** = Multy Layer Perceptron

## Format pisania testov

“meno_testu":[”typ_modelu”, “parameter1”, “parameter2”, …, “parameterN”]
"meno_testu":["typ_modelu", "typ_kalibracie", "threshold"]

priklad: “test1”:[”lr”, 123, “balanced”, 0.8, “lbfgs”, “l2”]

## Parametre podla typu modelu

**lr**: max_iter, class_weight, C, penalty, solver

**knn**: n_neighbours, weights, algoritms, leaf_size, p

**rf**: n_estimators, criterion, max_depth, class_weight, min_samples_split, min_samples_leaf, max_features

**mlp**: lr, epochs, optimizer