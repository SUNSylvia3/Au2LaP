import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import uniform
def random_search_lightgbm(X_train, y_train, X_test, y_test, cv_folds=5, n_iter=50, random_state=42):
    """
    Perform RandomizedSearchCV to find the best hyperparameters for LGBMClassifier.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_test: Testing features
    - y_test: Testing labels
    - cv_folds: Number of cross-validation folds
    - n_iter: Number of parameter settings sampled
    - random_state: Random seed for reproducibility

    Returns:
    - best_model: Best LGBMClassifier model after random search
    - best_params: Best hyperparameters found
    """
    # Define the parameter grid

    param_distributions = {

        'learning_rate': uniform(0.01, 0.1),
        'colsample_bytree': [0.8, 0.9, 1],
        'subsample': [0.8, 0.9, 1],
        'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3],
        'reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3],
        'num_leaves': range(20, 60, 2),
        'max_depth': [5, 6, 7, 8, 9, 10],
    }

    # Initialize the LGBMClassifier
    lgbm = LGBMClassifier(random_state=random_state)

    # Set up cross-validation strategy
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring='accuracy',
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    # Perform random search
    print("Starting RandomizedSearchCV...")
    random_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    print("Best parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy with best parameters: {acc:.4f}")

    return best_model, best_params

# Example usage (replace X_train, X_test, y_train, y_test with actual data)
if __name__ == "__main__":
    # Placeholder for actual train-test split data
    X_train, X_test, y_train, y_test = None, None, None, None  # Replace with actual data

    best_model, best_params = random_search_lightgbm(X_train, y_train, X_test, y_test)
