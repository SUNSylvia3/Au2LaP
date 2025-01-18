from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_lgbm(X_train, y_train, X_test, y_test):
    clf = LGBMClassifier(
        n_estimators=10000,
        learning_rate=0.05,
        num_leaves=32,
        colsample_bytree=0.8,
        subsample=0.9,
        max_depth=8,
        reg_alpha=0.08,
        reg_lambda=0.01,
        min_split_gain=0.01,
        min_child_weight=2,
        verbose=-1,
        seed=71
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='multi_error',
        early_stopping_rounds=75,
        verbose=False
    )

    return clf


def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    metrics = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    return metrics
