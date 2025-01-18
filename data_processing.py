import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data():
    df_A = pd.read_csv('data/sample_feature_2D_A.csv')
    df_B = pd.read_csv('data/sample_feature_2D_B.csv')
    df_A_B = pd.read_csv('data/sample_feature_2D_A+B.csv')
    return df_A, df_B, df_A_B


def preprocess_data(df_A):
    X = df_A.iloc[:, 4:].fillna(0)
    yall = df_A.iloc[:, 2].fillna(0)

    # Encode y labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(yall)

    return X, y_encoded, label_encoder


def split_data(X, y):
    return train_test_split(X, y, test_size=0.5, random_state=41, stratify=y)
