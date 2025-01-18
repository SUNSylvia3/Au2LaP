import data_processing as dp
import model_training as mt
import shap_analysis as sa
import pickle
import pandas as pd
import numpy as np



def compute_shap_importance(model, data):
    explainer = sa.get_shap_explainer(model, data)
    shap_values = explainer.shap_values(data)
    return np.mean(np.abs(shap_values), axis=0)


def feature_selection(model, X_train):
    def rank_features_by_importance():
        shap_importance = compute_shap_importance(model, X_train)
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': shap_importance
        }).sort_values(by='importance', ascending=False)
        return importance_df

    ranked_features = rank_features_by_importance()
    return ranked_features['feature'].tolist(), ranked_features


def save_model(model, path):
    print(f"Saving model to {path}...")
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    print("Model successfully saved!")


def predict_and_save(clf, predict_file_path, top_features, output_file):
    try:
        predict_df = pd.read_csv(predict_file_path)
        print(f"Loaded prediction data from {predict_file_path}")

        predict_df_filled = predict_df.fillna(0)[top_features]
        predictions = clf.predict(predict_df_filled)

        predict_df['Predicted_Class'] = predictions
        predict_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except KeyError as e:
        print(f"Feature mismatch: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    def workflow():
        print("Starting workflow...")

        # Data Loading
        print("Loading datasets...")
        df_A, df_B, df_A_B = dp.load_data()

        # Preprocessing
        print("Preprocessing data...")
        X, y, label_encoder = dp.preprocess_data(df_A)
        X_train, X_test, y_train, y_test = dp.split_data(X, y)

        # Initial Model Training
        print("Training initial model...")
        clf = mt.train_lgbm(X_train, y_train, X_test, y_test)

        # Feature Selection
        print("Performing feature selection...")
        top_features, importance_df = feature_selection(clf, X_train)
        print(f"Selected top features: {top_features[:10]} (truncated)")

        X_train = X_train[top_features]
        X_test = X_test[top_features]

        # Retraining Model
        print("Retraining model with selected features...")
        clf = mt.train_lgbm(X_train, y_train, X_test, y_test)

        # Saving Model
        save_model(clf, 'saved_model/model.pkl')

        # Model Evaluation
        print("Evaluating model...")
        metrics = mt.evaluate_model(clf, X_test, y_test)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        # Step 9: SHAP Analysis
        print("Performing SHAP analysis on the final model...")
        sa.perform_shap_analysis(clf, X_train, X_test)

        # Step 10: Predictions
        print("Starting prediction...")
        predict_file_path = 'data/sample_feature_2D_predict.csv'
        output_file = 'data/sample_feature_2D_predictions_output.csv'
        predict_and_save(clf, predict_file_path, top_features, output_file)

    workflow()


if __name__ == "__main__":
    main()
