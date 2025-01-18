import pickle
import pandas as pd


def load_model(model_path):
    """Load the trained model from a file."""
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    print("Model loaded from", model_path)
    return clf


def load_predict_data(file_path):
    """Load prediction data from a CSV file."""
    try:
        predict_data = pd.read_csv(file_path)
        print(f"Loaded prediction data from {file_path}")
        return predict_data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None


def predict_new_data(model, new_data):
    """Predict the class for new data."""
    predictions = model.predict(new_data)
    return predictions


if __name__ == "__main__":
    # Step 1: Load the saved model
    model_path = 'saved_model/model.pkl'
    clf = load_model(model_path)

    # Step 2: Load prediction data
    predict_file_path = 'data/sample_feature_2D_predict.csv'
    predict_df = load_predict_data(predict_file_path)

    # Check if data was loaded successfully
    if predict_df is not None:
        # Step 3: Ensure the data has the correct features
        # You may need to preprocess this data to match the features used during training
        predict_df_filled = predict_df.fillna(0)  # Handle missing values, if any

        # Step 4: Perform prediction
        predictions = predict_new_data(clf, predict_df_filled)
        print("Predictions:")
        print(predictions)

        # Step 5: Save predictions to a CSV file
        predict_df['Predicted_Class'] = predictions
        output_file = 'data/sample_feature_2D_predictions_output.csv'
        predict_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
