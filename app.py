import gradio as gr
import pandas as pd


# Load CSV data
def load_data():
    try:
        data = pd.read_csv("data/sample_feature_2D_A.csv")  # Replace with your CSV file path
        return data
    except FileNotFoundError:
        return None


# Prediction function
def predict_layer_group(formula):
    if not formula.strip():
        return "Please enter a valid formula!", "", ""

    result = data[data['formula'] == formula]
    if not result.empty:
        layer_group = result.iloc[0]['Layer Group']
        layer_group_number = result.iloc[0]['Layer Group Number']
        return f"Prediction successful:", f"Layer Group: {layer_group}", f"Layer Group Number: {layer_group_number}"
    else:
        return "No matching results found!", "", ""


# Load data
data = load_data()
if data is None:
    raise FileNotFoundError("'data/sample_feature_2D_A.csv' not found! Please check the file path.")

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Formula Prediction Tool")

    with gr.Row():
        formula_input = gr.Textbox(label="Enter Formula", placeholder="e.g., H2O")

    predict_button = gr.Button("Predict")
    result_label = gr.Label(label="Result")
    layer_group_label = gr.Label(label="Layer Group")
    layer_group_number_label = gr.Label(label="Layer Group Number")

    predict_button.click(
        fn=predict_layer_group,
        inputs=[formula_input],
        outputs=[result_label, layer_group_label, layer_group_number_label],
    )

# Run application
if __name__ == "__main__":
    demo.launch()
