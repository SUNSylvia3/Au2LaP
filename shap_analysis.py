import shap
import numpy as np
import matplotlib.pyplot as plt

def perform_shap_analysis(clf, X_train, X_test):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)

    # Compute average SHAP values across all classes
    average_shap_values = np.mean(shap_values, axis=2)

    # Generate SHAP summary plot
    feature_names = X_train.columns.tolist()
    shap.summary_plot(shap_values[:, :, 1], X_test, feature_names=feature_names, show=False)
    plt.show()
