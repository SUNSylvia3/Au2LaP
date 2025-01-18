# Au2LaP: Automated Descriptor Selection Enhanced 2D Material Layer Group Predictor

Crystal symmetry is a fundamental aspect of material properties and plays a pivotal role in the discovery and design of new materials. Au2LaP (Automated Descriptor Selection Enhanced 2D Material Layer Group Predictor) is the first machine learning framework specifically designed to predict **layer groups** of two-dimensional (2D) materials directly from their chemical composition.

## Key Features
- **Layer Group Prediction**: Predicts layer groups for 2D materials, capturing in-plane and out-of-plane symmetries often overlooked in space group classifications.
- **Automated Descriptor Selection**: Integrates LightGBM with SHAP to optimize predictive accuracy and interpretability using as few as 20 chemical descriptors.
- **High Performance**: Achieves state-of-the-art results with:
  - **Top-1 Accuracy**: 81.02%
  - **Top-3 Accuracy**: 90.48%
- **Explainable AI**: Provides transparency by identifying the most significant chemical descriptors contributing to predictions.
- **Polymorph Structure Prediction**: Capable of predicting multiple possible layer groups for given compositions, facilitating polymorph studies.

## Installation
To use Au2LaP, clone this repository and install the required dependencies.

```bash
git clone https://github.com/SUNSylvia3/Au2LaP-.git
cd Au2LaP-
pip install -r requirements.txt
