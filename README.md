# Au2LaP: Automated Descriptor Selection Enhanced 2D Material Layer Group Predictor

Crystal symmetry is a fundamental aspect of material properties and plays a pivotal role in the discovery and design of new materials. Au2LaP (Automated Descriptor Selection Enhanced 2D Material Layer Group Predictor) is the first machine learning framework specifically designed to predict **layer groups** of two-dimensional (2D) materials directly from their chemical composition.

## Key Features
- **Layer Group Prediction**: Predicts layer groups for 2D materials with composition.
- **Automated Descriptor Selection**: Integrates LightGBM or other models with SHAP to optimize predictive accuracy and interpretability.
- **Explainable AI**: Provides transparency by identifying the most significant chemical descriptors contributing to predictions.
- **Polymorph Structure Prediction**: Capable of predicting multiple possible layer groups for given compositions, facilitating polymorph studies.

## Installation
To use Au2LaP, clone this repository and install the required dependencies.


## Dataset Preparation
You can refer to our paper to prepare the dataset `A`, `B`, and `A+B` accordingly. 


```bash
git clone https://github.com/SUNSylvia3/Au2LaP-.git
cd Au2LaP-
pip install -r requirements.txt
