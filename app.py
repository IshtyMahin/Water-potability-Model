import pickle
import numpy as np
import gradio as gr

from sklearn.base import BaseEstimator, TransformerMixin


class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(X)


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_water_potability(
    ph,
    Hardness,
    Solids,
    Chloramines,
    Sulfate,
    Conductivity,
    Organic_carbon,
    Trihalomethanes,
    Turbidity
):
    input_data = np.array([[
        ph,
        Hardness,
        Solids,
        Chloramines,
        Sulfate,
        Conductivity,
        Organic_carbon,
        Trihalomethanes,
        Turbidity
    ]])

    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        return f"Potable Water"
    else:
        return f"Not Potable Water"


interface = gr.Interface(
    fn=predict_water_potability,

    inputs=[
        gr.Number(label="pH", value=7.0),
        gr.Number(label="Hardness", value=200),
        gr.Number(label="Solids", value=20000),
        gr.Number(label="Chloramines", value=7),
        gr.Number(label="Sulfate", value=300),
        gr.Number(label="Conductivity", value=400),
        gr.Number(label="Organic Carbon", value=10),
        gr.Number(label="Trihalomethanes", value=60),
        gr.Number(label="Turbidity", value=4),
    ],

    outputs=gr.Textbox(label="Prediction Result"),

    examples=[
        [7.2, 180, 15000, 6.5, 320, 420, 8.5, 55, 3.5],

        [5.5, 350, 35000, 9.5, 450, 650, 18, 95, 8],

        [6.8, 250, 22000, 7.8, 380, 500, 12, 70, 5],
        [8.316766,214.373394,22018.417441,8.059332,356.886136,363.266516,18.436524,100.341674,4.628771]
    ],

    title="Water Potability Prediction System",

)

if __name__ == "__main__":
    interface.launch()
