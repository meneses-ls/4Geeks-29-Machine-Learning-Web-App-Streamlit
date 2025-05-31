import streamlit as st
import pandas as pd
import joblib

@st.cache_data(show_spinner=False)
def load_model():
    return joblib.load('src/model.pkl')

def predict_diabetes(input_data, model):
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[:, 1]
    return prediction[0], prob[0]

def main():
    st.title("Predicción de Diabetes")
    st.write("Ingresa los datos para predecir si una persona es diabética o no.")

    pregnancies = st.number_input("Número de Embarazos (Pregnancies)", 0, 20, 0)
    glucose = st.number_input("Glucosa (Glucose)", 0, 300, 120)
    blood_pressure = st.number_input("Presión Sanguínea (BloodPressure)", 0, 200, 70)
    skin_thickness = st.number_input("Grosor de Piel (SkinThickness)", 0, 100, 20)
    insulin = st.number_input("Insulina (Insulin)", 0, 900, 79)
    bmi = st.number_input("Índice de Masa Corporal (BMI)", 0.0, 70.0, 25.0, format="%.2f")
    dpf = st.number_input("Función de Pedigrí de Diabetes (DiabetesPedigreeFunction)", 0.0, 3.0, 0.47, format="%.3f")
    age = st.number_input("Edad (Age)", 0, 120, 33)

    input_dict = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    input_df = pd.DataFrame([input_dict])
    model = load_model()

    if st.button("Predecir"):
        pred, prob = predict_diabetes(input_df, model)
        if pred == 1:
            st.error(f"Resultado: Posible diabetes. Probabilidad: {prob:.2%}")
        else:
            st.success(f"Resultado: No diabético. Probabilidad: {prob:.2%}")

if __name__ == "__main__":
    main()