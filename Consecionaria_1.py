from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import re

# Carpeta donde está este script
script_dir = Path(__file__).parent.resolve()

# -----------------------------
# Rutas relativas
# -----------------------------
rand_search = joblib.load(script_dir / "modelo_entrenado.pkl")
base = pd.read_csv(script_dir / "brand_marca.csv", sep=";")
variabeles_categoricas = pd.read_csv(script_dir / "vars_cat.csv")
X_train = pd.read_csv(script_dir / "X_train_cols.csv")

variabeles_categoricas = variabeles_categoricas.drop(columns=['Unnamed: 0'])
X_train = X_train.drop(columns=['Unnamed: 0'])
X_train = X_train.columns

st.title("Estimador de precios de vehículos 🚗💰")


make = st.selectbox("Marca", base["make"].unique())
fila = base[base["make"] == make].iloc[0]
model = st.selectbox("Modelo", base[base["make"] == make]['model'].unique().tolist())
fila_model = base[base["model"] == model].iloc[0]

year = st.number_input("Año del vehículo", min_value=1990, max_value=2025, value=2018)
mileage = st.number_input("Kilometraje (millas)", min_value=0, value=41000)
engine_hp = st.selectbox("Potencia HP", re.findall(r"'(.*?)'", fila_model['engine_hp']))
owner_count = st.number_input("Número de dueños", min_value=1, value=2)

transmission = st.selectbox("Transmisión", re.findall(r"'(.*?)'", fila_model['transmission']))
accident_history = st.selectbox("Historial de accidentes", ["sin_accidente", "Minor", "Major"])
seller_type = st.selectbox("Tipo de vendedor", re.findall(r"'(.*?)'", fila_model['seller_type']))
condition = st.selectbox("Condición", re.findall(r"'(.*?)'", fila_model['condition']))
trim = st.selectbox("Versión / Trim", re.findall(r"'(.*?)'", fila_model['trim']))
fuel_type = st.selectbox("Combustible", re.findall(r"'(.*?)'", fila_model['fuel_type']))
drivetrain = st.selectbox("Tracción",  re.findall(r"'(.*?)'", fila_model['drivetrain']))

if st.button("Estimar precio"):
    datos = {
        'year': year,
        'mileage': mileage,
        'engine_hp': engine_hp,
        'owner_count': owner_count,
        'make': make,
        'model': model,
        'transmission': transmission,
        'accident_history': accident_history,
        'seller_type': seller_type,
        'condition': condition,
        'trim': trim,
        'fuel_type': fuel_type,
        'drivetrain': drivetrain,
    }

    X_new = pd.DataFrame([datos])

    X_new['brand_popularity'] = base.loc[base['model'] == base['model'], 'brand_popularit'].mean()

    X_new_onehot = pd.get_dummies(X_new, columns=variabeles_categoricas.columns)
    X_new_onehot = X_new_onehot.reindex(columns=X_train, fill_value=0)
    X_new_onehot = X_new_onehot.astype(float)

    # Predicción
    y_pred = rand_search.predict(X_new_onehot)
    st.success(f"💵 El precio estimado es: **${y_pred[0]:,.2f}**")
