
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def mostrar_modelos():
    st.header("🧠 Evaluación de Modelos ML y Regresión")

    st.markdown("Esta sección presenta la comparación de modelos de clasificación utilizados en la plataforma, así como un modelo de regresión para predecir el consumo energético continuo.")

    st.markdown("---")
    st.subheader("🔌 Variables utilizadas en cada modelo")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Modelo de Clasificación del Consumo")
        st.markdown("""
        - `mean_voltage`: Voltaje promedio del aparato conectado.  
        - `mean_current`: Corriente eléctrica promedio.  
        - `mean_pf`: Factor de potencia (eficiencia del uso energético).  
        """)

    with col2:
        st.markdown("#### Modelo de Clasificación del Dispositivo")
        st.markdown("""
        - `mean_voltage`: Voltaje promedio medido en el dispositivo.  
        - `mean_current`: Corriente promedio consumida.  
        - `mean_power`: Potencia activa promedio.  
        - `mean_pf`: Factor de potencia (eficiencia eléctrica).  
        """)

    st.markdown("---")
    st.subheader("⚡ Comparación de Modelos – Clasificación del Consumo (Alto/Normal)")

    datos_consumo = {
        "Modelo": [
            "Regresión Logística", "Árbol de Decisión", "Random Forest", 
            "Gradient Boosting", "XGBoost"
        ],
        "Accuracy": [0.744, 0.978, 0.983, 0.983, 0.989],
        "Precision": [0.778, 0.989, 1.000, 1.000, 1.000],
        "Recall": [0.729, 0.969, 0.969, 0.969, 0.979],
        "F1-score": [0.753, 0.979, 0.984, 0.984, 0.989]
    }

    df_consumo = pd.DataFrame(datos_consumo)

    def resaltar_consumo(fila):
        return ['background-color: lightgreen' if fila["Modelo"] == "XGBoost" else "" for _ in fila]

    st.dataframe(df_consumo.style.apply(resaltar_consumo, axis=1))
    st.success("Se seleccionó **XGBoost** como modelo final por su gran balance entre precisión, recall y F1-score.")

    st.markdown("---")
    st.subheader("🔍 Comparación de Modelos – Clasificación del Tipo de Dispositivo")

    datos_dispositivo = {
        "Modelo": [
            "Regresión Logística", "Árbol de Decisión", "Random Forest",
            "Gradient Boosting", "XGBoost"
        ],
        "Accuracy": [0.761, 0.917, 0.950, 0.922, 0.911],
        "F1-score": [0.703, 0.918, 0.941, 0.916, 0.906]
    }

    df_dispositivo = pd.DataFrame(datos_dispositivo)

    def resaltar_dispositivo(fila):
        return ['background-color: lightblue' if fila["Modelo"] == "Random Forest" else "" for _ in fila]

    st.dataframe(df_dispositivo.style.apply(resaltar_dispositivo, axis=1))
    st.success("Se seleccionó **Random Forest** como modelo para detección de dispositivos debido a su alta precisión multiclase.")

    st.markdown("---")
    st.subheader("📈 Modelo de Regresión ")

    df = pd.read_csv("smart_home.csv")

    def extract_mean(column):
        return df[column].apply(lambda x: np.mean(eval(x)) if pd.notna(x) else np.nan)

    df["mean_voltage"] = extract_mean("voltages")
    df["mean_current"] = extract_mean("currents")
    df["mean_pf"] = extract_mean("powerFactors")
    df["mean_power"] = extract_mean("activePowers")

    df = df.dropna(subset=["mean_voltage", "mean_current", "mean_pf", "mean_power", "appliance", "brand", "application"])

    le_appliance = LabelEncoder()
    le_brand = LabelEncoder()
    le_event = LabelEncoder()

    df["appliance_encoded"] = le_appliance.fit_transform(df["appliance"].astype(str))
    df["brand_encoded"] = le_brand.fit_transform(df["brand"].astype(str))
    df["event_encoded"] = le_event.fit_transform(df["application"].astype(str))

    X = df[["mean_voltage", "mean_current", "mean_pf", "appliance_encoded", "brand_encoded", "event_encoded"]]
    y = df["mean_power"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_rf.fit(X_train, y_train)

    y_pred = modelo_rf.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.markdown("### 📊 Métricas del modelo")
    st.markdown(f"- **MAE:** {mae:.3f} W")
    st.markdown(f"- **RMSE:** {rmse:.3f} W")
    st.markdown(f"- **R²:** {r2:.3f}")

    df_pred = pd.DataFrame({
        "consumo_real": y_test,
        "consumo_predicho": y_pred
    })

    fig = px.scatter(
        df_pred,
        x="consumo_real",
        y="consumo_predicho",
        trendline="ols",
        title="Consumo Real vs Predicho (Modelo)",
        labels={"consumo_real": "Consumo Real (W)", "consumo_predicho": "Consumo Predicho (W)"}
    )
    st.plotly_chart(fig, use_container_width=True)
