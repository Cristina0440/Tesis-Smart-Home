
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
    st.subheader("📊 Evaluación del Modelo de Regresión")

    st.markdown("""
    Para evaluar el rendimiento del modelo, se aplicaron dos estrategias complementarias:

    - **Validación Cruzada (Cross-Validation):** Técnica que permite estimar el rendimiento del modelo dividiendo el conjunto de entrenamiento en varios bloques (folds). Proporciona métricas promedio más estables.
    - **Evaluación Final en el Conjunto de Test:** Se separó el 20% del dataset original (con datos no vistos) para simular el comportamiento del modelo en un entorno real.

    La división fue aproximadamente 80% para entrenamiento y 20% para test final.
    """)

    # Resultados reales obtenidos por ti
    resumen_resultados = pd.DataFrame({
        "Modelo": ["Random Forest"],
        "MAE (CV)": [0.2497],
        "RMSE (CV)": [0.6006],
        "R² (CV)": [0.9957],
        "MAE (Test)": [0.2362],
        "RMSE (Test)": [0.5015],
        "R² (Test)": [0.9963]
    })

    st.dataframe(resumen_resultados.style.format({
        "MAE (CV)": "{:.4f}", "RMSE (CV)": "{:.4f}", "R² (CV)": "{:.4f}",
        "MAE (Test)": "{:.4f}", "RMSE (Test)": "{:.4f}", "R² (Test)": "{:.4f}"
    }))

    st.success("El modelo mostró un **alto rendimiento y generalización**, manteniendo métricas estables entre validación cruzada y evaluación final.")

    st.markdown("---")
    st.subheader("📋 Predicciones del Modelo (Casos de ejemplo)")

    data_pred = {
        "Caso": [1, 2, 3, 4, 5],
        "Valor Real (kWh)": [3.893, 14.496, 3.920, 31.990, 2.442],
        "Predicción (kWh)": [3.916, 14.426, 3.930, 32.027, 2.459]
    }

    df_pred = pd.DataFrame(data_pred)
    st.dataframe(df_pred, use_container_width=True)

    st.markdown("### 📉 Dispersión: Consumo Real vs Predicho")
    fig_dispersion = px.scatter(
        df_pred,
        x="Valor Real (kWh)",
        y="Predicción (kWh)",
        trendline="ols",
        labels={"Valor Real (kWh)": "Valor Real (kWh)", "Predicción (kWh)": "Predicción (kWh)"},
        title="Gráfico de Dispersión: Valor Real vs Predicho"
    )
    st.plotly_chart(fig_dispersion, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Comparación de TMS (RMSE) entre Evaluaciones")

    fig_tms = px.bar(
        x=["RMSE (Validación Cruzada)", "RMSE (Test Final)"],
        y=[0.6006, 0.5015],
        labels={"x": "Tipo de Evaluación", "y": "RMSE (W)"},
        text=[0.6006, 0.5015],
        title="Comparación de Error Cuadrático Medio (TMS / RMSE)"
    )
    fig_tms.update_traces(textposition='outside')
    st.plotly_chart(fig_tms, use_container_width=True)


"""
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
"""