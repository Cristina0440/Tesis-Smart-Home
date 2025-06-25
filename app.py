import streamlit as st
import pandas as pd
import numpy as np
import joblib

from modulo_sistema import mostrar_sistema
from modulo_analisis import mostrar_analisis
from modulo_automatico import mostrar_automatico
from modulo_modelos import mostrar_modelos


# ---------------- CONFIGURACIÓN INICIAL ---------------- #
st.set_page_config(page_title="Smart Home Modular", layout="wide")
st.title("🏠 Plataforma Modular de Consumo Energético Inteligente")

# ---------------- CARGA DE MODELOS ---------------- #
modelo_consumo = joblib.load("modelo_consumo.pkl")
modelo_appliance = joblib.load("modelo_appliance.pkl")
encoder_appliance = joblib.load("encoder_appliance.pkl")

# ---------------- CARGA Y PROCESAMIENTO DEL DATASET ---------------- #
df = pd.read_csv("smart_home.csv")

# Extraer promedios de columnas que contienen listas
def extract_mean(column):
    return df[column].apply(lambda x: np.mean(eval(x)) if pd.notna(x) else np.nan)

df["mean_voltage"] = extract_mean("voltages")
df["mean_current"] = extract_mean("currents")
df["mean_power"] = extract_mean("activePowers")
df["mean_pf"] = extract_mean("powerFactors")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hora"] = df["timestamp"].dt.hour
df["dia_semana"] = df["timestamp"].dt.day_name()
df["mes"] = df["timestamp"].dt.month

def obtener_estacion(mes):
    if mes in [12, 1, 2]:
        return "Verano"
    elif mes in [3, 4, 5]:
        return "Otoño"
    elif mes in [6, 7, 8]:
        return "Invierno"
    else:
        return "Primavera"

df["estacion"] = df["mes"].apply(obtener_estacion)
df_graph = df.dropna(subset=["mean_power", "appliance"])

df_graph["Dispositivo"] = df_graph["appliance"].str.strip().str.capitalize()


# Clasificación automática del consumo
umbral = df_graph["mean_power"].median()
df_graph["ConsumoAlto"] = (df_graph["mean_power"] > umbral).astype(int)

# Mapas fijos de codificación para el módulo de predicción
appliance_map = {"cell-phone": 0, "tv": 1, "refrigerator": 2}
brand_map = {"iphone-10": 0, "samsung-tv": 1, "lg-fridge": 2}
application_map = {"alipay": 0, "youtube": 1, "fridge-monitor": 2}

# ---------------- MENÚ PRINCIPAL ---------------- #
opcion = st.sidebar.radio("Selecciona un módulo", [
    "🔌 Sistema Inteligente",
    "📊 Análisis de Datos",
    "🤖 Modo Automático con Gráfico en Vivo",
    "🧠 Comparación de Modelos de ML"
])


if opcion == "🔌 Sistema Inteligente":
    mostrar_sistema(modelo_consumo, appliance_map, brand_map, application_map)

elif opcion == "📊 Análisis de Datos":
    mostrar_analisis(df_graph)

elif opcion == "🤖 Modo Automático con Gráfico en Vivo":
    mostrar_automatico(modelo_appliance, encoder_appliance, modelo_consumo)

elif opcion == "🧠 Comparación de Modelos de ML":
    mostrar_modelos()

