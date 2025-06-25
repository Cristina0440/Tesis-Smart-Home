
import streamlit as st
import pandas as pd
import random

def mostrar_sistema(modelo_consumo, appliance_map, brand_map, application_map):
    st.header("🧠 Clasificación manual o simulada del consumo")

    modo = st.radio("Modo de entrada", ["Simulación aleatoria", "Ingreso manual"])

    if modo == "Simulación aleatoria":
        if st.button("Simular lectura"):
            mean_voltage = round(random.uniform(210, 250), 2)
            mean_current = round(random.uniform(0.5, 5.0), 2)
            mean_pf = round(random.uniform(0.5, 1.0), 2)
            appliance = random.choice(list(appliance_map.keys()))
            brand = random.choice(list(brand_map.keys()))
            application = random.choice(list(application_map.keys()))

            st.write("### Datos simulados")
            st.write(f"Voltaje: {mean_voltage} V")
            st.write(f"Corriente: {mean_current} A")
            st.write(f"Factor potencia: {mean_pf}")
            st.write(f"Dispositivo: {appliance}")
            st.write(f"Marca: {brand}")
            st.write(f"Aplicación: {application}")

            input_df = pd.DataFrame([[
                mean_voltage, mean_current, mean_pf,
                application_map[application],
                brand_map[brand],
                appliance_map[appliance]
            ]], columns=[
                "mean_voltage", "mean_current", "mean_pf",
                "application_encoded", "brand_encoded", "appliance_encoded"
            ])

            resultado = modelo_consumo.predict(input_df)[0]
            st.subheader("Resultado")
            if resultado == 1:
                st.error("⚠️ Alto consumo detectado.")
            else:
                st.success("✅ Consumo dentro de lo normal.")

    else:
        st.subheader("Ingresa los datos")
        mean_voltage = st.slider("Voltaje (V)", 210.0, 250.0, 230.0)
        mean_current = st.slider("Corriente (A)", 0.0, 5.0, 2.5)
        mean_pf = st.slider("Factor de potencia", 0.0, 1.0, 0.8)
        appliance = st.selectbox("Dispositivo", list(appliance_map.keys()))
        brand = st.selectbox("Marca", list(brand_map.keys()))
        application = st.selectbox("Aplicación", list(application_map.keys()))

        if st.button("Clasificar consumo"):
            input_df = pd.DataFrame([[
                mean_voltage, mean_current, mean_pf,
                application_map[application],
                brand_map[brand],
                appliance_map[appliance]
            ]], columns=[
                "mean_voltage", "mean_current", "mean_pf",
                "application_encoded", "brand_encoded", "appliance_encoded"
            ])

            resultado = modelo_consumo.predict(input_df)[0]
            st.subheader("Resultado")
            if resultado == 1:
                st.error("⚠️ Alto consumo detectado.")
            else:
                st.success("✅ Consumo dentro de lo normal.")
