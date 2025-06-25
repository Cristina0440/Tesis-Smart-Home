import streamlit as st
import pandas as pd
import plotly.express as px
import random
import time

def mostrar_automatico(modelo_appliance, encoder_appliance, modelo_consumo):
    st.header("📡 Modo Automático: Monitoreo en Tiempo Real")

    st.markdown("El sistema simula automáticamente una nueva lectura cada 2 segundos. Observa cómo cambia el consumo de energía en tiempo real.")

    if "lecturas" not in st.session_state:
        st.session_state.lecturas = []

    iniciar = st.button("Iniciar monitoreo")

    if iniciar:
        placeholder = st.empty()  # contenedor dinámico

        for _ in range(15):  # número de lecturas simuladas
            mean_voltage = round(random.uniform(210, 250), 2)
            mean_current = round(random.uniform(0.2, 4.5), 2)
            mean_pf = round(random.uniform(0.5, 1.0), 2)
            mean_power = round(mean_voltage * mean_current * mean_pf, 2)
            hora_actual = pd.Timestamp.now().strftime("%H:%M:%S")

            input_appliance = pd.DataFrame([[
                mean_voltage, mean_current, mean_power, mean_pf
            ]], columns=["mean_voltage", "mean_current", "mean_power", "mean_pf"])

            pred_appliance = modelo_appliance.predict(input_appliance)[0]
            appliance_name = encoder_appliance.inverse_transform([pred_appliance])[0]

            input_consumo = pd.DataFrame([[
    mean_voltage, mean_current, mean_pf
]], columns=["mean_voltage", "mean_current", "mean_pf"])

            resultado = modelo_consumo.predict(input_consumo)[0]
            nivel = "Alto" if resultado == 1 else "Normal"

            st.session_state.lecturas.append({
                "Hora": hora_actual,
                "Voltaje": mean_voltage,
                "Corriente": mean_current,
                "Potencia": mean_power,
                "PF": mean_pf,
                "Dispositivo": appliance_name,
                "Consumo": nivel
            })

            with placeholder.container():
                df_vivo = pd.DataFrame(st.session_state.lecturas)
                df_plot = df_vivo.tail(15)

                st.subheader("🔁 Gráfico de Potencia en tiempo real")
                fig = px.line(df_plot, x="Hora", y="Potencia", markers=True)
                fig.update_layout(xaxis=dict(tickmode='linear'), yaxis_title="Potencia (W)")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("📋 Últimas lecturas")
                st.dataframe(df_vivo.tail(15), use_container_width=True)

            time.sleep(2)

    # Mostrar tabla explicativa al final (fuera del bucle)
    tabla_dispositivos = pd.DataFrame({
        "Dispositivo": [
            "Cell-phone", "Laptop", "TV", "Refrigerador",
            "Microondas", "Lavadora", "Aire Acondicionado"
        ],
        "Corriente (A)": [
            "0.1 – 0.5", "0.5 – 1.0", "1.0 – 2.0", "2.5 – 4.0",
            "4.0 – 7.0", "3.0 – 5.0", "3.5 – 6.5"
        ],
        "Potencia (W)": [
            "1 – 10", "30 – 60", "50 – 100", "100 – 200",
            "700 – 1200", "300 – 800", "600 – 1200"
        ],
        "Factor de Potencia": [
            "0.9 – 1.0", "0.85 – 0.95", "0.7 – 0.9", "0.5 – 0.95",
            "0.6 – 0.9", "0.4 – 0.85", "0.5 – 0.8"
        ],
        "Descripción": [
            "Consumo bajo, eficiente",
            "Moderado, constante",
            "Moderado, prolongado",
            "Cíclico, picos de corriente",
            "Breve pero muy alto",
            "Alto, con ciclos",
            "Alto sostenido"
        ]
    })

    st.markdown("### 🧠 ¿Cómo el modelo detecta qué dispositivo es?")
    st.write("El modelo fue entrenado con variables como corriente, potencia y eficiencia (factor de potencia), que crean una 'firma eléctrica' distinta para cada tipo de aparato.")
    st.dataframe(tabla_dispositivos)