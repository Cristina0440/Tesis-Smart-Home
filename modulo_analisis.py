import streamlit as st
import plotly.express as px
import pandas as pd

def mostrar_analisis(df_graph):
    st.header("📈 Análisis del Consumo Eléctrico")

    # Mostrar dispositivos únicos
    st.write("Dispositivos detectados:", df_graph["Dispositivo"].unique())

    # Gráfico 1: Consumo promedio por tipo de dispositivo
    st.subheader("🔌 Consumo promedio por tipo de dispositivo")
    consumo_promedio = df_graph.groupby("Dispositivo")["mean_power"].mean().reset_index()
    consumo_promedio = consumo_promedio.sort_values(by="mean_power", ascending=False)

    fig1 = px.bar(
        consumo_promedio,
        x="Dispositivo",
        y="mean_power",
        color="Dispositivo",
        labels={"mean_power": "Potencia Promedio (W)"},
        title="Promedio de Consumo Eléctrico por Dispositivo"
    )
    fig1.update_layout(
        yaxis=dict(range=[0, consumo_promedio["mean_power"].max() * 1.3]),
        height=600
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Gráfico 2: Distribución del consumo alto vs normal
    st.subheader("📊 Distribución del consumo: ¿Alto o Normal?")
    fig2 = px.histogram(
        df_graph,
        x="mean_power",
        color="ConsumoAlto",
        nbins=30,
        barmode="overlay",
        labels={"mean_power": "Potencia (W)", "ConsumoAlto": "¿Alto consumo?"},
        title="Distribución de Potencia según Nivel de Consumo"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Gráfico 3: Distribución por marca
    st.subheader("🏷️ Distribución del consumo por marca")
    marca = st.selectbox("Selecciona una marca", df_graph["brand"].dropna().unique())
    df_filtrado = df_graph[df_graph["brand"] == marca]
    fig3 = px.box(
        df_filtrado,
        x="Dispositivo",
        y="mean_power",
        color="ConsumoAlto",
        labels={"mean_power": "Potencia (W)", "Dispositivo": "Dispositivo"},
        title=f"Distribución de Potencia por Dispositivo en la marca {marca}"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Gráfico 4: Relación entre corriente y potencia
    st.subheader("🔁 Relación entre Corriente y Potencia")
    fig4 = px.scatter(
        df_graph,
        x="mean_current",
        y="mean_power",
        color="Dispositivo",
        size="mean_pf",
        title="Relación entre Corriente y Potencia (Tamaño = Factor de Potencia)",
        labels={"mean_current": "Corriente (A)", "mean_power": "Potencia (W)"}
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Gráfico 5: Factor de potencia por dispositivo
    st.subheader("📐 Eficiencia eléctrica promedio (Factor de Potencia)")
    fp_promedio = df_graph.groupby("Dispositivo")["mean_pf"].mean().reset_index()
    fig5 = px.bar(
        fp_promedio,
        x="Dispositivo",
        y="mean_pf",
        color="Dispositivo",
        labels={"mean_pf": "Factor de Potencia"},
        title="Factor de Potencia Promedio por Dispositivo"
    )
    st.plotly_chart(fig5, use_container_width=True)

    # Gráfico 6: Mapa de correlaciones
    st.subheader("🧠 Mapa de correlaciones entre variables")
    corr = df_graph[["mean_voltage", "mean_current", "mean_power", "mean_pf"]].corr()
    fig6 = px.imshow(corr, text_auto=True, aspect="auto", title="Mapa de Correlación")
    st.plotly_chart(fig6, use_container_width=True)

    # Gráfico 7: Consumo por hora del día
    st.subheader("🕒 Consumo promedio por hora del día")
    fig_hora = px.line(
        df_graph.groupby("hora")["mean_power"].mean().reset_index(),
        x="hora", y="mean_power",
        markers=True,
        title="Promedio de Consumo Eléctrico por Hora del Día",
        labels={"mean_power": "Potencia Promedio (W)", "hora": "Hora del Día"}
    )
    st.plotly_chart(fig_hora, use_container_width=True)

    # Gráfico 8: Consumo por día de la semana
    st.subheader("📅 Consumo promedio por día de la semana")
    dias_ordenados = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df_dia = df_graph.groupby("dia_semana")["mean_power"].mean().reindex(dias_ordenados).reset_index()

    fig_dia = px.bar(
        df_dia, x="dia_semana", y="mean_power", color="dia_semana",
        title="Consumo Promedio por Día de la Semana",
        labels={"mean_power": "Potencia Promedio (W)", "dia_semana": "Día"}
    )
    st.plotly_chart(fig_dia, use_container_width=True)

    # Gráfico 9: Consumo por estación
    st.subheader("🍂 Consumo promedio por estación del año")
    fig_estacion = px.box(
        df_graph,
        x="estacion",
        y="mean_power",
        color="estacion",
        title="Distribución de Consumo por Estación del Año",
        labels={"mean_power": "Potencia (W)", "estacion": "Estación"}
    )
    st.plotly_chart(fig_estacion, use_container_width=True)



