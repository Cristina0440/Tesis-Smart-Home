import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Cargar dataset
df = pd.read_csv("smart_home.csv")  # Asegúrate que esté en la misma carpeta

# Extraer valores promedio de listas
def extract_mean(column):
    return df[column].apply(lambda x: np.mean(eval(x)) if pd.notna(x) else np.nan)

df["mean_voltage"] = extract_mean("voltages")
df["mean_current"] = extract_mean("currents")
df["mean_power"] = extract_mean("activePowers")
df["mean_pf"] = extract_mean("powerFactors")

# Limpiar datos incompletos
df = df.dropna(subset=["mean_voltage", "mean_current", "mean_power", "mean_pf"])

# Crear variable objetivo: consumo alto o normal
umbral = df["mean_power"].median()
df["consumo_alto"] = (df["mean_power"] > umbral).astype(int)

# Entrenamiento solo con variables físicas reales
X = df[["mean_voltage", "mean_current", "mean_pf"]]
y = df["consumo_alto"]

# Separar datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir modelos a comparar
modelos = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "Árbol de Decisión": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Entrenar y evaluar modelos
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    print(f"\n🧠 {nombre}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.3f}")
    print("-" * 30)

# Guardar el mejor modelo (puedes elegir manualmente luego)
modelo_final = modelos["XGBoost"]
joblib.dump(modelo_final, "modelo_consumo.pkl")

# Mostrar columnas y balance para validar
print("\n🔍 Columnas usadas como entrada (X):")
print(X.columns)

print("\n📊 Balance de clases en 'consumo_alto':")
print(df["consumo_alto"].value_counts())
