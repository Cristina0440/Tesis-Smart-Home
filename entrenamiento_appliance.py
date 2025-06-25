import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Cargar dataset
df = pd.read_csv("smart_home.csv")

# Extraer valores promedio
def extract_mean(column):
    return df[column].apply(lambda x: np.mean(eval(x)) if pd.notna(x) else np.nan)

df["mean_voltage"] = extract_mean("voltages")
df["mean_current"] = extract_mean("currents")
df["mean_power"] = extract_mean("activePowers")
df["mean_pf"] = extract_mean("powerFactors")

# Eliminar filas incompletas
df = df.dropna(subset=["mean_voltage", "mean_current", "mean_power", "mean_pf", "appliance"])

# Codificar la variable objetivo (appliance)
label_encoder = LabelEncoder()
df["appliance_encoded"] = label_encoder.fit_transform(df["appliance"])

# Variables de entrada y salida
X = df[["mean_voltage", "mean_current", "mean_power", "mean_pf"]]
y = df["appliance_encoded"]

# Separar en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir modelos
modelos = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "Árbol de Decisión": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Entrenar y evaluar
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    print(f"\n🔌 {nombre}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1-score:  {f1_score(y_test, y_pred, average='weighted'):.3f}")
    print("-" * 30)

# Guardar el mejor modelo (puedes elegir el que quieras)
modelo_final = modelos["XGBoost"]
joblib.dump(modelo_final, "modelo_appliance.pkl")

# Guardar también el encoder de clases (para decodificar luego)
joblib.dump(label_encoder, "encoder_appliance.pkl")
