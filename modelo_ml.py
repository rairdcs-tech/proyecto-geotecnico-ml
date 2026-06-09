import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# =============================================
# CARGAR DATASET
# =============================================

df = pd.read_csv("dataset_ml.csv")

print("=" * 55)
print("  MODELO ML — PREDICCIÓN DE % COMPACTACIÓN")
print("=" * 55)
print(f"\nDataset cargado: {len(df)} ensayos")

# =============================================
# SEPARAR VARIABLES
# X = lo que el sensor mide (armónicos)
# y = lo que queremos predecir (% compactación)
# =============================================

X = df.drop(columns=["compactacion_pct"])
y = df["compactacion_pct"]

print(f"Variables de entrada: {list(X.columns)}")
print(f"Variable objetivo: compactacion_pct")

# =============================================
# DIVIDIR EN ENTRENAMIENTO Y PRUEBA
# 80% entrena, 20% evalúa
# Analogía: 80% de puntos de control para
# calibrar, 20% para verificar
# =============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\nEntrenamiento: {len(X_train)} ensayos")
print(f"Prueba:        {len(X_test)} ensayos")

# =============================================
# ENTRENAR MODELOS
# =============================================

# Modelo simple de referencia
modelo_lr = LinearRegression()
modelo_lr.fit(X_train, y_train)
y_pred_lr = modelo_lr.predict(X_test)

# Modelo principal
modelo_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=3,
    random_state=42
)
modelo_rf.fit(X_train, y_train)
y_pred_rf = modelo_rf.predict(X_test)

# =============================================
# EVALUAR
# =============================================

print("\n" + "=" * 55)
print("  RESULTADOS")
print("=" * 55)

for nombre, y_pred in [("Regresión Lineal", y_pred_lr),
                        ("Random Forest",   y_pred_rf)]:
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n  [{nombre}]")
    print(f"  R²  = {r2:.4f}  {'✓ BUENO' if r2 > 0.85 else '✗ INSUFICIENTE'}")
    print(f"  MAE = {mae:.3f}% {'✓ ACEPTABLE' if mae < 2.0 else '✗ REVISAR'}")

# =============================================
# IMPORTANCIA DE VARIABLES
# =============================================

print("\n" + "=" * 55)
print("  VARIABLES MÁS IMPORTANTES (Random Forest)")
print("=" * 55)

importancias = pd.Series(
    modelo_rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

for var, imp in importancias.items():
    barra = "█" * int(imp * 50)
    print(f"  {var:<12} {imp:.4f}  {barra}")

# =============================================
# GRÁFICO
# =============================================

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.7, color="#2E86AB", s=40)
lim = [y_test.min() - 1, y_test.max() + 1]
plt.plot(lim, lim, "r--", linewidth=2, label="Predicción perfecta")
plt.xlabel("% Compactación Real")
plt.ylabel("% Compactación Predicha")
plt.title("Random Forest — Predicciones vs Real")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("resultado_modelo.png", dpi=150)
plt.show()

print("\n✓ Gráfico guardado como resultado_modelo.png")