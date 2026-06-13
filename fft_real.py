import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================
# FFT VERSIÓN 2.0
# Cambios respecto a v1.0:
# - Amplitud convertida de g a m/s²
# - Componente DC eliminada antes de la FFT
# =============================================

# PASO 1: Cargar CSV real del sensor
df = pd.read_csv("acelerometro_simulado.csv")
df['Hora'] = pd.to_datetime(df['Hora'], format='%H:%M:%S')

# PASO 2: Filtrar parte estable del sensor
# Los primeros segundos el sensor estaba irregular
hora_inicio = pd.to_datetime("01:23:17", format='%H:%M:%S')
df = df[df['Hora'] >= hora_inicio].reset_index(drop=True)
print(f"Filas después del filtro: {len(df)}")

# PASO 3: Extraer eje Z y convertir g → m/s²
# 1g = 9.8 m/s² (unidades del sistema internacional)
z = df['Z'].values * 9.8

# PASO 4: Eliminar componente DC (la gravedad constante)
# Sin esto el pico de gravedad aplasta todo el gráfico
z = z - np.mean(z)

# PASO 5: Frecuencia de muestreo real
fs = 4.14  # muestras por segundo (medido empíricamente)

# PASO 6: Aplicar FFT
n = len(z)
fft_vals = np.fft.rfft(z)           # FFT solo parte positiva
fft_amp  = np.abs(fft_vals) / n     # amplitud normalizada
freqs    = np.fft.rfftfreq(n, d=1/fs)  # eje de frecuencias en Hz

# PASO 7: Graficar
plt.figure(figsize=(10, 5))
plt.plot(freqs, fft_amp, color="#2E86AB")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud (m/s²)")
plt.title("FFT v2.0 — Eje Z — Datos Reales del Sensor")
plt.grid(True)
plt.tight_layout()
plt.savefig("fft_v2.png")
plt.show()
print("Gráfico guardado como fft_v2.png")