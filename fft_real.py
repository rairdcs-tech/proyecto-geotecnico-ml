import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carga datos reales
df = pd.read_csv("acelerometro_simulado.csv")
df['Hora'] = pd.to_datetime(df['Hora'], format='%H:%M:%S')

# Usamos solo filas desde el segundo 17 en adelante (parte estable)
hora_inicio = pd.to_datetime("01:23:17", format='%H:%M:%S')
df = df[df['Hora'] >= hora_inicio].reset_index(drop=True)

print(f"Filas después del filtro: {len(df)}")

# Señal: solo eje Z
z = df['Z'].values

# Frecuencia de muestreo real
fs = 4.14  # muestras por segundo

# Calcular FFT
n = len(z)                        # número de muestras
fft_vals = np.fft.rfft(z)         # FFT solo parte positiva
fft_amp = np.abs(fft_vals) / n    # amplitud normalizada
freqs = np.fft.rfftfreq(n, d=1/fs) # eje de frecuencias en Hz

# Graficar
plt.figure(figsize=(10, 5))
plt.plot(freqs, fft_amp)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud (g)")
plt.title("FFT - Eje Z - Datos Reales del Sensor")
plt.grid(True)
plt.tight_layout()
plt.savefig("fft_real.png")
plt.show()
print("Gráfico guardado como fft_real.png")