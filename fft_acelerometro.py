import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================
# CARGAR EL CSV
# =============================================

df = pd.read_csv("acelerometro_simulado.csv")

# Solo trabajamos con Z (eje vertical)
señal = df["Z"].values

# =============================================
# CONFIGURACIÓN DEL SENSOR
# Frecuencia de muestreo: 1 medición por segundo
# En el sensor real de Giancarlo puede ser mayor
# =============================================

fs = 1  # Hz — 1 muestra por segundo

# =============================================
# APLICAR FFT
# Convertir aceleración/tiempo → frecuencia/amplitud
# =============================================

n = len(señal)                    # Número total de muestras
fft_resultado = np.fft.fft(señal) # Aplica la FFT
amplitud = np.abs(fft_resultado)  # Extrae la amplitud
frecuencias = np.fft.fftfreq(n, d=1/fs)  # Calcula las frecuencias

# Solo la mitad positiva (la otra mitad es espejo)
mitad = n // 2
frecuencias = frecuencias[:mitad]
amplitud = amplitud[:mitad]

# =============================================
# GRAFICAR
# =============================================

plt.figure(figsize=(12, 5))
plt.plot(frecuencias, amplitud, color="#2E86AB")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.title("FFT — Señal del Acelerómetro (Eje Z)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fft_resultado.png", dpi=150)
plt.show()

print("FFT completada")
print(f"Frecuencia dominante: {frecuencias[np.argmax(amplitud)]:.4f} Hz")