import pandas as pd
import numpy as np

# =============================================
# CARGAR EL CSV
# =============================================

df = pd.read_csv("acelerometro_simulado.csv")
señal = df["Z"].values

# =============================================
# APLICAR FFT
# =============================================

fs = 1  # Frecuencia de muestreo: 1 muestra/segundo
n = len(señal)
fft_resultado = np.fft.fft(señal)
amplitud = np.abs(fft_resultado)[:n//2]
frecuencias = np.fft.fftfreq(n, d=1/fs)[:n//2]

# =============================================
# IGNORAR COMPONENTE DC (frecuencia 0)
# Es la gravedad, no aporta información útil
# =============================================

amplitud[0] = 0

# =============================================
# EXTRAER LOS 5 ARMÓNICOS PRINCIPALES
# Analógía: como quedarte con las 5 frecuencias
# que más energía tienen en la señal del suelo
# Son las "huellas digitales" de la compactación
# =============================================

N_ARMONICOS = 5
indices_top = np.argsort(amplitud)[::-1][:N_ARMONICOS]

armonicos = []
for i, idx in enumerate(indices_top):
    armonicos.append({
        "armonico":   i + 1,
        "frecuencia": round(frecuencias[idx], 6),
        "amplitud":   round(amplitud[idx], 6)
    })

df_armonicos = pd.DataFrame(armonicos)

# =============================================
# MOSTRAR RESULTADOS
# =============================================

print("=" * 45)
print("  ARMÓNICOS EXTRAÍDOS DE LA SEÑAL")
print("=" * 45)
print(df_armonicos.to_string(index=False))

# =============================================
# GUARDAR COMO CSV
# Este archivo será la entrada del modelo ML
# =============================================

df_armonicos.to_csv("armonicos_extraidos.csv", index=False)
print("\n✓ Guardado en armonicos_extraidos.csv")