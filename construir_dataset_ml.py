import pandas as pd
import numpy as np

# =============================================
# CONFIGURACIÓN
# =============================================

np.random.seed(42)
N_ENSAYOS = 200  # Número de ensayos simulados

# =============================================
# FUNCIÓN: Simular una medición del sensor
# para un nivel de compactación dado
#
# Analogía: suelo más compacto = motor rebota
# más fuerte = frecuencias más altas y
# amplitudes mayores en la FFT
# =============================================

def simular_medicion(compactacion, fs=1, n=500):
    # Suelo más compacto → mayor frecuencia dominante
    freq_dominante = 0.05 + (compactacion - 85) * 0.003
    
    # Suelo más compacto → mayor amplitud
    amplitud_base = 1.0 + (compactacion - 85) * 0.08
    
    # Generar señal con esa frecuencia + ruido
    t = np.arange(n) / fs
    señal = (amplitud_base * np.sin(2 * np.pi * freq_dominante * t)
             + np.random.normal(0, 0.1, n) + 1.2)
    
    # Aplicar FFT
    fft_resultado = np.fft.fft(señal)
    amplitud_fft = np.abs(fft_resultado)[:n//2]
    frecuencias = np.fft.fftfreq(n, d=1/fs)[:n//2]
    
    # Ignorar componente DC
    amplitud_fft[0] = 0
    
    # Extraer 5 armónicos principales
    indices_top = np.argsort(amplitud_fft)[::-1][:5]
    
    caracteristicas = {}
    for i, idx in enumerate(indices_top):
        caracteristicas[f"freq_{i+1}"] = round(frecuencias[idx], 6)
        caracteristicas[f"amp_{i+1}"] = round(amplitud_fft[idx], 6)
    
    return caracteristicas

# =============================================
# GENERAR 200 ENSAYOS
# Compactación entre 88% y 102%
# =============================================

registros = []

for i in range(N_ENSAYOS):
    # Compactación aleatoria entre 88% y 102%
    compactacion = round(np.random.uniform(88, 102), 2)
    
    # Simular medición del sensor para esa compactación
    caracteristicas = simular_medicion(compactacion)
    
    # Agregar compactación como variable objetivo
    caracteristicas["compactacion_pct"] = compactacion
    
    registros.append(caracteristicas)

df = pd.DataFrame(registros)

# =============================================
# MOSTRAR Y GUARDAR
# =============================================

print("=" * 55)
print("  DATASET DE ENTRENAMIENTO — ML")
print("=" * 55)
print(f"\nEnsayos generados: {len(df)}")
print(f"Variables de entrada: {len(df.columns) - 1}")
print(f"\nPrimeras 5 filas:")
print(df.head().to_string(index=False))
print(f"\nRango compactación:")
print(f"  Mínimo: {df['compactacion_pct'].min()}%")
print(f"  Máximo: {df['compactacion_pct'].max()}%")

df.to_csv("dataset_ml.csv", index=False)
print("\n✓ Guardado en dataset_ml.csv")