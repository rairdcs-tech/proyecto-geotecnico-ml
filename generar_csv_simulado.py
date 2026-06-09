import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Semilla para reproducibilidad
np.random.seed(42)

# Configuración
N_MUESTRAS = 500
hora_inicio = datetime.strptime("01:23:00", "%H:%M:%S")

registros = []

for i in range(N_MUESTRAS):
    # Hora con incremento de 1 segundo
    hora = (hora_inicio + timedelta(seconds=i)).strftime("%H:%M:%S")
    
    # X e Y: vibración lateral pequeña (ruido alrededor de 0)
    x = round(np.random.normal(0, 0.03), 7)
    y = round(np.random.normal(0, 0.008), 7)
    
    # Z: eje vertical, alrededor de 1g (gravedad) con vibración del motor
    z = round(np.random.normal(1.2, 0.08), 7)
    
    registros.append({"Hora": hora, "X": x, "Y": y, "Z": z})

df = pd.DataFrame(registros)
df.to_csv("acelerometro_simulado.csv", index=False)

print(f"CSV generado: {len(df)} filas")
print(df.head())