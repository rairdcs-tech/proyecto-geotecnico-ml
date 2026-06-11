import pandas as pd

df = pd.read_csv("acelerometro_simulado.csv")

df['Hora'] = pd.to_datetime(df['Hora'], format='%H:%M:%S')

muestras_por_segundo = df.groupby('Hora').size()
print("Muestras por segundo:")
print(muestras_por_segundo)

duracion = (df['Hora'].max() - df['Hora'].min()).seconds
print(f"\nDuración total: {duracion} segundos")
print(f"Total de filas: {len(df)}")
print(f"Promedio de muestras/segundo: {len(df)/duracion:.2f}")