import pandas as pd
import csv 
import numpy as np
import random
import matplotlib.pyplot as plt
from filters import SignalProcessing

# Leer  archivos CSV
normal_signals_dt = pd.read_csv('new_normal_signal.csv')
abnormal_signals_dt = pd.read_csv('new_abnormal_signal.csv')

normal_signals_data = normal_signals_dt.to_numpy()
abnormal_signals_data = abnormal_signals_dt.to_numpy()

final_normal_signals = []
final_abnormal_signals = []

cutoff = 2*np.pi*60 #frecuencia de corte rad/s
fs = 2*np.pi*125 #sampling frequency rad/s 

signal_processor = SignalProcessing(fs)

for row in normal_signals_data:
    spectogram = signal_processor.image_to_array(row)
    final_normal_signals.append(spectogram)

for row in abnormal_signals_data:
    spectogram = signal_processor.image_to_array(row)
    final_abnormal_signals.append(spectogram)
print("Tamaño de señales normales:", len(final_normal_signals))
print("Tamaño de señales anormales:", len(final_abnormal_signals))
# Convertir las listas de espectrogramas en arrays 

while len(final_abnormal_signals) > len(final_normal_signals):
    random_index = random.randint(0, len(final_abnormal_signals) - 1)
    del final_abnormal_signals[random_index]
    
print("Tamaño de señales normales:", len(final_normal_signals))
print("Tamaño de señales anormales:", len(final_abnormal_signals))

# Nombres de los archivos CSV
nombre_archivo1 = 'normal_spectograms.csv'
nombre_archivo2 = 'abnormal_spectograms.csv'

# Escribir los datos del primer arreglo en el primer archivo CSV
with open(nombre_archivo1, mode='w', newline='') as archivo_csv1:
    escritor_csv1 = csv.writer(archivo_csv1)
    for row in final_normal_signals:
        escritor_csv1.writerow(row)

print(f'Se ha creado el archivo {nombre_archivo1} con los datos del primer arreglo.')

# Escribir los datos del segundo arreglo en el segundo archivo CSV
with open(nombre_archivo2, mode='w', newline='') as archivo_csv2:
    escritor_csv2 = csv.writer(archivo_csv2)
    for row in final_abnormal_signals:
        escritor_csv2.writerow(row)
