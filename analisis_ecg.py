import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from filters import SignalProcessing
from time import sleep
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Leer  archivos CSV
normal_signals_dt = pd.read_csv('ptbdb_normal.csv')
abnormal_signals_dt = pd.read_csv('ptbdb_abnormal.csv')

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
print("Tamaño de señalesanormales:", len(final_abnormal_signals))
# Convertir las listas de espectrogramas en arrays 

while len(array2) > len(array1):
    random_index = random.randint(0, len(array2) - 1)
    del array2[random_index]
    
X_normal = np.array(final_normal_signals)
X_abnormal = np.array(final_abnormal_signals)

# Crear las etiquetas (0 normales y 1 anormales)
y_normal = np.zeros(len(X_normal))
y_abnormal = np.ones(len(X_abnormal))

# Concatenar los datos y las etiquetas
X = np.concatenate((X_normal, X_abnormal), axis=0)
y = np.concatenate((y_normal, y_abnormal), axis=0)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalizar 
X_train = X_train / 255
X_test = X_test / 255

# Imprimir las formas de los conjuntos de datos de entrenamiento y prueba
print("Forma de X_train:", X_train.shape)
print("Forma de y_train:", y_train.shape)
print("Forma de X_test:", X_test.shape)
print("Forma de y_test:", y_test.shape)
print("Forma de X_test:", X_val.shape)
print("Forma de y_test:", y_val.shape)

model = keras.Sequential([
    #Aplicar filtros altos en esta parte
    layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same',
                  input_shape=[64, 64, 1]),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same',),  
    layers.MaxPool2D(),
    layers.Dropout(0.2),

    # Block Two
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same',),  
    layers.MaxPool2D(),
    layers.Dropout(0.2),

    # Head Red normal, solo hacer filtos pequeños porque es una clasificación binaria
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),  
    layers.Dropout(0.5),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    #layers.Dense(1, activation='sigmoid'),
    layers.Dense(2, activation='softmax'),
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

history_model = model.fit(X_train, y_train,
                    epochs=10,
                    verbose=1,
                     validation_data=(X_val, y_val),
)

# Graficar la precisión de entrenamiento y validación por época
plt.plot(history_model.history['accuracy'])
plt.plot(history_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Graficar la pérdida de entrenamiento y validación por época
plt.plot(history_model.history['loss'])
plt.plot(history_model.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Guardar el modelo
model.save('CNN_ECG.h5')
