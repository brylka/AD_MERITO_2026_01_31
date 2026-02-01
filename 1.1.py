import datetime

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,1])

model = keras.Sequential([
    layers.Dense(2, activation='relu', input_shape=(2,)),
    layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

log_dir = 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorcloard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(X, y, epochs=500, verbose=1, callbacks=[tensorcloard_callback])

predyction = model.predict(X)
print("Predykcja:")
for i in range(len(X)):
    x = X[i]
    pred = predyction[i]
    print(f"{x[0]} OR {x[1]} = {pred[0]:.4f} -> {int(pred[0] > 0.5)}")

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.title('Funkcja strat w czasie treningu')
plt.xlabel('Epoki')
plt.ylabel('loss')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'])
plt.title('Dokładność w czasie treningu')
plt.xlabel('Epoki')
plt.ylabel('accuracy')
plt.grid(True)

plt.show()
