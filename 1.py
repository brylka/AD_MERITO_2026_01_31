# 0 0 0
# 0 1 0
# 1 0 0
# 1 1 1

import numpy as np
from tensorflow import keras

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,0,0,1])

model = keras.Sequential([
    keras.layers.Dense(1, activation='sigmoid', input_shape=(2,))
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=100, verbose=1)

predyction = model.predict(X)
print("Predykcja:")
for i in range(len(X)):
    x = X[i]
    pred = predyction[i]
    print(f"{x[0]} AND {x[1]} = {pred[0]:.4f} -> {int(pred[0] > 0.5)}")