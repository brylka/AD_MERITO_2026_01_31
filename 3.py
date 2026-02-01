import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(f"Dane treningowe: {x_train.shape}")
print(f"Dane testowe: {x_test.shape}")

model = keras.Sequential([
    layers.Flatten(input_shape=(28,28)),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),

    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


model.summary()

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

predictions = model.predict(x_test[:5])
for i in range(5):
    prediction = np.argmax(predictions[i])
    actual = y_test[i]
    pred = predictions[i][prediction]
    print(f"Obraz {i}: przewidziano {prediction} (pewność: {pred}, prawdziwa: {actual}")
