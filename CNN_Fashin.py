# CNN using Fashion MNIST Dataset

# Step 1: Import Libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load Fashion MNIST Dataset

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Step 3: Display Dataset Shape

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Step 4: Normalize Data
# Convert values from 0-255 to 0-1

X_train = X_train / 255.0
X_test = X_test / 255.0

# Step 5: Reshape Data for CNN
# CNN expects 4D input

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Step 6: Build CNN Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

model = Sequential()

# Convolution Layer

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(28,28,1)
    )
)

# Pooling Layer

model.add(MaxPooling2D((2,2)))

# Flatten Layer

model.add(Flatten())

# Hidden Layer

model.add(Dense(64, activation='relu'))

# Output Layer
# 10 classes in Fashion MNIST

model.add(Dense(10, activation='softmax'))

# Step 7: Compile Model

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 8: Train Model

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# Step 9: Evaluate Model

loss, accuracy = model.evaluate(X_test, y_test)

print("Loss:", loss)
print("Accuracy:", accuracy)

# Step 10: Predict Test Data

y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)

# Step 11: Display Sample Prediction

plt.imshow(X_test[0].reshape(28,28), cmap='gray')

plt.title(
    f"Predicted: {y_pred_classes[0]} | Actual: {y_test[0]}"
)

plt.show()
