# Assignment 4
# Convolutional Neural Network (CNN)
# Fashion Clothing Classification using MNIST Fashion Dataset

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Dataset
data = pd.read_csv('fashion-mnist.csv')

# Step 3: Display First 5 Rows
print(data.head())

# Step 4: Dataset Information
print(data.info())

# Step 5: Define Input and Output
# First column = label
# Remaining columns = pixel values

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Step 6: Normalize Pixel Values
# Convert values from 0-255 to 0-1

X = X / 255.0

# Step 7: Reshape Data for CNN
# CNN expects 4D input:
# (samples, height, width, channels)

X = X.reshape(-1, 28, 28, 1)

# Step 8: Train Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)

# Step 9: Build CNN Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

model = Sequential()

# First Convolution Layer

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(28,28,1)
    )
)

# Pooling Layer

model.add(MaxPooling2D(pool_size=(2,2)))

# Second Convolution Layer

model.add(Conv2D(64, (3,3), activation='relu'))

# Second Pooling Layer

model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten Layer

model.add(Flatten())

# Hidden Layer

model.add(Dense(128, activation='relu'))

# Output Layer
# 10 classes for fashion categories

model.add(Dense(10, activation='softmax'))

# Step 10: Compile Model

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 11: Train Model

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# Step 12: Predict Test Data

y_pred = model.predict(X_test)

# Convert probabilities into class labels

y_pred_classes = np.argmax(y_pred, axis=1)

# Step 13: Evaluate Model

loss, accuracy = model.evaluate(X_test, y_test)

print("Loss:", loss)
print("Accuracy:", accuracy)

# Step 14: Actual vs Predicted Values

print("\nActual Values:")
print(y_test[:10])

print("\nPredicted Values:")
print(y_pred_classes[:10])

# Step 15: Plot Accuracy Graph

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'])

plt.show()

# Step 16: Display Sample Prediction

plt.imshow(X_test[0].reshape(28,28), cmap='gray')

plt.title(
    f"Predicted: {y_pred_classes[0]} | Actual: {y_test[0]}"
)

plt.show()
