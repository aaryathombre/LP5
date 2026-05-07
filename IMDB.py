# Assignment 3
# Binary Classification using Deep Neural Network
# IMDB Movie Review Sentiment Classification

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Load Dataset
data = pd.read_csv('IMDB_Dataset.csv')

# Step 3: Display First 5 Rows
print(data.head())

# Step 4: Dataset Information
print(data.info())

# Step 5: Define Input and Output

X = data['review']
y = data['sentiment']

# Step 6: Convert Labels into Numerical Form
# positive = 1
# negative = 0

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y = encoder.fit_transform(y)

# Step 7: Convert Text into Numerical Data

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X).toarray()

# Step 8: Train Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)

# Step 9: Build Deep Neural Network

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

# Input Layer + Hidden Layers

model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))

# Output Layer
# sigmoid for binary classification

model.add(Dense(1, activation='sigmoid'))

# Step 10: Compile Model

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 11: Train Model

history = model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=32,
    validation_split=0.1
)

# Step 12: Predict Test Data

y_pred = model.predict(X_test)

# Convert probabilities into 0 and 1

y_pred_classes = (y_pred > 0.5).astype(int)

# Step 13: Evaluate Model

loss, accuracy = model.evaluate(X_test, y_test)

print("Loss:", loss)
print("Accuracy:", accuracy)

# Step 14: Actual vs Predicted Values

print("\nActual Values:")
print(y_test[:10])

print("\nPredicted Values:")
print(y_pred_classes[:10].flatten())

# Step 15: Plot Accuracy Graph

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend(['Train', 'Validation'])

plt.show()
