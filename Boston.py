# Assignment 1
# Boston House Price Prediction using Linear Regression

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Dataset
data = pd.read_csv('boston_train.csv')

# Step 3: Display First 5 Rows
print(data.head())

# Step 4: Dataset Information
print(data.info())

# Step 5: Remove ID Column
data.drop('ID', axis=1, inplace=True)

# Step 6: Check Correlation
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Step 7: Define Input and Output
X = data.drop('medv', axis=1)
y = data['medv']

# Step 8: Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)

# Step 9: Feature Scaling
# Scaling improves performance and keeps all features in same range
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 10: Import Linear Regression Model
from sklearn.linear_model import LinearRegression

# Create Linear Regression Model
model = LinearRegression()

# Train the Model
model.fit(X_train, y_train)

# Step 11: Predict Test Data
y_pred = model.predict(X_test)

# Step 12: Evaluate Model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print Evaluation Results
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

# Step 13: Actual vs Predicted Graph
plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")

plt.show()
