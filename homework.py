
# 1. Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report


# 2. Define X and Y arrays

x = np.arange(10).reshape(-1, 1)

y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


# 3. Display dataset

print("X values:")
print(x)

print("Y values:")
print(y)


# 4. Check classification type

classes = np.unique(y)

print("Classes:", classes)

if len(classes) == 2:
    print("Binary Classification")
else:
    print("Multi-class Classification")


# 5. Split dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)


# 6. Create Logistic Regression model

model = LogisticRegression()

model.fit(X_train, y_train)


# 7. Make predictions

y_pred = model.predict(X_test)

print("Actual values:", y_test)
print("Predicted values:", y_pred)


# 8. Check model accuracy

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# 9. Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# 10. Classification Report

print(classification_report(y_test, y_pred))


# 11. Visualize the results

plt.scatter(x, y, color='blue', label='Actual Data')

plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Logistic Regression Classification")

plt.legend()
plt.show()
