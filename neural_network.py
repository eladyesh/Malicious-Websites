import numpy as np
import pandas as pd
import tensorflow.keras as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse


def plot_confusion_matrix(cm):
    """
    This function takes in true labels and predicted labels as NumPy arrays and plots the corresponding confusion matrix using Seaborn.

    Parameters:
        y_true (numpy.ndarray): True labels represented as a NumPy array.
        y_pred (numpy.ndarray): Predicted labels represented as a NumPy array.

    Returns:
        None
    """
    # Set the figure size
    plt.figure(figsize=(8, 6))

    # Create a heatmap using Seaborn
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)

    # Add labels to the x and y axes
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Show the plot
    plt.show()


# Define class labels
class_names = ['Class 0', 'Class 1']

# Load the dataset
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff'
# df = pd.read_csv(url, header=None, comment='@', sep=',')

# Load the Excel file into a pandas DataFrame, ignoring headers
df = pd.read_excel("train_website.xlsx", header=None, skiprows=1)
pd.DataFrame(df)

# Split the dataset into features and target
X = np.array(df.iloc[:, :-1])  # all columns before the last --> data
Y = np.array(df.iloc[:, -1])  # the last column --> result

y1 = Y*0
y1[np.argwhere(Y == 1)] = 1
y2 = Y*0
y2[np.argwhere(Y == -1)] = 1
y_new = np.array([y1, y2]).T

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_new, test_size=0.2, random_state=1)

# Build the neural network model
model = tf.models.Sequential()
# model.add(tf.layers.Dense(units=30, activation='relu'))  # input - 30 attributes
# model.add(tf.layers.Dense(units=2, activation='tanh'))  # output - 1 attribute
# model.add(tf.layers.Dense(units=2, activation='softmax'))  # output - 1 attribute

model.add(tf.layers.Dense(units=2, activation='relu'))  # output - 1 attribute
model.add(tf.layers.Dense(units=2, activation='softmax'))  # output - 1 attribute

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=1000, validation_split=0.2, verbose=0)

# Predict on the test set
y_pred = np.round(model.predict(X_test))
# y_pred = (y_pred > 0.5).astype(int)

# Example features
example_features = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# example_features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Reshape example features into a 2D numpy array
example_features_array = np.array(example_features).reshape(1, -1)
state = np.array([0, 1])

# Predict the class (malicious or not) of the example URL
prediction = model.predict(example_features_array)
prediction = np.dot(prediction, state)[0]
print(f"Prediction: {prediction}")
print("=" * 32)
if int(prediction) == 0:
    print("The example URL is predicted to be non-malicious.")
elif int(prediction) == 1:
    print("The example URL is predicted to be malicious.")

# Evaluate the model
cm = confusion_matrix(np.dot(y_test, state), np.dot(y_pred, state))
print("=" * 32)
print('Confusion Matrix:\n', cm)
print("=" * 32)
plot_confusion_matrix(cm)

print("=" * 32)
print(f'Loss: {history.history["loss"][-1]}')
loss = history.history['loss']
plt.plot(loss)

print("=" * 32)
print('mse:', mse(y_test, y_pred))
