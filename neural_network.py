import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, classes, normalize=False, cmap='Blues'):
    """
    This function plots a confusion matrix using Seaborn.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # Create a figure and set the plot size
    plt.figure(figsize=(len(classes)+1,len(classes)+1))

    # Create the heatmap using Seaborn
    sns.heatmap(cm, annot=True, cmap=cmap, xticklabels=classes, yticklabels=classes)

    # Set the plot labels
    plt.title('Confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
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
X = df.iloc[:, :-1] # all columns before the last --> data
y = df.iloc[:, -1] # the last column --> result

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=1000, validation_split=0.2, verbose=0)

# Predict on the test set
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

print("=" * 32)
print(f'Loss: {history.history["loss"][-1]}')
loss=history.history['loss']
plt.plot(loss)

# Evaluate the model
print("=" * 32)
print('Accuracy:', accuracy_score(y_test, y_pred))
print("=" * 32)
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print("=" * 32)

# Example features
example_features = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
# example_features = [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]

# Reshape example features into a 2D numpy array
example_features_array = np.array(example_features).reshape(1, -1)

# Predict the class (malicious or not) of the example URL
prediction = model.predict(example_features_array)
print(f"Prediction: {prediction}")
print("=" * 32)
if prediction > 0.5:
    print("\n\n\nThe example URL is predicted to be malicious.")
else:
    print("\n\n\nThe example URL is predicted to be non-malicious.")

# Plot the confusion matrix using the function
# plot_confusion_matrix(cm, class_names, normalize=True, cmap='Blues')
