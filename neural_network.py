import random

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow.keras as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error as mse
from Data import CollectData

URL = 'https://www.amazon.com/'

urls = [
    "https://www.nike.com",
    "https://www.hulu.com",
    "https://www.wikipedia.org",
    "https://www.airbnb.com",
    "https://www.spotify.com",
    "https://www.khanacademy.org",
    "https://www.codecademy.com",
    "https://www.dropbox.com",
    "https://www.udemy.com",
    "https://www.salesforce.com",
]


def plot_confusion_matrix(cm):
    """
    This function takes in true labels and predicted labels as NumPy arrays and plots the corresponding confusion matrix using Seaborn.
    Parameters:
        cm: the confusion matrix
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


df = pd.read_csv('Training_Dataset.csv')
print('Got CSV')

df = pd.DataFrame((df['having_IP_Address'], df['URL_Length'], df['Shortening_Service'],
                   df['having_At_Symbol'], df['double_slash_redirecting'], df['Prefix_Suffix'],
                   df['having_Sub_Domain'], df['SSLfinal_State'], df['Domain_registration_length'],
                   df['Favicon'], df['HTTPS_token'], df['Request_URL'], df['URL_of_Anchor'], df['Links_in_tags'],
                   df['Redirect'], df['on_mouseover'], df['RightClick'], df['popUpWindow'], df['Iframe'],
                   df['age_of_domain'], df['DNSRecord'], df['Result'])).T

# Split the dataset into features and target
x = np.array(df.iloc[:, :-1])  # data
y = np.array(df.iloc[:, -1])  # result

# Classify
y1 = y * 0
y1[np.argwhere(y == 1)] = 1
y2 = y * 0
y2[np.argwhere(y == -1)] = 1
y_new = np.array([y1, y2]).T

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y_new, test_size=0.2, random_state=1)

model = tf.models.Sequential()
model.add(tf.layers.Dense(units=2, activation='sigmoid'))  # output - 1 attribute
model.add(tf.layers.Dense(units=2, activation='softmax'))  # output - 1 attribute

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=500, validation_split=0.2, verbose=0)

example_features = CollectData()[URL]
# example_features = [random.randint(-1, 1) for i in range(21)]
if example_features is None:
    quit(print('DATA EXTRACTION FAILED'))

# Predict on the test set
y_pred = np.round(model.predict(x_test))

# Reshape example features into a 2D numpy array
example_features_array = np.array(example_features).reshape(1, -1)
state = np.array([0, 1])

# Evaluate the model
cm = confusion_matrix(np.dot(y_test, state), np.dot(y_pred, state))
print('Confusion Matrix:\n', cm)

# Loss
print(f'Loss: {history.history["loss"][-1]}')
loss = history.history['loss']

# MSE
print('MSE:', mse(y_test, y_pred))

# Predict the class (malicious or not) of the example URL
prediction = model.predict(example_features_array)
prediction = np.dot(prediction, state)[0]
print(f"Prediction: {prediction}")
if prediction > 0.5:
    print("The Features List: ", example_features)
    print("The example URL is predicted to be non-malicious.")
else:
    print("The Features List: ", example_features)
    print("The example URL is predicted to be malicious.")

# Plot
plt.plot(loss)
plot_confusion_matrix(cm)
