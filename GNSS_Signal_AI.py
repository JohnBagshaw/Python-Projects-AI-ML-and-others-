//By John Bagshaw

import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('C:/Users/HP/Downloads/GNSS_Signal_Dataset.csv')

# Simulating labels for the dataset (0=normal, 1=spoofed, 2=jammed)
labels = np.random.choice([0, 1, 2], len(df))

# Features and labels preparation
x_train = df.values
y_train = to_categorical(labels)

# Model configuration
input_shape = x_train.shape[1:]
num_classes = 3

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=input_shape),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Visualization of model performance
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
