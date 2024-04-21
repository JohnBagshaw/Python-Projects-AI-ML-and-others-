import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('C:/Users/HP/Downloads/Larger_GNSS_Signal_Dataset.csv')

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
    Dense(256, activation='relu', input_shape=input_shape, kernel_regularizer='l2'),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_regularizer='l2'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping monitor
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=100)

# Model training
history = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=1, callbacks=[early_stopping_monitor])

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
