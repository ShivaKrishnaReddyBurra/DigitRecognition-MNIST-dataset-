import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import cv2

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Convert labels to categorical (one-hot encoding)
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# Save the model
model.save('digit_classifier_model.keras')

# Function to predict a digit from a new image
# def predict_digit(image_path):
#     # Load the image
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     # Resize the image to 28x28 pixels
#     img = cv2.resize(img, (28, 28))
#     # Normalize the image
#     img = img.astype('float32') / 255
#     # Reshape the image to match the input shape of the model
#     img = img.reshape((1, 28, 28, 1))
#     # Load the trained model
#     model = tf.keras.models.load_model('digit_classifier_model.keras')
#     # Predict the digit
#     prediction = model.predict(img)
#     # Get the digit with the highest probability
#     digit = np.argmax(prediction)
#     return digit

# # Example usage
# # print(predict_digit('path_to_handwritten_digit_image.png'))

# print(predict_digit('./ml_test_data/5.png'))
# print(predict_digit('./ml_test_data/9.jpg'))
