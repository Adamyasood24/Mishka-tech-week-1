# Function to preprocess the image
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the MNIST model
model = load_model('C:/Users/sooda/Desktop/dev/New folder/Mishka-tech-internship/mnist.h5')

def preprocess_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to the image


    # Resize the image to 28x28 (MNIST input size)
    image = cv2.resize(image, (28, 28))

    #image = cv2.GaussianBlur(image, (5, 5), 0)

    # Normalize the image pixels to be between 0 and 1
    #image = image / 255.0

    # Reshape the image to match the model's input shape
    image = image.reshape((1, 784))

    return image

# Get the image path from the user
image_path = input("Enter the path to the image file: ")

# Preprocess the image
image = preprocess_image(image_path)

# Make predictions using the model
predictions = model.predict(image)

# Get the predicted digit
predicted_digit = np.argmax(predictions)

print("Predicted digit:", predicted_digit)