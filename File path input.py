import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the MNIST model
model = load_model('C:/Users/sooda/Desktop/dev/New folder/Mishka-tech-internship/mnist.h5')

def preprocess_image(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    image = cv2.bitwise_not(image)

    image = cv2.resize(image, (28,28), interpolation=cv2.INTER_LINEAR)

    image = image / 255.0

    image = image.reshape((1, 784))

    return image

image_path = input("Enter the path to the image file: ")


image = preprocess_image(image_path)

predictions = model.predict(image)

predicted_digit = np.argmax(predictions)

print("Predicted digit:", predicted_digit)