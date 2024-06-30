
import cv2
import numpy as np
from tensorflow.keras.models import load_model
model = load_model('C:/Users/sooda/Desktop/dev/New folder/Mishka-tech-internship/mnist1.h5')

import numpy as np


def prediction(img_grey, model):
    # Check the shape of the original array
    original_shape = img_grey.shape
    print(f"Original shape: {original_shape}")

    # Reshape the array into a compatible shape
    if original_shape != (28, 28):
        raise ValueError("Invalid shape for img_grey. Expected (28, 28)")

    # Normalize the pixel values
    img = img_grey / 255.0

    # Reshape the array into a compatible shape
    img = img.reshape((1, 784))

    # Make a prediction using the model
    predict = model.predict(img)

    # Get the maximum probability
    prob = np.amax(predict)

    # Get the index of the maximum probability
    result = np.argmax(predict)

    return result, prob
# Initialize the camera
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

# Get the frame dimensions
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the bounding box size
bbox_size = (28, 28)

while True:
    _, frame = cap.read()

    # Create a copy of the original frame
    frame_copy = frame.copy()

    # Calculate the bounding box coordinates
    x1 = int(WIDTH // 2 - bbox_size[0] // 2)
    y1 = int(HEIGHT // 2 - bbox_size[1] // 2)
    x2 = int(WIDTH // 2 + bbox_size[0] // 2)
    y2 = int(HEIGHT // 2 + bbox_size[1] // 2)
    bbox = [(x1, y1), (x2, y2)]

    # Crop the region of interest
    img_c = frame[y1:y2, x1:x2]

    # Convert the cropped image to grayscale
    img_grey = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

    # Display the original and cropped frames
    cv2.imshow("cropped", img_grey)
    result , probability = prediction(img_grey, model)
    cv2.putText(frame_copy, f"prediction: {result}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255),2 ,cv2.LINE_AA)
    cv2.putText(frame_copy, "probability: "+"{:.2f}".format(probability), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255),2 ,cv2.LINE_AA, )
    cv2.rectangle(frame_copy, bbox[0],bbox[1],(0, 255, 0), 3)
    cv2.imshow("input", frame_copy)


    # Exit on pressing the 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cv2.destroyAllWindows()
cap.release()