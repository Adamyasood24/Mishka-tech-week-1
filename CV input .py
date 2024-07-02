import cv2
import numpy as np
from tensorflow.keras.models import load_model

TF_ENABLE_ONEDNN_OPTS = 0
model = load_model('C:/Users/sooda/Desktop/dev/New folder/Mishka-tech-internship/mnist1.h5')

import numpy as np


def prediction(img_grey, model):
    img_grey = cv2.resize(img_grey, (28, 28), interpolation=cv2.INTER_LINEAR)

    original_shape = img_grey.shape

    img = cv2.bitwise_not(img_grey)

    img = img_grey / 255.0

    img = img.reshape((1, 784))

    predict = model.predict(img)

    prob = np.amax(predict)

    result = np.argmax(predict)

    return result, prob


cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

bbox_size = (60, 60)

while True:
    _, frame = cap.read()

    frame_copy = frame.copy()

    x1 = int(WIDTH // 2 - bbox_size[0] // 2)
    y1 = int(HEIGHT // 2 - bbox_size[1] // 2)
    x2 = int(WIDTH // 2 + bbox_size[0] // 2)
    y2 = int(HEIGHT // 2 + bbox_size[1] // 2)
    bbox = [(x1, y1), (x2, y2)]

    img_c = frame[y1:y2, x1:x2]

    img_grey = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

    cv2.imshow("cropped", img_grey)
    result, probability = prediction(img_grey, model)
    cv2.putText(frame_copy, f"prediction: {result}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2,
                cv2.LINE_AA)
    cv2.putText(frame_copy, "probability: " + "{:.2f}".format(probability), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 0, 255), 2, cv2.LINE_AA, )
    cv2.rectangle(frame_copy, bbox[0], bbox[1], (0, 255, 0), 3)
    cv2.imshow("input", frame_copy)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()
