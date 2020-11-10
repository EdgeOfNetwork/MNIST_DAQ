import tensorflow as tf
import cv2
import numpy as np
import math


def process(img_input):
    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    (thresh, img_binary) = cv2.threshold(gray, 0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    h, w = img_binary.shape

    ratio = 100 / h
    new_h = 100
    new_w = w * ratio

    img_empty = np.zeros((110, 110), dtype=img_binary.dtype)
    img_binary = cv2.resize(img_binary, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)
    img_empty[:img_binary.shape[0], :img_binary.shape[1]] = img_binary

    img_binary = img_empty

    cnts = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어의 무게중심 좌표를 구한다
    M = cv2.moments(cnts[0][0])
    center_x = (M["m10"] / M["m00"])
    center_y = (M["m01"] / M["m00"])

    # 무게 중심이 이미지 중심으로 오도록 이동
    height, width = img_binary.shape[:2]
    shift_x = width / 2 - center_x
    shift_y = height / 2 - center_y

    translation_matrix = np.float32([1, 0, shift_x], [0, 1, shift_y])
    img_binary = cv2.warpAffine(img_binary, translation_matrix, (width, height))

    img_binary = cv2.resize(img_binary, (28, 28), interpolation=cv2.INTER_AREA)
    flatten = img_binary.flatten() / 255.0

    return flatten


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(512, activation=tf.nn.relu),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.load_weights('mnist_checkpoint')

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:

    ret, img_color = cap.read()

    if not ret:
        break

    img_input = img_color.copy()
    cv2.rectangle(img_color, (250, 150), (width - 250, height - 150), (0, 0, 255), 3)
    cv2.imshow('bgr', img_color)

    img_roi = img_input[150:height - 150, 250:width - 250]

    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == 32:
        flatten = process(img_roi)

        predictions = model.predict(flatten[np.newaxis, :])

        cv2.imshow('img_roi', img_roi)
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
