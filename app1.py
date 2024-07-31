import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

model = load_model('detection.h5')
labels = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'butter',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'cheese',
 'chilli pepper',
 'cocacola',
 'corn',
 'cream',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'ice cream',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'milk',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']
cam = cv2.VideoCapture(0)

objects = []

while True:
    ret , frame = cam.read()
    #img = cv2.rectangle(frame,(20,80),(550,350),(0,0,0),2)
    image = cv2.resize(image, (180, 180), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(0, 180, 180, 3)
    cv2.imshow('frame',image)
    score =tf.nn.softmax(model.predict(image)) 
    objects.append(labels[np.argmax(score)])

    if cv2.waitKey(1) == ord('q'):
        break
print(objects)
cam.release()
cv2.destroyAllWindows()