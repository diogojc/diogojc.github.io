from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np


def getImageForVGG(path, h, w):
    img = image.load_img(path, target_size=(h, w))
    arr = image.img_to_array(img)
    arre = np.expand_dims(arr, axis=0)
    pp = preprocess_input(arre)
    return pp


def getRandomImageForVGG(h, w):
    MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
    arre = (np.random.rand(1, h, w, 3)*255),
    pp = arre[0] - MEAN_VALUE
    return pp.astype(np.float32)


def addNoise(image, sigma=30):
    m, h, w, c = image.shape
    noise = np.random.normal(loc=sigma/2, scale=sigma, size=[m, h, w, c])
    imageWithNoise = image + noise
    return np.clip(imageWithNoise, -123, 152).astype(np.float32)
