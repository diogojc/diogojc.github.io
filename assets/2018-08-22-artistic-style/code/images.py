from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np


def getImageForVGG(path):
    img = image.load_img(path, target_size=(400, 530))
    arr = image.img_to_array(img)
    arre = np.expand_dims(arr, axis=0)
    pp = preprocess_input(arre)
    return pp


def getRandomImageForVGG():
    MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
    arre = (np.random.rand(1, 400, 530, 3)*255),
    pp = arre[0] - MEAN_VALUE
    # https://stackoverflow.com/questions/46545986/how-to-use-tf-clip-by-value-on-sliced-tensor-in-tensorflow
    # return tf.Variable(pp.astype(np.float32), constraint=lambda x: tf.clip_by_value(x, -123, 152))
    return pp.astype(np.float32)


def addNoise(image, sigma=30):
    m, h, w, c = image.shape
    noise = np.random.normal(loc=sigma/2, scale=sigma, size=[m, h, w, c])
    imageWithNoise = image + noise
    return np.clip(imageWithNoise, -123, 152).astype(np.float32)
