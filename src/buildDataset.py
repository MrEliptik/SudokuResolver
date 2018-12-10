import cv2 as cv
import numpy as np
from random import randint
from sklearn.utils import shuffle

nb_classes = 10
nb_objects_per_class = 1167
nb_objects = nb_classes*nb_objects_per_class

def read_data():
    # X data
    x = np.empty(shape=(nb_objects, 28, 28, 1))
    y = np.empty(nb_objects)

    # Go through the samples classes
    for i in range(nb_classes):
        # Go through the sample of that class
        for j in range(nb_objects_per_class):
            im = cv.imread('data/Sample' + format(i+1, '03d') 
                            + '/img' + format(i+1, '03d') + '-' 
                            + format(j+1, '05d') + '.png')
            imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
            res = cv.resize(imgray.copy(), (28, 28), interpolation = cv.INTER_AREA)
            res = res.astype(dtype="float32")
            res = np.reshape(res, (28, 28, 1))
            x[(i*nb_objects_per_class)+j][:][:][:] = res
            y[(i*nb_objects_per_class)+j] = i
    return x, y

def load_data():
   x,y = read_data()
   x,y = shuffle(x, y, random_state=0)
   x_train, y_train, x_test, y_test = split_data(x,y)
   return x_train, y_train, x_test, y_test

def split_data(x,y,split=0.85):
    maxIndex = int(split*x.shape[0])
    x_train = x[:maxIndex][:][:][:]
    x_test = x[maxIndex:][:][:][:]
    y_train = y[:maxIndex]
    y_test = y[maxIndex:]
    return x_train, y_train, x_test, y_test

def main():
    x_train, y_train, x_test, y_test = load_data()
    print(y_train.shape, y_test.shape)
    print(x_train[-3][:][:][:])

if __name__ == "__main__":
    main()
