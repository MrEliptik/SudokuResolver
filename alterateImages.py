import cv2 as cv
from random import randint

def random_crop(im):
    (h, w, chan) = im.shape

    for i in range(11):
        percentage = randint(70, 85)/100.0
        case = randint(0, 3)

        if(case == 0):
            crop_img = im[0:int(h*percentage), 0:int(w*percentage)]
        elif(case == 1):
            crop_img = im[h-int(h*percentage):h, w-int(w*percentage):w]
        elif(case == 2):
            crop_img = im[0:int(h*percentage), w-int(w*percentage):w]
        elif(case == 3):
            crop_img = im[h-int(h*percentage):h, 0:int(w*percentage)]
        res = cv.resize(crop_img, (128, 128), interpolation=cv.INTER_AREA)
    return res

'''
im = cv.imread('cnn/data/Sample005/img005-01013.png')
im = random_crop(im)

res = cv.resize(im, (28, 28), interpolation=cv.INTER_AREA)
cv.imshow("res", res)
cv.waitKey(0)
'''
