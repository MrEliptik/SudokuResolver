import cv2 as cv
import alterateImages as alt
from random import randint

# Go through the samples classes
for i in range(10):
    # Add 150 alterated examples
    for j in range(151):
        index = randint(1, 1016)
        im = cv.imread('cnn/data/Sample' + format(i+1, '03d')
                        + '/img' + format(i+1, '03d') + '-'
                        + format(index, '05d') + '.png')
        res = alt.random_crop(im)
        cv.imwrite('cnn/data/Sample' + format(i+1, '03d')
                   + '/img' + format(i+1, '03d') + '-'
                   + format(1017+j, '05d') + '.png', res)
