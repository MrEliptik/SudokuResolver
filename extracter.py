import cv2 as cv
import numpy as np
import imutils
import operator

def preProcess(im):
    # Convert to grayscale
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    # Light blur to reduce noise before thresholding
    blurred = cv.GaussianBlur(imgray, (5, 5), 0)
    cv.imshow('blurred', blurred)

    # Binary threshold (poor results)
    #ret, thresh = cv.threshold(blurred, 127, 255, 0)
    #cv.imshow('tresh', thresh)

    # Adaptive threshold    - THRESH_BINARY_INV to inverse color
    #                       - Use 3 as blockSize because noise points are small
    adaptive_thresh = cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                                            cv.THRESH_BINARY_INV, 3, 2)
    cv.imshow('adaptive tresh', adaptive_thresh)

    # Create a cross kernel
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)

    # Dilate to connect the grid correctly
    dilated = cv.dilate(adaptive_thresh, kernel, iterations = 1)
    cv.imshow('dilated', dilated)

    cv.waitKey(0)
    cv.destroyAllWindows()

    return dilated
    
def findCorners(im):
    # Find the external contours
    im2, ext, hierarchy = cv.findContours(im.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Put back the image in color to display contours
    im = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    ext_contours = cv.drawContours(im, ext, -1, (0,0,255), 2)
    cv.imshow('ext', ext_contours)

    # Sort the contours by area, descending order
    contours = sorted(ext, key=cv.contourArea, reverse=True)
    # First one is the largest
    polygon = contours[0]  

    # Find the 4 points defining the grid
    # Bottom-right point has the largest (x + y) value
	# Top-left has point smallest (x + y) value
	# Bottom-left point has smallest (x - y) value
	# Top-right point has largest (x - y) value
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _     = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _  = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _    = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    cv.waitKey(0)
    cv.destroyAllWindows()

    # Return an array of all 4 points using the indices
	# Each point is in its own array of one coordinate
    return [tuple(polygon[top_left][0]), tuple(polygon[top_right][0]), tuple(polygon[bottom_right][0]), tuple(polygon[bottom_left][0])]

def main():
    # Load image
    im = cv.imread('sudoku_photo.jfif')
    cv.imshow('original', im)
    pre_processed = preProcess(im)
    corners = findCorners(pre_processed)
    for corner in corners:
        print(corner)
        cv.circle(im, corner, 3, (0,0,255), -1)
    cv.imshow('Corners', im)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()