import cv2 as cv
import numpy as np
import imutils
import operator
import math
from keras.models import load_model
import solver

def preProcess(im):
    # Convert to grayscale
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    # Light blur to reduce noise before thresholding
    blurred = cv.GaussianBlur(imgray, (5, 5), 0)
    #cv.imshow('blurred', blurred)

    # Binary threshold (poor results)
    #ret, thresh = cv.threshold(blurred, 127, 255, 0)
    #cv.imshow('tresh', thresh)

    # Adaptive threshold    - THRESH_BINARY_INV to inverse color
    #                       - Use 3 as blockSize because noise points are small
    adaptive_thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_MEAN_C,\
                                            cv.THRESH_BINARY_INV, 3, 2)
    #cv.imshow('adaptive tresh', adaptive_thresh)

    # Create a cross kernel
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)

    # Dilate to connect the grid correctly
    dilated = cv.dilate(adaptive_thresh, kernel, iterations = 1)
    #cv.imshow('dilated', dilated)

    #cv.waitKey(0)
    #cv.destroyAllWindows()

    return dilated
    
def findCorners(im):
    # Find the external contours
    im2, ext, hierarchy = cv.findContours(im.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Put back the image in color to display contours
    im = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    ext_contours = cv.drawContours(im, ext, -1, (0,0,255), 2)
    #cv.imshow('ext', ext_contours)

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

    #cv.waitKey(0)
    #cv.destroyAllWindows()

    # Return an array of all 4 points using the indices
	# Each point is in its own array of one coordinate
    return [tuple(polygon[top_left][0]), tuple(polygon[top_right][0]), tuple(polygon[bottom_right][0]), tuple(polygon[bottom_left][0])]

def fixPerspective(im, corners):
    # Explicitly set the data type to float32 or 
    # `getPerspectiveTransform` will throw an error
	src = np.array([corners[0], corners[1], corners[2], corners[3]], dtype='float32')

    # Get the longest side in the rectangle
	side = max([
            distance(corners[2], corners[1]),
            distance(corners[0], corners[3]),
            distance(corners[2], corners[3]),
            distance(corners[0], corners[1])
	    ])
    # Describe a square with side of the calculated length, 
    # this is the new perspective we want to warp to
	dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32') 

    # Gets the transformation matrix for skewing the image to 
    # fit a square by comparing the 4 before and after points
	m = cv.getPerspectiveTransform(src, dst)

    # Performs the transformation on the original image
	return cv.warpPerspective(im, m, (int(side), int(side)))

def grid(im, size=9):
    squares = []
    side = im.shape[:1][0]
    side = side / size
    for i in range(9):
        for j in range(9):
            p1 = (i * side, j * side)  # Top left corner of a bounding box
            p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
            squares.append((p1, p2))
    return squares

def extractCells(im, squares):
    cells = []
    for i, square in enumerate(squares):
        cell = im[square[0][1]:square[1][1], square[0][0]:square[1][0]]
        cells.append(cell)
    return cells

def findLargestConnectedComponent(inp_img, scan_tl=None, scan_br=None):
    im = inp_img.copy()  # Copy the image, leaving the original untouched
    height, width = im.shape[:2]

    max_area = 0
    seed_point = (None, None)

    if scan_tl is None:
        scan_tl = [0, 0]

    if scan_br is None:
        scan_br = [width, height]

    # Loop through the image
    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            # Only operate on light or white squares
            # Note that .item() appears to take input as y, x
            if(im.item(y, x) == 255 and x < width and y < height):  
                area = cv.floodFill(im, None, (x, y), 64)
                if(area[0] > max_area):  # Gets the maximum bound area which should be the grid
                    max_area = area[0]
                    seed_point = (x, y)

    # Colour everything grey (compensates for features outside of our middle scanning range
    for x in range(width):
        for y in range(height):
            if im.item(y, x) == 255 and x < width and y < height:
                cv.floodFill(im, None, (x, y), 64)

    mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

    # Highlight the main feature
    if all([p is not None for p in seed_point]):
        cv.floodFill(im, mask, seed_point, 255)

    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):
            if im.item(y, x) == 64:  # Hide anything that isn't the main feature
                cv.floodFill(im, mask, (x, y), 0)

            # Find the bounding parameters
            if im.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bbox = [[left, top], [right, bottom]]
    return im, np.array(bbox, dtype='float32'), seed_point

def extractDigit(cell, bbox, size):
    cell = cell[int(bbox[0][1]):int(bbox[1][1]), int(bbox[0][0]):int(bbox[1][0])]

    # Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    # Ignore any small bounding boxes
    if w > 0 and h > 0 and (w * h) > 100 and len(cell) > 0:
        return scale_and_centre(cell, size, 4)
    else:
        return np.zeros((size, size), np.uint8)

def scale_and_centre(img, size, margin=0, background=0):
	"""Scales and centres an image onto a new background square."""
	h, w = img.shape[:2]

	def centre_pad(length):
		"""Handles centering for a given length that may be odd or even."""
		if length % 2 == 0:
			side1 = int((size - length) / 2)
			side2 = side1
		else:
			side1 = int((size - length) / 2)
			side2 = side1 + 1
		return side1, side2

	def scale(r, x):
		return int(r * x)

	if h > w:
		t_pad = int(margin / 2)
		b_pad = t_pad
		ratio = (size - margin) / h
		w, h = scale(ratio, w), scale(ratio, h)
		l_pad, r_pad = centre_pad(w)
	else:
		l_pad = int(margin / 2)
		r_pad = l_pad
		ratio = (size - margin) / w
		w, h = scale(ratio, w), scale(ratio, h)
		t_pad, b_pad = centre_pad(h)

	img = cv.resize(img, (w, h))
	img = cv.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv.BORDER_CONSTANT, None, background)
	return cv.resize(img, (size, size))

def readSudoku(cells):
    sudoku_matrix = np.full((1, 81), -1)

    # Load trained model
    #model = load_model('ressources/models/mnist_keras_cnn_model.h5')
    #model = load_model('ressources/models/custom_keras_cnn_model.h5')
    model = load_model('ressources/models/custom_w_altered_keras_cnn_model.h5')

    for i, cell in enumerate(cells):
        if(np.allclose(cell, 0)):
            sudoku_matrix[0][i] = 0
        else:
            # Erode
            # Create a cross kernel
            kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
            res = cv.erode(cell.copy(), kernel, iterations=1)
            res = cv.bitwise_not(res)

            # Resize and reshape image
            dim = (28, 28)
            res = cv.resize(res, dim, interpolation=cv.INTER_AREA)
            res = np.reshape(res, (1, 28, 28, 1))

            # Convert to float values between 0. and 1.
            res = res.astype(dtype="float32")
            if(res.max() != 0.):
                res /= 255

            # Use model to predict
            nb = model.predict(res)

            for j in range(nb.shape[0]):
                for k in range(nb.shape[1]):
                    if(nb[j][k] > 0.9):
                        sudoku_matrix[0][i] = int(k)

    sudoku_matrix = sudoku_matrix.reshape((9, 9))
    return sudoku_matrix.T

def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def main():
    # Load image
    im = cv.imread('ressources/img/sudoku_photo.jfif')
    cv.imshow('original', im)

    # Pre process (remove noise, treshold image, get contours)
    pre_processed = preProcess(im)

    # Find corners based on the biggest contour
    corners = findCorners(pre_processed)
    im_corners = im.copy()
    for corner in corners:
        cv.circle(im_corners, corner, 3, (0,0,255), -1)
    #cv.imshow('Corners', im_corners)

    # Fix the perspective based on the 4 corners found
    sudoku = fixPerspective(pre_processed, corners)
    cv.imshow('Cropped and fixed', sudoku)

    # Match a grid to the image
    squares = grid(sudoku)

    # Convert to int
    squares_int = [[tuple(int(x) for x in tup) for tup in square] for square in squares]

    im_squares = cv.cvtColor(sudoku.copy(), cv.COLOR_GRAY2RGB)
    # Draw squares
    for square in squares_int:
        cv.rectangle(im_squares,square[0],square[1],(0,255,0),1)
    cv.imshow('Grid applied', im_squares)

    # Get individual cells 
    cells = extractCells(sudoku, squares_int)

    # Get each digit 
    extractedCells = []
    for cell in cells:
        h, w = cell.shape[:2]
        margin = int(np.mean([h, w]) / 2.5)
        _, bbox, seed = findLargestConnectedComponent(cell, [margin, margin], [w - margin, h - margin])
        extractedCells.append(extractDigit(cell, bbox, 28))

    columns = []
    # Recreate a grid with all the extracted cells and digits
    with_border = [cv.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv.BORDER_CONSTANT, None, 255) for img in extractedCells]
    for i in range(9):
        column = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=0)
        columns.append(column)
    res = np.concatenate(columns, axis=1)

    cv.imshow("Res", res)

    # Get matrix by using CNN to recognize digits
    sudoku = readSudoku(extractedCells)
    print("Extracted sudoku :")
    print(sudoku)

    # Resolve sudoku
    resolved = sudoku.copy()
    solver.solveSudoku(resolved)
    print("Solved sudoku :")
    print(resolved)

    cv.waitKey(0)  
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
