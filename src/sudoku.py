import extracter
import solver
import numpy as np
import cv2 as cv
import json

def solve(im):
    im = cv.imdecode(np.fromstring(im.read(), np.uint8), 1)

    # Pre process (remove noise, treshold image, get contours)
    pre_processed = extracter.preProcess(im)

    # Find corners based on the biggest contour
    corners = extracter.findCorners(pre_processed)

    # Fix the perspective based on the 4 corners found
    sudoku = extracter.fixPerspective(pre_processed, corners)

    # Match a grid to the image
    squares = extracter.grid(sudoku)

    # Convert to int
    squares_int = [[tuple(int(x) for x in tup) for tup in square] for square in squares]

    im_squares = cv.cvtColor(sudoku.copy(), cv.COLOR_GRAY2RGB)

    # Get individual cells 
    cells = extracter.extractCells(sudoku, squares_int)

    # Get each digit 
    extractedCells = []
    for cell in cells:
        h, w = cell.shape[:2]
        margin = int(np.mean([h, w]) / 2.5)
        _, bbox, seed = extracter.findLargestConnectedComponent(cell, [margin, margin], [w - margin, h - margin])
        extractedCells.append(extracter.extractDigit(cell, bbox, 28))

    # Get matrix by using CNN to recognize digits
    sudoku = extracter.readSudoku(extractedCells)
    print("Extracted sudoku :")
    print(sudoku)

    # Resolve sudoku
    resolved = sudoku.copy()
    solver.solveSudoku(resolved)

    print("Resolved: ")
    print(resolved)

    # JSONify
    result = {}
    result['success'] = True;
    result['grid'] = resolved.tolist();
    json_result = json.dumps(result)

    return json_result
