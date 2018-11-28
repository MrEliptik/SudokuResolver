# SudokuResolver
A computer vision python program to resolve sudoku taken from a camera

# File structure
- **data**        : Contains 10 folders, one for each digit (1167 examples)
- **examples**    : Contains images at different stage of the processing
- **ressources**  :
  - **img**               : Input image(s) to test
  - **models**            : Different Keras CNN models for digit classification
- **src**         : All the python scripts
  - *extracter.py*      : The entry point, where all the magic happens
  - *solver.py*         : Algorithm(s) to solve the sudoku
  - *alterateImages.py* : Function(s) to alterate images (crop, skew, rotate)
  - *addBadImages.py*   : Add 150 alterated images to each digit class
  - *buildDataset.py*   : Is used by buildMnist.py to create a dataset in memory
  - *buildMnist.py*     : Create, train, test and save a CNN model

# Steps
![alt text](https://raw.githubusercontent.com/MrEliptik/SudokuResolver/master/examples/original.png)
![alt text](https://raw.githubusercontent.com/MrEliptik/SudokuResolver/master/examples/blurred.png)
![alt text](https://raw.githubusercontent.com/MrEliptik/SudokuResolver/master/examples/adaptive_treshold.png)
![alt text](https://raw.githubusercontent.com/MrEliptik/SudokuResolver/master/examples/dilated.png)
![alt text](https://raw.githubusercontent.com/MrEliptik/SudokuResolver/master/examples/contours.png)
![alt text](https://raw.githubusercontent.com/MrEliptik/SudokuResolver/master/examples/corners.png)
![alt text](https://raw.githubusercontent.com/MrEliptik/SudokuResolver/master/examples/cropped_perspective_fixed.png)
![alt text](https://raw.githubusercontent.com/MrEliptik/SudokuResolver/master/examples/grid_applied.png)
![alt text](https://raw.githubusercontent.com/MrEliptik/SudokuResolver/master/examples/result.png)

# TODO 
- Re-train CNN with MNIST data (hand writtten digit) & test perfomance
- Add a web interface
- Use an RCNN to localize digits
