import json
import os
from bottle import route, run, static_file, request
import sudoku
from keras.models import load_model

currentVersion = '0.1'
model_path = '../ressources/models/custom_w_altered_keras_cnn_model.h5'
# Global variable for the keras model
model = None

def initiliaze():
    global model
    model = load_model(model_path)

@route('/')
def docs():
    return 'Under construction..'

@route('/solve', method='POST')
def solve():
    global model
    file     = request.files.get('upload')
    name, ext = os.path.splitext(file.filename)
    if ext not in ('.png','.jpg','.jpeg'):
        return 'File extension not allowed.'
    '''
    To read the JSON array:
    b_new = json.loads(obj_text)
    a_new = np.array(b_new)
    '''
    try:
        return sudoku.solve(model, file.file)
    except expression as identifier:
        return "Error when trying to read sudoku.."   
    

@route('/version')
def version():
    return 'Current version: ' + currentVersion

if __name__ == "__main__":
    initiliaze()
    run(host='localhost', port=8080, debug=True)