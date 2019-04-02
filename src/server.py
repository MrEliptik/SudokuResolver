import json
import os
from bottle import route, run, static_file, request, response
import sudoku
from keras.models import load_model

currentVersion = '0.1'
model_path = '../ressources/models/custom_w_altered_keras_cnn_model.h5'
# Global variable for the keras model
model = None

def cors(func):
    def wrapper(*args, **kwargs):
        response.set_header("Access-Control-Allow-Origin", "*")
        response.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        response.set_header("Access-Control-Allow-Headers", "Origin, Content-Type")
        
        # skip the function if it is not needed
        if request.method == 'OPTIONS':
            return

        return func(*args, **kwargs)
    return wrapper

def initiliaze():
    global model
    model = load_model(model_path)

@route('/')
@cors
def docs():
    return 'Under construction..'

@route('/solve', method='POST')
@cors
def solve():
    global model
    file     = request.files.get('upload')
    name, ext = os.path.splitext(file.filename)
    if ext not in ('.png','.jpg','.jpeg'):
        result = {}
        result['success'] = False;
        result['error_msg'] = "File type not supported!";
        result['grid'] = '';
        json_result = json.dumps(result)
        return json_result
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
    run(host='192.168.0.21', port=9092)
