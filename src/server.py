import json
import os
from bottle import route, run, static_file, request
import sudoku

currentVersion = '0.1'

@route('/')
def docs():
    return 'Under construction..'

@route('/solve', method='POST')
def solve():
    file     = request.files.get('upload')
    name, ext = os.path.splitext(file.filename)
    if ext not in ('.png','.jpg','.jpeg'):
        return 'File extension not allowed.'
    print(file.name)
    resolved = sudoku.solve(file.file)
    return str(resolved)

@route('/version')
def version():
    return 'Current version: ' + currentVersion


run(host='localhost', port=8080, debug=True)