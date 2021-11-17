import os
from flask import Flask,request
from Display import Display
import cv2
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/upload/'
@app.route('/')
def hello_world():
    return 'Hello World!'
@app.route('/api/v1/upload',methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        input = request.form.get('input')
        output = request.form.get('output')
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], f.filename), cv2.IMREAD_UNCHANGED)
        jpg_img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], f.filename), cv2.IMREAD_COLOR)
        name, _ = f.filename.split('.')
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], name)+'.png', jpg_img)
        print('success')
        path_list = display.run(img, int(input), int(output), os.path.join(app.config['UPLOAD_FOLDER'], name)+'.png')
        print(input)
        print(output)
        return path_list
    elif request.method == 'GET':

        return 'success'
    else:
        return 'error'
if __name__ == '__main__':
    path = './models/07-20-15:34_RSIR/63.pth.tar'
    display = Display(path)
    app.run(host='0.0.0.0',port=5200, debug=True)